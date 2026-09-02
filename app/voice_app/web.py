from __future__ import annotations

import asyncio
import json
import queue
import shutil
import struct
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import AppSettings
from .conversation import (
    InMemoryConversationHistory,
    TurnState,
)
from .media_v10 import V10MediaPipeline
from .orchestrator import VoiceOrchestrator


PCM_WS_MAGIC = b"V10P"
PCM_WS_VERSION = 1
PCM_WS_HEADER = struct.Struct("<4sHHIIIIQQQQQQ")


@dataclass
class AudioPacket:
    raw: bytes
    meta: dict[str, Any]


def encode_pcm_websocket_packet(packet: AudioPacket, *, sent_ns: int | None = None) -> bytes:
    """Encode one ordered PCM16 chunk with its end-to-end trace header."""
    meta = packet.meta
    t5_ns = sent_ns or time.time_ns()
    meta["t5_websocket_send_ns"] = t5_ns
    header = PCM_WS_HEADER.pack(
        PCM_WS_MAGIC,
        PCM_WS_VERSION,
        PCM_WS_HEADER.size,
        int(meta.get("chunk_index") or 0),
        int(meta.get("unit_index") or 0),
        int(meta.get("unit_chunk_index") or 0),
        int(meta.get("frame_count") or len(packet.raw) // 2),
        int(meta.get("t0_model_yield_ns") or 0),
        int(meta.get("t1_service_yield_ns") or 0),
        int(meta.get("t2_client_receive_ns") or 0),
        int(meta.get("t3_media_enqueue_ns") or 0),
        int(meta.get("t4_web_audio_enqueue_ns") or 0),
        t5_ns,
    )
    return header + packet.raw


@dataclass
class TurnSession:
    turn_id: str
    conversation_id: str
    generation: int
    created_at: float
    event_queue: queue.Queue[dict[str, Any]] = field(default_factory=lambda: queue.Queue(maxsize=2048))
    audio_queue: queue.Queue[AudioPacket | None] = field(default_factory=queue.Queue)
    finished: threading.Event = field(default_factory=threading.Event)
    final_audio: str | None = None
    browser_metrics: dict[str, Any] = field(default_factory=dict)
    temp_audio_dir: Path | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    cancel_reason: str | None = None
    worker: threading.Thread | None = None
    event_log: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=2048))
    event_condition: threading.Condition = field(default_factory=threading.Condition)
    event_sequence: int = 0


class TurnManager:
    def __init__(self, orchestrator: VoiceOrchestrator, history=None) -> None:
        self.orchestrator = orchestrator
        self.pipeline = V10MediaPipeline(orchestrator)
        self.history = history or InMemoryConversationHistory()
        self._turns: dict[str, TurnSession] = {}
        self._lock = threading.Lock()

    def _new(self, conversation_id: str | None = None) -> tuple[TurnSession, list]:
        conversation = self.history.get(conversation_id) if conversation_id else self.history.create()
        history = list(self.history.messages(conversation.conversation_id, self.orchestrator.settings.max_history_turns))
        turn_id = uuid.uuid4().hex
        record = self.history.begin_turn(conversation.conversation_id, turn_id)
        session = TurnSession(
            turn_id=turn_id,
            conversation_id=conversation.conversation_id,
            generation=record.generation,
            created_at=time.time(),
            audio_queue=queue.Queue(maxsize=max(16, self.orchestrator.settings.tts_audio_queue_max * 4)),
        )
        with self._lock:
            # Keep the demo server bounded across long interactive sessions. Only
            # completed turns are evicted; active turn streams are never touched.
            if len(self._turns) >= 64:
                finished = sorted(
                    (item for item in self._turns.values() if item.finished.is_set()),
                    key=lambda item: item.created_at,
                )
                for old in finished[: max(0, len(self._turns) - 31)]:
                    self._turns.pop(old.turn_id, None)
            self._turns[session.turn_id] = session
        return session, history

    def get(self, turn_id: str) -> TurnSession:
        with self._lock:
            session = self._turns.get(turn_id)
        if session is None:
            raise KeyError(turn_id)
        return session

    @staticmethod
    def _pcm_float32_to_int16(raw: bytes) -> bytes:
        if not raw or len(raw) % 4:
            raise ValueError("float32 PCM is empty or not frame-aligned")
        samples = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(samples).all():
            raise ValueError("PCM contains NaN/Inf")
        return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def _emit(self, session: TurnSession, event: dict[str, Any]) -> None:
        if session.cancel_event.is_set() and event.get("type") not in {
            "cancelled", "interrupted", "stream_closed", "error"
        }:
            return
        event.setdefault("turn_id", session.turn_id)
        event.setdefault("conversation_id", session.conversation_id)
        event.setdefault("generation", session.generation)
        with session.event_condition:
            session.event_sequence += 1
            event.setdefault("event_sequence", session.event_sequence)
            session.event_log.append(event)
            session.event_condition.notify_all()
        try:
            session.event_queue.put_nowait(event)
        except queue.Full:
            try:
                session.event_queue.get_nowait()
            except queue.Empty:
                pass
            session.event_queue.put_nowait(event)

    @staticmethod
    def next_event(session: TurnSession, after: int, timeout: float = 15.0) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout
        with session.event_condition:
            while True:
                match = next(
                    (item for item in session.event_log if int(item.get("event_sequence") or 0) > after),
                    None,
                )
                if match is not None:
                    return match
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                session.event_condition.wait(remaining)

    @staticmethod
    def _drain(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    def cancel(self, turn_id: str, *, reason: str = "user_cancelled") -> TurnSession:
        session = self.get(turn_id)
        if session.finished.is_set() or session.cancel_event.is_set():
            return session
        session.cancel_reason = reason
        session.cancel_event.set()
        state = TurnState.INTERRUPTED if reason == "barge_in" else TurnState.CANCELLED
        self.history.transition(session.conversation_id, session.turn_id, state)
        self._drain(session.audio_queue)
        session.audio_queue.put_nowait(None)
        self._drain(session.event_queue)
        with session.event_condition:
            session.event_log.clear()
        self._emit(session, {"type": state.value, "reason": reason, "state": state.value})
        return session

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            active = [
                session for session in self._turns.values()
                if session.conversation_id == conversation_id and not session.finished.is_set()
            ]
        for session in active:
            self.cancel(session.turn_id, reason="conversation_reset")
        return self.history.delete(conversation_id)

    def start(self, *, conversation_id=None, query_text=None, audio_path=None, audio_temp_dir=None, rag_mode=None, rerank=None, top_k=None, voice_index=None):
        session, history = self._new(conversation_id)
        session.temp_audio_dir = audio_temp_dir

        def worker() -> None:
            try:
                for event in self.pipeline.stream_turn(
                    audio_path=audio_path,
                    query_text=query_text,
                    rag_mode=rag_mode,
                    rerank=rerank,
                    top_k=top_k,
                    voice_index=voice_index,
                    history=history,
                    cancel_event=session.cancel_event,
                ):
                    if session.cancel_event.is_set() and event.get("type") not in {"cancelled", "error"}:
                        continue
                    kind = event.get("type")
                    if kind == "stage":
                        state = {
                            "stt": TurnState.TRANSCRIBING,
                            "rag": TurnState.RETRIEVING,
                            "llm": TurnState.THINKING,
                        }.get(event.get("stage"))
                        if state:
                            self.history.transition(session.conversation_id, session.turn_id, state)
                    elif kind == "transcript_ready":
                        self.history.transition(
                            session.conversation_id,
                            session.turn_id,
                            TurnState.RETRIEVING,
                            user_text=str(event.get("transcript") or ""),
                        )
                    elif kind == "prepared":
                        self.history.transition(
                            session.conversation_id,
                            session.turn_id,
                            TurnState.THINKING,
                            user_text=str(event.get("transcript") or ""),
                            sources=list(event.get("sources") or []),
                        )
                    elif kind == "pcm":
                        self.history.transition(session.conversation_id, session.turn_id, TurnState.SPEAKING)
                    elif kind == "done":
                        self.history.transition(
                            session.conversation_id,
                            session.turn_id,
                            TurnState.COMPLETED,
                            user_text=str(event.get("transcript") or ""),
                            assistant_text=str(event.get("answer") or ""),
                            sources=list(event.get("sources") or []),
                        )
                    if event.get("type") == "pcm":
                        pcm16 = self._pcm_float32_to_int16(event["raw"])
                        packet_meta = {k: v for k, v in event.items() if k != "raw"}
                        packet_meta["t4_web_audio_enqueue_ns"] = time.time_ns()
                        packet_meta["web_audio_queue_depth_at_enqueue"] = session.audio_queue.qsize()
                        packet_meta["frame_count"] = len(pcm16) // 2
                        # V10 web boundary is intentionally non-blocking for control:
                        # a slow browser audio reader must not delay WebSocket events.
                        # Turn length is bounded by LLM_MAX_TOKENS, so this per-turn
                        # queue is finite in practice; the TTS/media worker remains
                        # single-threaded and bounded upstream.
                        packet = AudioPacket(pcm16, packet_meta)
                        while not session.cancel_event.is_set():
                            try:
                                session.audio_queue.put(packet, timeout=0.05)
                                break
                            except queue.Full:
                                continue
                        self._emit(session, packet_meta)
                    else:
                        if event.get("type") == "tts_done":
                            session.final_audio = event.get("final_audio")
                        self._emit(session, event)
            except Exception as exc:
                if not session.cancel_event.is_set():
                    self.history.transition(
                        session.conversation_id,
                        session.turn_id,
                        TurnState.FAILED,
                        error=str(exc),
                    )
                    self._emit(session, {"type": "error", "stage": "web-media", "error": str(exc)})
            finally:
                try:
                    session.audio_queue.put_nowait(None)
                except queue.Full:
                    self._drain(session.audio_queue)
                    session.audio_queue.put_nowait(None)
                self._emit(session, {"type": "stream_closed"})
                session.finished.set()
                if session.temp_audio_dir is not None:
                    shutil.rmtree(session.temp_audio_dir, ignore_errors=True)

        session.worker = threading.Thread(target=worker, name=f"v10-turn-{session.turn_id[:8]}", daemon=True)
        session.worker.start()
        return session


class TextTurnRequest(BaseModel):
    query: str = Field(min_length=1)
    conversation_id: str | None = None
    rag_mode: str | None = None
    rerank: bool | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    voice_index: int | None = Field(default=None, ge=0)


class BrowserMetrics(BaseModel):
    first_pcm_received_ms: float | None = None
    first_audio_scheduled_ms: float | None = None
    first_audio_started_estimate_ms: float | None = None
    audio_context_sample_rate: float | None = None
    underrun_count: int = 0
    underrun_frames: int = 0
    rebuffer_count: int = 0
    max_schedule_gap_ms: float = 0.0
    min_buffer_ahead_ms: float | None = None
    max_buffer_ahead_ms: float | None = None
    final_buffer_ahead_ms: float | None = None
    media_transport: str = "binary-websocket-pcm-v1"
    pcm_sequence_errors: int = 0
    max_browser_receive_gap_ms: float = 0.0
    producer_rate: float | None = None
    producer_throughput_status: str | None = None
    stale_packets_dropped: int = 0
    control_reconnect_count: int = 0
    audio_reconnect_count: int = 0
    barge_in_detected_ms: float | None = None
    playback_flush_ms: float | None = None
    pcm_trace: list[dict[str, Any]] = Field(default_factory=list)


class CancelTurnRequest(BaseModel):
    reason: str = Field(default="user_cancelled", min_length=1, max_length=64)


def _safe_filename(name: str | None, fallback: str) -> str:
    value = Path(name or fallback).name.strip()
    return value or fallback


async def _read_upload_limited(upload: UploadFile, limit: int) -> bytes:
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(min(1024 * 1024, limit + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {limit} bytes")
        parts.append(chunk)
    return b"".join(parts)


def create_app():
    settings = AppSettings()
    orchestrator = VoiceOrchestrator(settings)
    manager = TurnManager(
        orchestrator,
        history=InMemoryConversationHistory(
            max_conversations=settings.conversation_max_count,
            ttl_s=settings.conversation_ttl_s,
        ),
    )
    web_dir = Path(__file__).resolve().parent / "web_assets"

    @asynccontextmanager
    async def lifespan(_app):
        yield
        with manager._lock:
            active = [session for session in manager._turns.values() if not session.finished.is_set()]
        for session in active:
            manager.cancel(session.turn_id, reason="server_shutdown")
        orchestrator.close()

    app = FastAPI(title="Eureka Voice Assistant V10", version="0.10.0", lifespan=lifespan)
    app.state.turn_manager = manager
    app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    @app.get("/")
    def index():
        return FileResponse(
            web_dir / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/health/live")
    def live():
        return {
            "status": "ok",
            "pipeline": "v10",
            "control_transport": "websocket",
            "media_transport": "binary-websocket-pcm-v1",
            "playout": "browser-audioworklet-rebuffering-ring-buffer",
        }

    @app.get("/health/ready")
    def ready():
        result = orchestrator.health()
        result["runtime"] = {
            "profile": settings.runtime_profile or None,
            "model_manifest_id": settings.model_manifest_id or None,
            "upstreams": {
                "stt": settings.stt_base_url,
                "rag": settings.rag_base_url,
                "llm": settings.llm_base_url,
                "tts": settings.tts_base_url,
            },
        }
        tts = ((result.get("services") or {}).get("tts") or {}).get("details") or {}
        warmup = tts.get("warmup") or {}
        result["v10"] = {
            "control_transport": "websocket",
            "media_transport": "binary-websocket-pcm-v1",
            "playout_owner": "browser-audioworklet-rebuffering-ring-buffer",
            "browser_prebuffer_ms": settings.browser_audio_prebuffer_ms,
            "browser_rebuffer_ms": settings.browser_audio_rebuffer_ms,
            "llm_tts_backpressure_decoupled": True,
            "tts_realtime_contract_ok": (
                tts.get("pipeline") == "v10"
                and tts.get("profile") == "realtime"
                and tts.get("streaming_mode") == "frame-pcm"
                and "onnx" in str(tts.get("backend") or "").lower()
            ),
            "tts_performance_probe": warmup.get("performance"),
            "conversation_store": "in-memory",
            "vad": {
                "threshold_db": settings.vad_threshold_db,
                "pre_roll_ms": settings.vad_pre_roll_ms,
                "min_speech_ms": settings.vad_min_speech_ms,
                "min_silence_ms": settings.vad_min_silence_ms,
                "activation_frames": settings.vad_activation_frames,
                "barge_in_debounce_ms": settings.vad_barge_in_debounce_ms,
                "max_utterance_ms": settings.vad_max_utterance_ms,
            },
        }
        return result

    def _turn_info(session: TurnSession):
        return {
            "turn_id": session.turn_id,
            "conversation_id": session.conversation_id,
            "generation": session.generation,
            "sample_rate": settings.tts_sample_rate,
            "pcm_dtype": "int16",
            "channels": 1,
            "prebuffer_ms": settings.browser_audio_prebuffer_ms,
            "rebuffer_ms": settings.browser_audio_rebuffer_ms,
            "control_transport": "websocket",
            "media_transport": "binary-websocket-pcm-v1",
            "vad": {
                "threshold_db": settings.vad_threshold_db,
                "pre_roll_ms": settings.vad_pre_roll_ms,
                "min_speech_ms": settings.vad_min_speech_ms,
                "min_silence_ms": settings.vad_min_silence_ms,
                "activation_frames": settings.vad_activation_frames,
                "barge_in_debounce_ms": settings.vad_barge_in_debounce_ms,
                "max_utterance_ms": settings.vad_max_utterance_ms,
            },
        }

    @app.post("/api/turn/text")
    def start_text(request: TextTurnRequest):
        try:
            session = manager.start(
                conversation_id=request.conversation_id,
                query_text=request.query.strip(), rag_mode=request.rag_mode, rerank=request.rerank,
                top_k=request.top_k, voice_index=request.voice_index,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown conversation") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _turn_info(session)

    @app.post("/api/turn/voice")
    async def start_voice(
        audio: UploadFile = File(...), rag_mode: str | None = Form(default=None),
        rerank: bool | None = Form(default=None), top_k: int | None = Form(default=None),
        voice_index: int | None = Form(default=None),
        conversation_id: str | None = Form(default=None),
    ):
        data = await _read_upload_limited(audio, settings.voice_upload_max_bytes)
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio upload")
        temp_dir = Path(tempfile.mkdtemp(prefix="eureka-v10-voice-"))
        filename = _safe_filename(audio.filename, "recording.webm")
        temp_path = temp_dir / filename
        temp_path.write_bytes(data)
        try:
            session = manager.start(
                conversation_id=conversation_id,
                audio_path=temp_path, audio_temp_dir=temp_dir, rag_mode=rag_mode, rerank=rerank,
                top_k=top_k, voice_index=voice_index,
            )
        except KeyError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=404, detail="Unknown conversation") from exc
        except RuntimeError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _turn_info(session)

    @app.post("/api/conversations")
    def create_conversation():
        return manager.history.create().as_dict()

    @app.get("/api/conversations/{conversation_id}")
    def read_conversation(conversation_id: str):
        try:
            return manager.history.get(conversation_id).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown conversation") from exc

    @app.delete("/api/conversations/{conversation_id}")
    def reset_conversation(conversation_id: str):
        if not manager.delete_conversation(conversation_id):
            raise HTTPException(status_code=404, detail="Unknown conversation")
        return {"ok": True, "conversation_id": conversation_id}

    @app.get("/api/turn/{turn_id}")
    def read_turn(turn_id: str):
        try:
            session = manager.get(turn_id)
            conversation = manager.history.get(session.conversation_id)
            turn = next(item for item in conversation.turns if item.turn_id == turn_id)
        except (KeyError, StopIteration) as exc:
            raise HTTPException(status_code=404, detail="Unknown turn") from exc
        return turn.as_dict()

    @app.post("/api/turn/{turn_id}/cancel")
    def cancel_turn(turn_id: str, request: CancelTurnRequest):
        try:
            session = manager.cancel(turn_id, reason=request.reason)
            conversation = manager.history.get(session.conversation_id)
            turn = next(item for item in conversation.turns if item.turn_id == turn_id)
        except (KeyError, StopIteration) as exc:
            raise HTTPException(status_code=404, detail="Unknown turn") from exc
        return turn.as_dict()

    @app.websocket("/ws/turn/{turn_id}")
    async def turn_control(websocket: WebSocket, turn_id: str):
        try:
            session = manager.get(turn_id)
        except KeyError:
            await websocket.close(code=4404, reason="Unknown turn")
            return
        await websocket.accept()
        cursor = int(websocket.query_params.get("after", "0") or 0)
        try:
            while True:
                event = await asyncio.to_thread(manager.next_event, session, cursor, 15.0)
                if event is None:
                    await websocket.send_json({"type": "keepalive", "at": time.time()})
                    if session.finished.is_set() and cursor >= session.event_sequence:
                        await websocket.send_json({"type": "stream_closed"})
                        break
                    continue
                await websocket.send_json(event)
                cursor = int(event.get("event_sequence") or cursor)
                if event.get("type") == "stream_closed":
                    break
        except WebSocketDisconnect:
            return
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.get("/api/turn/{turn_id}/audio")
    def turn_audio(turn_id: str):
        try:
            manager.get(turn_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown turn") from exc
        return JSONResponse(
            status_code=410,
            content={
                "detail": "Raw HTTP audio was retired to prevent competing consumers",
                "replacement": f"/ws/turn/{turn_id}/audio",
            },
        )

    @app.websocket("/ws/turn/{turn_id}/audio")
    async def turn_audio_websocket(websocket: WebSocket, turn_id: str):
        """Framed PCM media plane with observable per-chunk delivery.

        Each binary message is one application PCM chunk with its sequence and
        T0-T5 timestamps. WebSocket message boundaries survive intermediaries;
        measured latency can therefore be attributed without guessing how an
        unframed HTTP response body was split or coalesced.
        """
        try:
            session = manager.get(turn_id)
        except KeyError:
            await websocket.close(code=4404, reason="Unknown turn")
            return
        await websocket.accept()
        try:
            while True:
                packet = await asyncio.to_thread(session.audio_queue.get)
                if packet is None:
                    await websocket.send_json({"type": "end"})
                    break
                await websocket.send_bytes(encode_pcm_websocket_packet(packet))
        except WebSocketDisconnect:
            return
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.post("/api/turn/{turn_id}/browser-metrics")
    def browser_metrics(turn_id: str, metrics: BrowserMetrics):
        try:
            session = manager.get(turn_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown turn") from exc
        session.browser_metrics = metrics.model_dump(mode="json")
        return {"ok": True}

    @app.get("/api/turn/{turn_id}/browser-metrics")
    def read_browser_metrics(turn_id: str):
        try:
            return manager.get(turn_id).browser_metrics
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown turn") from exc

    @app.get("/api/turn/{turn_id}/replay.wav")
    def replay(turn_id: str):
        try:
            session = manager.get(turn_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown turn") from exc
        if not session.final_audio or not Path(session.final_audio).exists():
            raise HTTPException(status_code=404, detail="Replay is not ready")
        return FileResponse(session.final_audio, media_type="audio/wav")

    @app.get("/api/knowledge/documents")
    def list_documents():
        return orchestrator.rag.list_documents()

    @app.delete("/api/knowledge/documents/{document_id}")
    def delete_document(document_id: str):
        try:
            return orchestrator.rag.delete_document(document_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/knowledge/upload")
    async def upload_knowledge(file: UploadFile = File(...), document_id: str | None = Form(default=None)):
        data = await _read_upload_limited(file, settings.knowledge_upload_max_bytes)
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        original_name = _safe_filename(file.filename, "document.bin")
        temp_dir = Path(tempfile.mkdtemp(prefix="eureka-v10-knowledge-"))
        temp_path = temp_dir / original_name
        temp_path.write_bytes(data)
        resolved_id = (document_id or "").strip() or Path(original_name).stem
        try:
            return orchestrator.rag.upload_document(
                temp_path,
                document_id=resolved_id,
                metadata={"source": "v10-web", "original_filename": original_name},
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @app.exception_handler(KeyError)
    async def key_error_handler(_request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    return app


app = create_app()


def main() -> None:
    import uvicorn
    settings = AppSettings()
    uvicorn.run("voice_app.web:app", host=settings.demo_host, port=settings.demo_port, log_level="info")


if __name__ == "__main__":
    main()
