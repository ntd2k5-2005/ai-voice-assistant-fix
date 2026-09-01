from __future__ import annotations

import asyncio
import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import AppSettings
from .media_v10 import V10MediaPipeline
from .orchestrator import VoiceOrchestrator


@dataclass
class TurnSession:
    turn_id: str
    created_at: float
    event_queue: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)
    audio_queue: queue.Queue[bytes | None] = field(default_factory=queue.Queue)
    finished: threading.Event = field(default_factory=threading.Event)
    final_audio: str | None = None
    browser_metrics: dict[str, Any] = field(default_factory=dict)
    temp_audio_dir: Path | None = None


class TurnManager:
    def __init__(self, orchestrator: VoiceOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.pipeline = V10MediaPipeline(orchestrator)
        self._turns: dict[str, TurnSession] = {}
        self._lock = threading.Lock()

    def _new(self) -> TurnSession:
        session = TurnSession(turn_id=uuid.uuid4().hex, created_at=time.time())
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
        return session

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

    def start(self, *, query_text=None, audio_path=None, audio_temp_dir=None, rag_mode=None, rerank=None, top_k=None, voice_index=None):
        session = self._new()
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
                ):
                    if event.get("type") == "pcm":
                        pcm16 = self._pcm_float32_to_int16(event["raw"])
                        # V10 web boundary is intentionally non-blocking for control:
                        # a slow browser audio reader must not delay WebSocket events.
                        # Turn length is bounded by LLM_MAX_TOKENS, so this per-turn
                        # queue is finite in practice; the TTS/media worker remains
                        # single-threaded and bounded upstream.
                        session.audio_queue.put_nowait(pcm16)
                        session.event_queue.put({k: v for k, v in event.items() if k != "raw"})
                    else:
                        if event.get("type") == "tts_done":
                            session.final_audio = event.get("final_audio")
                        session.event_queue.put(event)
            except Exception as exc:
                session.event_queue.put({"type": "error", "stage": "web-media", "error": str(exc)})
            finally:
                session.audio_queue.put(None)
                session.event_queue.put({"type": "stream_closed"})
                session.finished.set()
                if session.temp_audio_dir is not None:
                    shutil.rmtree(session.temp_audio_dir, ignore_errors=True)

        threading.Thread(target=worker, name=f"v10-turn-{session.turn_id[:8]}", daemon=True).start()
        return session


class TextTurnRequest(BaseModel):
    query: str = Field(min_length=1)
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


def _safe_filename(name: str | None, fallback: str) -> str:
    value = Path(name or fallback).name.strip()
    return value or fallback


def create_app():
    settings = AppSettings()
    orchestrator = VoiceOrchestrator(settings)
    manager = TurnManager(orchestrator)
    web_dir = Path(__file__).resolve().parent / "web_assets"

    app = FastAPI(title="Eureka Voice Assistant V10", version="0.10.0")
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
            "media_transport": "raw-http-pcm",
            "playout": "browser-audioworklet-rebuffering-ring-buffer",
        }

    @app.get("/health/ready")
    def ready():
        result = orchestrator.health()
        tts = ((result.get("services") or {}).get("tts") or {}).get("details") or {}
        warmup = tts.get("warmup") or {}
        result["v10"] = {
            "control_transport": "websocket",
            "media_transport": "single-http-raw-pcm-stream",
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
        }
        return result

    def _turn_info(session: TurnSession):
        return {
            "turn_id": session.turn_id,
            "sample_rate": settings.tts_sample_rate,
            "pcm_dtype": "int16",
            "channels": 1,
            "prebuffer_ms": settings.browser_audio_prebuffer_ms,
            "rebuffer_ms": settings.browser_audio_rebuffer_ms,
            "control_transport": "websocket",
            "media_transport": "raw-pcm16",
        }

    @app.post("/api/turn/text")
    def start_text(request: TextTurnRequest):
        session = manager.start(
            query_text=request.query.strip(), rag_mode=request.rag_mode, rerank=request.rerank,
            top_k=request.top_k, voice_index=request.voice_index,
        )
        return _turn_info(session)

    @app.post("/api/turn/voice")
    async def start_voice(
        audio: UploadFile = File(...), rag_mode: str | None = Form(default=None),
        rerank: bool | None = Form(default=None), top_k: int | None = Form(default=None),
        voice_index: int | None = Form(default=None),
    ):
        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio upload")
        temp_dir = Path(tempfile.mkdtemp(prefix="eureka-v10-voice-"))
        filename = _safe_filename(audio.filename, "recording.webm")
        temp_path = temp_dir / filename
        temp_path.write_bytes(data)
        session = manager.start(
            audio_path=temp_path, audio_temp_dir=temp_dir, rag_mode=rag_mode, rerank=rerank,
            top_k=top_k, voice_index=voice_index,
        )
        return _turn_info(session)

    @app.websocket("/ws/turn/{turn_id}")
    async def turn_control(websocket: WebSocket, turn_id: str):
        try:
            session = manager.get(turn_id)
        except KeyError:
            await websocket.close(code=4404, reason="Unknown turn")
            return
        await websocket.accept()
        try:
            while True:
                try:
                    event = await asyncio.to_thread(session.event_queue.get, True, 15.0)
                except queue.Empty:
                    await websocket.send_json({"type": "keepalive", "at": time.time()})
                    if session.finished.is_set() and session.event_queue.empty():
                        await websocket.send_json({"type": "stream_closed"})
                        break
                    continue
                await websocket.send_json(event)
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
            session = manager.get(turn_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown turn") from exc

        def source():
            while True:
                chunk = session.audio_queue.get()
                if chunk is None:
                    return
                yield chunk

        return StreamingResponse(
            source(), media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no",
                "X-Audio-Sample-Rate": str(settings.tts_sample_rate), "X-Audio-Channels": "1",
                "X-Audio-Dtype": "int16", "X-Media-Plane": "audioworklet-raw-pcm-v10",
            },
        )

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
        data = await file.read()
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
