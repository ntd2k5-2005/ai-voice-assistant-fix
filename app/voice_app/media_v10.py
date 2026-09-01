from __future__ import annotations

import queue
import statistics
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .contracts import ChatMessage
from .realtime.speech_units import (
    AdaptiveSpeechUnitChunker,
    SpeechUnitProfile,
    SpeechUnitScheduler,
    SpeechUnitTask,
)


_SPOOL_DONE = object()


class V10MediaPipeline:
    """Decoupled V10 control/media pipeline.

    Thread boundaries are deliberate:
    1) LLM reader: network -> UI/control + unbounded text spool. Never waits on TTS.
    2) Speech segmenter: text spool -> adaptive semantic units -> bounded TTS queue.
    3) TTS worker: one safe VieNeu stream at a time -> bounded PCM media queue.

    This means a slow TTS can lag behind the text, but it cannot throttle LLM
    token delivery or the control channel anymore.
    """

    def __init__(self, orchestrator: Any) -> None:
        self.core = orchestrator
        self.settings = orchestrator.settings

    def stream_turn(
        self,
        *,
        audio_path: str | Path | None = None,
        query_text: str | None = None,
        history: Sequence[ChatMessage] | None = None,
        rag_mode: str | None = None,
        rerank: bool | None = None,
        top_k: int | None = None,
        voice_index: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()

        def rel_ms() -> float:
            return (time.perf_counter() - started) * 1000.0

        stt_result = None
        transcript = (query_text or "").strip()
        if not transcript:
            if audio_path is None:
                raise ValueError("Provide audio_path or query_text")
            yield {"type": "stage", "stage": "stt", "message": "Đang nhận dạng giọng nói…", "at_ms": rel_ms()}
            stt_result = self.core.stt.transcribe(audio_path)
            transcript = stt_result.text.strip()
        if not transcript:
            raise ValueError("STT returned an empty transcript")
        yield {"type": "transcript_ready", "transcript": transcript, "at_ms": rel_ms()}

        mode = (rag_mode or self.settings.rag_mode).strip().lower()
        rerank_value = self.settings.rag_rerank if rerank is None else bool(rerank)
        top_k_value = top_k or self.settings.rag_top_k
        yield {"type": "stage", "stage": "rag", "message": "Đang truy xuất kho tri thức…", "at_ms": rel_ms()}
        rag_result = self.core.rag.retrieve(
            transcript,
            mode=mode,
            rerank=rerank_value,
            top_k=top_k_value,
            candidate_k=max(self.settings.rag_candidate_k, top_k_value),
            debug=False,
        )
        messages = self.core.prompt_builder.build(
            question=transcript,
            contexts=rag_result.contexts,
            history=history,
        )
        sources = [source.model_dump(mode="json") for source in self.core._sources(rag_result)]
        yield {
            "type": "prepared",
            "transcript": transcript,
            "sources": sources,
            "timings": {
                "stt_ms": stt_result.client_latency_ms if stt_result else 0.0,
                "rag_ms": rag_result.client_latency_ms,
                "rag_internal_ms": rag_result.latency.total_ms,
            },
        }

        control_q: queue.Queue[dict[str, Any]] = queue.Queue()
        # The text spool is intentionally unbounded. LLM_MAX_TOKENS bounds a turn,
        # and decoupling this path is the fix for V9's 8.4s LLM/TTS backpressure.
        speech_spool: queue.Queue[str | object] = queue.Queue()
        media_q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=self.settings.tts_audio_queue_max)
        cancel = threading.Event()
        llm_done = threading.Event()
        segmenter_done = threading.Event()
        tts_done = threading.Event()
        units = SpeechUnitScheduler(
            maxsize=self.settings.tts_sentence_queue_max,
            started_at=started,
            cancel_event=cancel,
        )
        voice = self.settings.tts_voice_index if voice_index is None else int(voice_index)

        answer_parts: list[str] = []
        unit_metrics: list[dict[str, Any]] = []
        final_pcm_parts: list[bytes] = []
        shared: dict[str, Any] = {
            "llm_done_ms": None,
            "segmenter_done_ms": None,
            "first_unit_queued_ms": None,
            "first_tts_request_ms": None,
            "first_tts_pcm_ms": None,
            "last_pcm_arrival_ms": None,
            "inter_pcm_gaps_ms": [],
            "stream_chunk_count": 0,
            "max_media_queue_depth": 0,
            "media_queue_backpressure_ms": 0.0,
            "max_control_queue_depth": 0,
            "max_speech_spool_depth": 0,
            "llm_meta": {},
        }

        def put_control(event: dict[str, Any]) -> None:
            control_q.put(event)
            shared["max_control_queue_depth"] = max(shared["max_control_queue_depth"], control_q.qsize())

        def put_media(event: dict[str, Any] | None) -> None:
            before = time.perf_counter()
            media_q.put(event)
            shared["media_queue_backpressure_ms"] += (time.perf_counter() - before) * 1000.0
            shared["max_media_queue_depth"] = max(shared["max_media_queue_depth"], media_q.qsize())

        def schedule_unit(text: str) -> None:
            task = units.put(text)
            if task is None:
                return
            if shared["first_unit_queued_ms"] is None:
                shared["first_unit_queued_ms"] = task.queued_ms
            put_control({
                "type": "speech_unit_queued",
                "unit_index": task.index,
                "speech_text": task.text,
                "queued_ms": task.queued_ms,
            })

        def llm_reader_worker() -> None:
            first_ttft = None
            try:
                put_control({"type": "stage", "stage": "llm", "message": "LLM đang trả lời…", "at_ms": rel_ms()})
                for event in self.core.llm.chat_stream(
                    messages,
                    temperature=self.settings.llm_temperature,
                    top_p=self.settings.llm_top_p,
                    max_tokens=self.settings.llm_max_tokens,
                ):
                    if event["type"] == "delta":
                        delta = str(event.get("content") or "")
                        if first_ttft is None:
                            first_ttft = event.get("client_ttft_ms")
                        answer_parts.append(delta)
                        # UI/control first. This put is unbounded and cannot wait on TTS.
                        put_control({
                            "type": "llm_delta",
                            "delta": delta,
                            "answer": "".join(answer_parts),
                            "meta": event,
                        })
                        if delta:
                            speech_spool.put(delta)
                            shared["max_speech_spool_depth"] = max(
                                shared["max_speech_spool_depth"], speech_spool.qsize()
                            )
                    elif event["type"] == "done":
                        shared["llm_done_ms"] = rel_ms()
                        event["client_ttft_ms"] = first_ttft
                        event["pipeline_done_ms"] = shared["llm_done_ms"]
                        shared["llm_meta"] = event
                        put_control({
                            "type": "llm_done",
                            "answer": "".join(answer_parts).strip(),
                            "meta": event,
                        })
            except Exception as exc:
                put_control({"type": "error", "stage": "llm", "error": str(exc)})
                cancel.set()
            finally:
                llm_done.set()
                speech_spool.put(_SPOOL_DONE)

        def speech_segmenter_worker() -> None:
            chunker = AdaptiveSpeechUnitChunker(
                first=SpeechUnitProfile(
                    self.settings.tts_first_unit_min_chars,
                    self.settings.tts_first_unit_target_chars,
                    self.settings.tts_first_unit_max_chars,
                ),
                steady=SpeechUnitProfile(
                    self.settings.tts_steady_unit_min_chars,
                    self.settings.tts_steady_unit_target_chars,
                    self.settings.tts_steady_unit_max_chars,
                ),
            )
            try:
                while not cancel.is_set():
                    item = speech_spool.get()
                    if item is _SPOOL_DONE:
                        for unit in chunker.flush():
                            schedule_unit(unit)
                        break
                    for unit in chunker.push(str(item)):
                        schedule_unit(unit)
            except Exception as exc:
                put_control({"type": "error", "stage": "speech-segmentation", "error": str(exc)})
                cancel.set()
            finally:
                shared["segmenter_done_ms"] = rel_ms()
                segmenter_done.set()
                units.close()

        def tts_worker() -> None:
            try:
                while True:
                    task = units.get()
                    if task is None:
                        break
                    assert isinstance(task, SpeechUnitTask)
                    tts_started_ms = rel_ms()
                    if shared["first_tts_request_ms"] is None:
                        shared["first_tts_request_ms"] = tts_started_ms
                    put_control({
                        "type": "tts_unit_start",
                        "unit_index": task.index,
                        "speech_text": task.text,
                        "started_ms": tts_started_ms,
                    })

                    first_pcm_ms = None
                    chunk_count = 0
                    audio_ms = 0.0
                    terminal: dict[str, Any] = {}
                    backend = "unknown"
                    streaming_mode = "unknown"
                    profile = "unknown"
                    upstream_commit = "unknown"
                    for tts_event in self.core.tts.stream_native_pcm(task.text, voice_index=voice):
                        if tts_event.get("type") == "audio":
                            now_ms = rel_ms()
                            if first_pcm_ms is None:
                                first_pcm_ms = now_ms
                                if shared["first_tts_pcm_ms"] is None:
                                    shared["first_tts_pcm_ms"] = now_ms
                            previous = shared["last_pcm_arrival_ms"]
                            if previous is not None:
                                shared["inter_pcm_gaps_ms"].append(now_ms - previous)
                            shared["last_pcm_arrival_ms"] = now_ms

                            raw = tts_event.get("raw") or b""
                            if not raw:
                                continue
                            duration_ms = float(tts_event.get("duration_ms") or 0.0)
                            if duration_ms <= 0:
                                raise RuntimeError("TTS returned invalid PCM duration")
                            chunk_count += 1
                            shared["stream_chunk_count"] += 1
                            audio_ms += duration_ms
                            final_pcm_parts.append(raw)
                            backend = str(tts_event.get("upstream_backend") or backend)
                            streaming_mode = str(tts_event.get("streaming_mode") or streaming_mode)
                            profile = str(tts_event.get("profile") or profile)
                            upstream_commit = str(tts_event.get("upstream_commit") or upstream_commit)
                            put_media({
                                "type": "pcm",
                                "raw": raw,
                                "duration_ms": duration_ms,
                                "unit_index": task.index,
                                "chunk_index": shared["stream_chunk_count"],
                                "unit_chunk_index": chunk_count,
                                "server_arrival_ms": now_ms,
                                "backend": backend,
                                "streaming_mode": streaming_mode,
                                "profile": profile,
                                "upstream_commit": upstream_commit,
                            })
                        elif tts_event.get("type") == "done":
                            terminal = dict(tts_event)

                    finished_ms = rel_ms()
                    metric = {
                        "unit": task.index,
                        "speech_text": task.text,
                        "chars": len(task.text),
                        "words": len(task.text.split()),
                        "queued_ms": task.queued_ms,
                        "synthesis_started_ms": tts_started_ms,
                        "first_pcm_ms": first_pcm_ms,
                        "synthesis_finished_ms": finished_ms,
                        "synthesis_ms": finished_ms - tts_started_ms,
                        "audio_duration_ms": audio_ms,
                        "chunk_count": chunk_count,
                        "ttfb_ms": terminal.get("ttfb_ms"),
                        "rtf": terminal.get("rtf"),
                        "backend": terminal.get("upstream_backend") or backend,
                        "streaming_mode": terminal.get("streaming_mode") or streaming_mode,
                        "profile": terminal.get("profile") or profile,
                        "upstream_commit": terminal.get("upstream_commit") or upstream_commit,
                    }
                    unit_metrics.append(metric)
                    put_control({"type": "tts_unit_done", **metric})
                    units.task_done()
            except Exception as exc:
                put_control({"type": "error", "stage": "tts", "error": str(exc)})
                cancel.set()
            finally:
                put_media(None)
                tts_done.set()

        threading.Thread(target=llm_reader_worker, name="v10-llm-reader", daemon=True).start()
        threading.Thread(target=speech_segmenter_worker, name="v10-speech-segmenter", daemon=True).start()
        threading.Thread(target=tts_worker, name="v10-tts", daemon=True).start()

        media_source_done = False
        had_error = False
        while True:
            progressed = False
            for _ in range(128):
                try:
                    event = control_q.get_nowait()
                except queue.Empty:
                    break
                progressed = True
                if event.get("type") == "error":
                    had_error = True
                    cancel.set()
                yield event

            if not media_source_done:
                try:
                    media = media_q.get(timeout=0.005 if not progressed else 0.0)
                    progressed = True
                except queue.Empty:
                    media = "__empty__"
                if media is None:
                    media_source_done = True
                elif media != "__empty__":
                    yield media

            if (
                llm_done.is_set()
                and segmenter_done.is_set()
                and tts_done.is_set()
                and media_source_done
                and control_q.empty()
            ):
                break
            if not progressed:
                time.sleep(0.002)

        if had_error:
            return

        final_audio_path = None
        if final_pcm_parts:
            save_fn = getattr(self.core.tts, "save_pcm_wav", None)
            if callable(save_fn):
                final_audio_path = save_fn(b"".join(final_pcm_parts), prefix="v10-final")

        gaps = list(shared["inter_pcm_gaps_ms"])
        rtfs = [float(x["rtf"]) for x in unit_metrics if x.get("rtf") is not None]
        p95_gap = None
        if gaps:
            p95_gap = gaps[0] if len(gaps) == 1 else statistics.quantiles(gaps, n=20, method="inclusive")[18]
        llm_done_ms = shared["llm_done_ms"]
        first_pcm_ms = shared["first_tts_pcm_ms"]
        tts_meta = {
            "pipeline_version": "v10",
            "architecture": "decoupled-control-plus-adaptive-speech-media-plane",
            "control_transport": "websocket",
            "segmentation_mode": "phase-adaptive-semantic-speech-units",
            "generation_strategy": "locked-vieneu-onnx-native-stream",
            "media_transport": "single-http-raw-pcm-stream",
            "playout_owner": "browser-audioworklet-rebuffering-ring-buffer",
            "server_playout_buffering": False,
            "server_browser_clock_estimation": False,
            "llm_reader_blocked_by_tts": False,
            "speech_unit_count": len(unit_metrics),
            "max_speech_unit_chars": max((x["chars"] for x in unit_metrics), default=0),
            "first_unit_profile": {
                "min": self.settings.tts_first_unit_min_chars,
                "target": self.settings.tts_first_unit_target_chars,
                "max": self.settings.tts_first_unit_max_chars,
            },
            "steady_unit_profile": {
                "min": self.settings.tts_steady_unit_min_chars,
                "target": self.settings.tts_steady_unit_target_chars,
                "max": self.settings.tts_steady_unit_max_chars,
            },
            "speech_unit_queue_capacity": self.settings.tts_sentence_queue_max,
            "max_speech_unit_queue_depth": units.max_depth,
            "tts_queue_backpressure_ms": units.total_backpressure_ms,
            "llm_reader_tts_backpressure_ms": 0.0,
            "max_speech_spool_depth": shared["max_speech_spool_depth"],
            "max_control_queue_depth": shared["max_control_queue_depth"],
            "stream_chunk_count": shared["stream_chunk_count"],
            "media_queue_capacity": self.settings.tts_audio_queue_max,
            "max_media_queue_depth": shared["max_media_queue_depth"],
            "media_queue_backpressure_ms": shared["media_queue_backpressure_ms"],
            "first_speech_unit_queued_ms": shared["first_unit_queued_ms"],
            "pipeline_first_tts_request_ms": shared["first_tts_request_ms"],
            "pipeline_first_tts_pcm_ms": first_pcm_ms,
            "llm_done_ms": llm_done_ms,
            "speech_segmenter_done_ms": shared["segmenter_done_ms"],
            "first_tts_pcm_before_llm_done": (
                first_pcm_ms is not None and llm_done_ms is not None and first_pcm_ms < llm_done_ms
            ),
            "mean_rtf": sum(rtfs) / len(rtfs) if rtfs else None,
            "max_rtf": max(rtfs) if rtfs else None,
            "realtime_headroom": bool(rtfs) and (sum(rtfs) / len(rtfs)) < 1.0,
            "max_inter_pcm_gap_ms": max(gaps) if gaps else None,
            "p95_inter_pcm_gap_ms": p95_gap,
            "final_audio_path": final_audio_path,
            "units": unit_metrics,
        }
        yield {"type": "tts_done", "meta": tts_meta, "final_audio": final_audio_path}

        llm_meta = shared["llm_meta"] or {}
        timings = {
            "stt_ms": stt_result.client_latency_ms if stt_result else 0.0,
            "rag_ms": rag_result.client_latency_ms,
            "rag_internal_ms": rag_result.latency.total_ms,
            "llm_ms": llm_meta.get("client_total_ms") or (llm_meta.get("metrics") or {}).get("total_latency_ms"),
            "llm_ttft_ms": llm_meta.get("client_ttft_ms") or (llm_meta.get("metrics") or {}).get("ttft_ms"),
            "first_tts_pcm_ms": first_pcm_ms,
            "total_ms": rel_ms(),
        }
        yield {
            "type": "done",
            "transcript": transcript,
            "answer": "".join(answer_parts).strip(),
            "sources": sources,
            "timings": timings,
            "details": {"tts": tts_meta},
        }
