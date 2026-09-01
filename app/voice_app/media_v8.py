from __future__ import annotations

import queue
import statistics
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .contracts import ChatMessage
from .realtime.sentence_scheduler import SentenceScheduler, SentenceTask
from .streaming import SentenceChunker


class V8MediaPipeline:
    """Realtime control/media pipeline with no server-side playout clock.

    V8 keeps sentence-first LLM -> TTS scheduling, but once a sentence starts
    VieNeu PCM is forwarded immediately. The browser owns prebuffering and the
    actual playback clock through Web Audio; this class never wraps live PCM as
    standalone WAV chunks and never estimates browser underruns.
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
        stt_result, transcript, rag_result, messages = self.core._prepare(
            audio_path=audio_path,
            query_text=query_text,
            history=history,
            rag_mode=rag_mode,
            rerank=rerank,
            top_k=top_k,
        )
        sources = [
            source.model_dump(mode="json")
            for source in self.core._sources(rag_result)
        ]

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
        media_q: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=self.settings.tts_audio_queue_max
        )
        cancel = threading.Event()
        llm_done = threading.Event()
        tts_done = threading.Event()
        sentences = SentenceScheduler(
            maxsize=self.settings.tts_sentence_queue_max,
            started_at=started,
            cancel_event=cancel,
        )
        voice = (
            self.settings.tts_voice_index
            if voice_index is None
            else int(voice_index)
        )

        answer_parts: list[str] = []
        sentence_texts: dict[int, str] = {}
        sentence_metrics: list[dict[str, Any]] = []
        final_pcm_parts: list[bytes] = []
        shared: dict[str, Any] = {
            "llm_done_ms": None,
            "first_sentence_queued_ms": None,
            "first_tts_request_ms": None,
            "first_tts_pcm_ms": None,
            "last_pcm_arrival_ms": None,
            "inter_pcm_gaps_ms": [],
            "stream_chunk_count": 0,
            "max_media_queue_depth": 0,
            "media_queue_backpressure_ms": 0.0,
            "llm_meta": {},
        }

        def rel_ms() -> float:
            return (time.perf_counter() - started) * 1000.0

        def put_media(event: dict[str, Any] | None) -> None:
            before = time.perf_counter()
            media_q.put(event)
            shared["media_queue_backpressure_ms"] += (
                time.perf_counter() - before
            ) * 1000.0
            shared["max_media_queue_depth"] = max(
                shared["max_media_queue_depth"], media_q.qsize()
            )

        def schedule_sentence(text: str) -> None:
            task = sentences.put(text)
            if task is None:
                return
            sentence_texts[task.index] = task.text
            if shared["first_sentence_queued_ms"] is None:
                shared["first_sentence_queued_ms"] = task.queued_ms
            control_q.put(
                {
                    "type": "sentence_queued",
                    "sentence_index": task.index,
                    "sentence": task.text,
                    "queued_ms": task.queued_ms,
                }
            )

        def llm_worker() -> None:
            chunker = SentenceChunker(
                min_chars=self.settings.tts_phrase_min_chars,
                soft_chars=self.settings.tts_phrase_soft_chars,
                hard_chars=self.settings.tts_phrase_hard_chars,
            )
            first_ttft = None
            try:
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
                        control_q.put(
                            {
                                "type": "llm_delta",
                                "delta": delta,
                                "answer": "".join(answer_parts),
                                "meta": event,
                            }
                        )
                        for sentence in chunker.push(delta):
                            schedule_sentence(sentence)
                    elif event["type"] == "done":
                        for sentence in chunker.flush():
                            schedule_sentence(sentence)
                        shared["llm_done_ms"] = rel_ms()
                        event["client_ttft_ms"] = first_ttft
                        event["pipeline_done_ms"] = shared["llm_done_ms"]
                        shared["llm_meta"] = event
                        control_q.put(
                            {
                                "type": "llm_done",
                                "answer": "".join(answer_parts).strip(),
                                "meta": event,
                            }
                        )
            except Exception as exc:
                control_q.put(
                    {"type": "error", "stage": "llm", "error": str(exc)}
                )
                cancel.set()
            finally:
                llm_done.set()
                sentences.close()

        def tts_worker() -> None:
            try:
                while True:
                    task = sentences.get()
                    if task is None:
                        sentences.task_done()
                        break
                    assert isinstance(task, SentenceTask)

                    tts_started_ms = rel_ms()
                    if shared["first_tts_request_ms"] is None:
                        shared["first_tts_request_ms"] = tts_started_ms
                    control_q.put(
                        {
                            "type": "tts_sentence_start",
                            "sentence_index": task.index,
                            "sentence": task.text,
                            "started_ms": tts_started_ms,
                        }
                    )

                    first_pcm_ms = None
                    chunk_count = 0
                    audio_ms = 0.0
                    terminal: dict[str, Any] = {}
                    backend = "unknown"
                    streaming_mode = "unknown"
                    profile = "unknown"

                    stream_fn = getattr(self.core.tts, "stream_native_pcm", None)
                    if not callable(stream_fn):
                        raise RuntimeError(
                            "V8 requires TTSClient.stream_native_pcm(); "
                            "legacy coalesced streaming is not accepted"
                        )

                    for tts_event in stream_fn(
                        task.text,
                        voice_index=voice,
                    ):
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
                            put_media(
                                {
                                    "type": "pcm",
                                    "raw": raw,
                                    "duration_ms": duration_ms,
                                    "sentence_index": task.index,
                                    "sentence": task.text,
                                    "chunk_index": shared["stream_chunk_count"],
                                    "sentence_chunk_index": chunk_count,
                                    "server_arrival_ms": now_ms,
                                    "backend": backend,
                                    "streaming_mode": streaming_mode,
                                    "profile": profile,
                                }
                            )
                        elif tts_event.get("type") == "done":
                            terminal = dict(tts_event)

                    finished_ms = rel_ms()
                    sentence_metric = {
                        "sentence": task.index,
                        "sentence_text": task.text,
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
                    }
                    sentence_metrics.append(sentence_metric)
                    control_q.put(
                        {
                            "type": "tts_sentence_done",
                            **sentence_metric,
                        }
                    )
                    sentences.task_done()
            except Exception as exc:
                control_q.put(
                    {"type": "error", "stage": "tts", "error": str(exc)}
                )
                cancel.set()
            finally:
                put_media(None)
                tts_done.set()

        threading.Thread(target=llm_worker, name="v8-llm", daemon=True).start()
        threading.Thread(target=tts_worker, name="v8-tts", daemon=True).start()

        media_source_done = False
        had_error = False
        while True:
            progressed = False

            for _ in range(64):
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

            finished = (
                llm_done.is_set()
                and tts_done.is_set()
                and media_source_done
                and control_q.empty()
            )
            if finished:
                break
            if not progressed:
                time.sleep(0.002)

        if had_error:
            return

        final_audio_path = None
        if final_pcm_parts:
            save_fn = getattr(self.core.tts, "save_pcm_wav", None)
            if callable(save_fn):
                final_audio_path = save_fn(
                    b"".join(final_pcm_parts),
                    prefix="v8-final",
                )

        gaps = list(shared["inter_pcm_gaps_ms"])
        rtfs = [
            float(item["rtf"])
            for item in sentence_metrics
            if item.get("rtf") is not None
        ]
        p95_gap = None
        if gaps:
            if len(gaps) == 1:
                p95_gap = gaps[0]
            else:
                p95_gap = statistics.quantiles(gaps, n=20, method="inclusive")[18]

        llm_done_ms = shared["llm_done_ms"]
        first_pcm_ms = shared["first_tts_pcm_ms"]
        tts_meta = {
            "pipeline_version": "v8",
            "architecture": "control-plane-plus-direct-media-plane",
            "segmentation_mode": "sentence-first",
            "generation_strategy": "vieneu-onnx-native-stream",
            "media_transport": "single-http-pcm-stream",
            "playout_owner": "browser-webaudio",
            "server_playout_buffering": False,
            "server_browser_clock_estimation": False,
            "sentence_count": len(sentence_metrics),
            "stream_chunk_count": shared["stream_chunk_count"],
            "sentence_queue_capacity": self.settings.tts_sentence_queue_max,
            "max_sentence_queue_depth": sentences.max_depth,
            "sentence_backpressure_ms": sentences.total_backpressure_ms,
            "media_queue_capacity": self.settings.tts_audio_queue_max,
            "max_media_queue_depth": shared["max_media_queue_depth"],
            "media_queue_backpressure_ms": shared["media_queue_backpressure_ms"],
            "first_sentence_queued_ms": shared["first_sentence_queued_ms"],
            "pipeline_first_tts_request_ms": shared["first_tts_request_ms"],
            "pipeline_first_tts_pcm_ms": first_pcm_ms,
            "llm_done_ms": llm_done_ms,
            "first_tts_pcm_before_llm_done": (
                first_pcm_ms is not None
                and llm_done_ms is not None
                and first_pcm_ms < llm_done_ms
            ),
            "mean_rtf": sum(rtfs) / len(rtfs) if rtfs else None,
            "max_rtf": max(rtfs) if rtfs else None,
            "max_inter_pcm_gap_ms": max(gaps) if gaps else None,
            "p95_inter_pcm_gap_ms": p95_gap,
            "final_audio_path": final_audio_path,
            "segments": sentence_metrics,
        }
        yield {
            "type": "tts_done",
            "meta": tts_meta,
            "final_audio": final_audio_path,
        }

        llm_meta = shared["llm_meta"] or {}
        timings = {
            "stt_ms": stt_result.client_latency_ms if stt_result else 0.0,
            "rag_ms": rag_result.client_latency_ms,
            "rag_internal_ms": rag_result.latency.total_ms,
            "llm_ms": (
                llm_meta.get("client_total_ms")
                or (llm_meta.get("metrics") or {}).get("total_latency_ms")
            ),
            "llm_ttft_ms": (
                llm_meta.get("client_ttft_ms")
                or (llm_meta.get("metrics") or {}).get("ttft_ms")
            ),
            "first_tts_pcm_ms": first_pcm_ms,
            "total_ms": rel_ms(),
        }
        yield {
            "type": "done",
            "transcript": transcript,
            "answer": "".join(answer_parts).strip(),
            "sources": sources,
            "timings": timings,
            "details": {
                "llm": llm_meta,
                "tts": tts_meta,
            },
            "final_audio": final_audio_path,
            "rag_mode": rag_result.mode,
            "rerank_applied": rag_result.rerank_applied,
        }
