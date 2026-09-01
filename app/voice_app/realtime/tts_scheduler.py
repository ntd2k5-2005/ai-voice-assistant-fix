from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any

from .audio import AudioPostProcessor
from .sentence_scheduler import SentenceScheduler, SentenceTask


@dataclass(frozen=True)
class PreparedSentenceAudio:
    index: int
    sentence: str
    raw: bytes
    duration_ms: float
    original_duration_ms: float
    trimmed_ms: float
    appended_pause_ms: float
    queued_ms: float
    sentence_wait_ms: float
    synthesis_started_ms: float
    synthesis_finished_ms: float
    synthesis_ms: float
    ttfb_ms: float | None
    rtf: float | None
    generation_mode: str
    streaming_mode: str
    upstream_backend: str
    sentence_queue_depth_at_enqueue: int
    sentence_queue_depth_at_start: int


class TTSPrefetchWorker:
    """Single safe TTS worker that prefetches whole sentence audio.

    The worker deliberately does not wait for browser playback. Once sentence A
    is synthesized and queued, it immediately starts sentence B. With a TTS RTF
    below one this creates the desired look-ahead: A can be audible while B is
    already being prepared. A bounded audio queue prevents runaway memory use.
    """

    def __init__(
        self,
        *,
        tts_client: Any,
        voice_index: int,
        generation_mode: str,
        postprocessor: AudioPostProcessor,
        audio_queue: queue.Queue,
        started_at: float,
    ) -> None:
        self.tts = tts_client
        self.voice_index = int(voice_index)
        self.generation_mode = generation_mode
        self.postprocessor = postprocessor
        self.audio_queue = audio_queue
        self.started_at = float(started_at)
        self.max_audio_queue_depth = 0
        self.total_audio_queue_backpressure_ms = 0.0
        self.error: Exception | None = None
        self.prepared: list[PreparedSentenceAudio] = []

    def _fallback_sentence_pcm(self, text: str) -> dict:
        stream_fn = getattr(self.tts, "stream_pcm_chunks", None)
        if not callable(stream_fn):
            stream_fn = getattr(self.tts, "stream_wav_chunks", None)
        if not callable(stream_fn):
            raise RuntimeError("TTS client has no sentence or streaming synthesis method")

        raw_parts: list[bytes] = []
        terminal: dict[str, Any] = {}
        started = time.perf_counter()
        first_pcm_at: float | None = None

        for event in stream_fn(text, voice_index=self.voice_index):
            if event.get("type") == "audio":
                raw = event.get("raw") or b""
                if raw:
                    if first_pcm_at is None:
                        first_pcm_at = time.perf_counter()
                    raw_parts.append(raw)
            elif event.get("type") == "done":
                terminal = dict(event)

        raw = b"".join(raw_parts)
        if not raw:
            raise RuntimeError("TTS fallback returned no PCM")
        total_ms = (time.perf_counter() - started) * 1000.0
        audio_ms = float(terminal.get("audio_duration_ms") or self.postprocessor.duration_ms(raw))
        return {
            "raw": raw,
            "ttfb_ms": (
                (first_pcm_at - started) * 1000.0
                if first_pcm_at is not None
                else None
            ),
            "total_ms": float(terminal.get("total_ms") or total_ms),
            "audio_duration_ms": audio_ms,
            "rtf": terminal.get("rtf") or (total_ms / audio_ms if audio_ms > 0 else None),
            "generation_mode": terminal.get("generation_mode") or "compat-stream",
            "streaming_mode": terminal.get("streaming_mode") or "unknown",
            "upstream_backend": terminal.get("upstream_backend") or "unknown",
        }

    def _synthesize(self, text: str) -> dict:
        fn = getattr(self.tts, "synthesize_sentence_pcm", None)
        if callable(fn):
            return fn(
                text,
                voice_index=self.voice_index,
                generation_mode=self.generation_mode,
            )
        return self._fallback_sentence_pcm(text)

    def run(self, sentences: SentenceScheduler) -> None:
        try:
            while True:
                task = sentences.get()
                if task is None:
                    sentences.task_done()
                    break
                assert isinstance(task, SentenceTask)

                queue_depth_at_start = sentences.qsize()
                started = time.perf_counter()
                started_ms = (started - self.started_at) * 1000.0
                sentence_wait_ms = (started - task.queued_at) * 1000.0

                result = self._synthesize(task.text)
                finished = time.perf_counter()
                finished_ms = (finished - self.started_at) * 1000.0

                processed = self.postprocessor.process(
                    result["raw"],
                    sentence=task.text,
                )
                synthesis_ms = float(result.get("total_ms") or ((finished - started) * 1000.0))
                source_audio_ms = float(
                    result.get("audio_duration_ms") or processed.original_duration_ms
                )
                rtf = result.get("rtf")
                if rtf is None and source_audio_ms > 0:
                    rtf = synthesis_ms / source_audio_ms

                item = PreparedSentenceAudio(
                    index=task.index,
                    sentence=task.text,
                    raw=processed.raw,
                    duration_ms=processed.duration_ms,
                    original_duration_ms=processed.original_duration_ms,
                    trimmed_ms=processed.trimmed_ms,
                    appended_pause_ms=processed.appended_pause_ms,
                    queued_ms=task.queued_ms,
                    sentence_wait_ms=sentence_wait_ms,
                    synthesis_started_ms=started_ms,
                    synthesis_finished_ms=finished_ms,
                    synthesis_ms=synthesis_ms,
                    ttfb_ms=result.get("ttfb_ms"),
                    rtf=float(rtf) if rtf is not None else None,
                    generation_mode=str(result.get("generation_mode") or self.generation_mode),
                    streaming_mode=str(result.get("streaming_mode") or "unknown"),
                    upstream_backend=str(result.get("upstream_backend") or "unknown"),
                    sentence_queue_depth_at_enqueue=task.queue_depth_at_enqueue,
                    sentence_queue_depth_at_start=queue_depth_at_start,
                )
                self.prepared.append(item)

                wait_started = time.perf_counter()
                self.audio_queue.put(item)
                self.total_audio_queue_backpressure_ms += (
                    time.perf_counter() - wait_started
                ) * 1000.0
                self.max_audio_queue_depth = max(
                    self.max_audio_queue_depth,
                    self.audio_queue.qsize(),
                )
                sentences.task_done()
        except Exception as exc:
            self.error = exc
            self.audio_queue.put({"type": "error", "stage": "tts", "error": str(exc)})
        finally:
            self.audio_queue.put(None)
