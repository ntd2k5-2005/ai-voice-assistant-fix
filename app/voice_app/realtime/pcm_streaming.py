from __future__ import annotations

import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .sentence_scheduler import SentenceScheduler, SentenceTask


_DTYPE_MAP = {
    "float32": np.dtype("<f4"),
    "int16": np.dtype("<i2"),
}


@dataclass(frozen=True)
class PCMStreamChunk:
    sentence_index: int
    sentence: str
    raw: bytes
    duration_ms: float
    chunk_index: int
    sentence_chunk_index: int
    queued_ms: float
    sentence_wait_ms: float
    synthesis_started_ms: float
    first_pcm_ms: float
    emitted_ms: float
    rolling_rtf: float | None
    streaming_mode: str
    upstream_backend: str


@dataclass(frozen=True)
class SentenceStreamResult:
    index: int
    sentence: str
    queued_ms: float
    sentence_wait_ms: float
    synthesis_started_ms: float
    first_pcm_ms: float | None
    synthesis_finished_ms: float
    synthesis_ms: float
    audio_duration_ms: float
    ttfb_ms: float | None
    rtf: float | None
    chunk_count: int
    generation_mode: str
    streaming_mode: str
    upstream_backend: str
    sentence_queue_depth_at_enqueue: int
    sentence_queue_depth_at_start: int


@dataclass(frozen=True)
class PCMDispatch:
    raw: bytes
    duration_ms: float
    sentence_indices: tuple[int, ...]


@dataclass
class StreamPlayoutMetrics:
    first_dispatch_ms: float | None = None
    dispatched_audio_ms: float = 0.0
    estimated_underrun_count: int = 0
    estimated_max_gap_ms: float = 0.0
    max_buffer_ahead_ms: float = 0.0
    startup_buffer_ms: float = 0.0
    startup_target_ms: float = 0.0


class ContinuousPCMBuffer:
    """Frame-aligned PCM queue that persists across sentence boundaries.

    A browser media chunk can therefore contain the tail of sentence A and the
    start of sentence B. This avoids forcing a short standalone WAV at every
    semantic boundary while preserving one GPU-safe sentence synthesis worker.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        dtype: str,
        target_chunk_ms: int,
    ) -> None:
        if sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"unsupported dtype {dtype!r}")
        if target_chunk_ms < 50:
            raise ValueError("target_chunk_ms must be >= 50")

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.dtype = dtype
        self.np_dtype = _DTYPE_MAP[dtype]
        self.frame_bytes = self.np_dtype.itemsize * self.channels
        self.target_chunk_ms = int(target_chunk_ms)
        self.target_bytes = self._bytes_for_ms(self.target_chunk_ms)
        self._raw = bytearray()
        self._spans: deque[list[int]] = deque()

    def _bytes_for_ms(self, duration_ms: float) -> int:
        raw_target = int(
            self.sample_rate
            * self.channels
            * self.np_dtype.itemsize
            * (float(duration_ms) / 1000.0)
        )
        return max(
            self.frame_bytes,
            (raw_target // self.frame_bytes) * self.frame_bytes,
        )

    @property
    def buffered_bytes(self) -> int:
        return len(self._raw)

    @property
    def buffered_ms(self) -> float:
        frames = len(self._raw) / self.frame_bytes
        return frames / self.sample_rate * 1000.0

    @property
    def ready(self) -> bool:
        return len(self._raw) >= self.target_bytes

    def ready_for(self, duration_ms: float) -> bool:
        return len(self._raw) >= self._bytes_for_ms(duration_ms)

    def append(self, raw: bytes, *, sentence_index: int) -> None:
        if not raw:
            return
        if len(raw) % self.frame_bytes != 0:
            raise ValueError("PCM chunk is not frame-aligned")
        self._raw.extend(raw)
        if self._spans and self._spans[-1][1] == int(sentence_index):
            self._spans[-1][0] += len(raw)
        else:
            self._spans.append([len(raw), int(sentence_index)])

    def pop(
        self,
        *,
        force: bool = False,
        target_ms: float | None = None,
    ) -> PCMDispatch | None:
        if not self._raw:
            return None

        wanted = self.target_bytes if target_ms is None else self._bytes_for_ms(target_ms)
        if not force and len(self._raw) < wanted:
            return None

        take = min(len(self._raw), wanted)
        if force and len(self._raw) < wanted:
            take = len(self._raw)
        take -= take % self.frame_bytes
        if take <= 0:
            return None

        raw = bytes(self._raw[:take])
        del self._raw[:take]

        remaining = take
        indices: list[int] = []
        while remaining > 0 and self._spans:
            span_bytes, sentence_index = self._spans[0]
            used = min(remaining, span_bytes)
            if not indices or indices[-1] != sentence_index:
                indices.append(sentence_index)
            remaining -= used
            span_bytes -= used
            if span_bytes == 0:
                self._spans.popleft()
            else:
                self._spans[0][0] = span_bytes

        duration_ms = (
            (take / self.frame_bytes) / float(self.sample_rate) * 1000.0
        )
        return PCMDispatch(
            raw=raw,
            duration_ms=duration_ms,
            sentence_indices=tuple(indices),
        )


class AdaptiveStreamJitterController:
    """Low-latency server-side prebuffer estimate for native PCM streaming.

    V7 can see PCM while VieNeu is still synthesizing the current sentence, so
    startup depends on already-buffered audio plus rolling generation RTF rather
    than waiting for a full sentence waveform. The browser remains the real
    playback clock; underrun values here are intentionally estimates.
    """

    def __init__(
        self,
        *,
        started_at: float,
        base_startup_ms: int,
        min_startup_ms: int,
        max_startup_ms: int,
        startup_max_wait_ms: int,
        max_buffer_ahead_ms: int,
    ) -> None:
        if min_startup_ms < 0 or max_startup_ms < min_startup_ms:
            raise ValueError("invalid startup buffer range")
        if not min_startup_ms <= base_startup_ms <= max_startup_ms:
            raise ValueError("base_startup_ms outside configured range")
        if startup_max_wait_ms < 0 or max_buffer_ahead_ms < 1:
            raise ValueError("invalid jitter timing")

        self.started_at = float(started_at)
        self.base_startup_ms = int(base_startup_ms)
        self.min_startup_ms = int(min_startup_ms)
        self.max_startup_ms = int(max_startup_ms)
        self.startup_max_wait_ms = int(startup_max_wait_ms)
        self.max_buffer_ahead_limit_ms = int(max_buffer_ahead_ms)
        self.metrics = StreamPlayoutMetrics(
            startup_target_ms=float(base_startup_ms)
        )
        self._first_pcm_at: float | None = None
        self._first_dispatch_at: float | None = None

    def target_for_rtf(self, rtf: float | None) -> float:
        if rtf is None:
            return float(self.base_startup_ms)
        if rtf <= 0.45:
            return float(self.min_startup_ms)
        if rtf <= 0.75:
            return float(self.base_startup_ms)
        if rtf <= 0.95:
            return float(min(self.max_startup_ms, self.base_startup_ms + 300))
        return float(self.max_startup_ms)

    def note_first_pcm(self) -> None:
        if self._first_pcm_at is None:
            self._first_pcm_at = time.perf_counter()

    def startup_ready(
        self,
        *,
        buffered_ms: float,
        rolling_rtf: float | None,
        source_done: bool,
    ) -> bool:
        if buffered_ms <= 0:
            return False
        self.note_first_pcm()
        target = self.target_for_rtf(rolling_rtf)
        self.metrics.startup_target_ms = target
        self.metrics.startup_buffer_ms = max(
            self.metrics.startup_buffer_ms,
            float(buffered_ms),
        )
        assert self._first_pcm_at is not None
        waited_ms = (time.perf_counter() - self._first_pcm_at) * 1000.0
        if source_done or buffered_ms >= target:
            return True
        return (
            waited_ms >= self.startup_max_wait_ms
            and buffered_ms >= self.min_startup_ms
        )

    def buffer_ahead_ms(self) -> float:
        if self._first_dispatch_at is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._first_dispatch_at) * 1000.0
        return max(0.0, self.metrics.dispatched_audio_ms - elapsed_ms)

    def record_dispatch(self, duration_ms: float) -> tuple[float, float]:
        now = time.perf_counter()
        if self._first_dispatch_at is None:
            self._first_dispatch_at = now
            self.metrics.first_dispatch_ms = (
                now - self.started_at
            ) * 1000.0
            estimated_gap_ms = 0.0
        else:
            elapsed_ms = (now - self._first_dispatch_at) * 1000.0
            estimated_gap_ms = max(
                0.0,
                elapsed_ms - self.metrics.dispatched_audio_ms,
            )
            if estimated_gap_ms >= 40.0:
                self.metrics.estimated_underrun_count += 1
                self.metrics.estimated_max_gap_ms = max(
                    self.metrics.estimated_max_gap_ms,
                    estimated_gap_ms,
                )

        self.metrics.dispatched_audio_ms += float(duration_ms)
        ahead = self.buffer_ahead_ms()
        self.metrics.max_buffer_ahead_ms = max(
            self.metrics.max_buffer_ahead_ms,
            ahead,
        )
        return estimated_gap_ms, ahead


class TTSStreamWorker:
    """Single GPU-safe sentence worker that forwards native PCM immediately.

    A sentence still starts only after the LLM emits a complete semantic unit.
    Once synthesis starts, however, every PCM chunk from ``infer_stream`` goes
    straight into a bounded queue. The same worker starts sentence B as soon as
    generation of A ends; it never waits for A browser playback to finish.
    """

    def __init__(
        self,
        *,
        tts_client: Any,
        voice_index: int,
        audio_queue: queue.Queue,
        started_at: float,
    ) -> None:
        self.tts = tts_client
        self.voice_index = int(voice_index)
        self.audio_queue = audio_queue
        self.started_at = float(started_at)
        self.results: list[SentenceStreamResult] = []
        self.error: Exception | None = None
        self.max_audio_queue_depth = 0
        self.total_audio_queue_backpressure_ms = 0.0
        self.total_stream_chunks = 0

    def _put_audio(self, item: PCMStreamChunk | dict | None) -> None:
        started = time.perf_counter()
        self.audio_queue.put(item)
        self.total_audio_queue_backpressure_ms += (
            time.perf_counter() - started
        ) * 1000.0
        self.max_audio_queue_depth = max(
            self.max_audio_queue_depth,
            self.audio_queue.qsize(),
        )

    def _stream_events(self, text: str):
        stream_fn = getattr(self.tts, "stream_pcm_chunks", None)
        if callable(stream_fn):
            yield from stream_fn(text, voice_index=self.voice_index)
            return

        # Compatibility fallback for old/fake clients. This makes the contract
        # functional, but metrics expose that the path was sentence-buffered.
        sentence_fn = getattr(self.tts, "synthesize_sentence_pcm", None)
        if not callable(sentence_fn):
            raise RuntimeError("TTS client has no streaming synthesis method")
        result = sentence_fn(
            text,
            voice_index=self.voice_index,
            generation_mode="sentence",
        )
        yield {
            "type": "audio",
            "raw": result["raw"],
            "duration_ms": result.get("audio_duration_ms"),
            "ttfb_ms": result.get("ttfb_ms"),
            "streaming_mode": result.get("streaming_mode") or "sentence-buffered",
            "upstream_backend": result.get("upstream_backend") or "unknown",
        }
        yield {
            "type": "done",
            "total_ms": result.get("total_ms"),
            "audio_duration_ms": result.get("audio_duration_ms"),
            "rtf": result.get("rtf"),
            "ttfb_ms": result.get("ttfb_ms"),
            "generation_mode": result.get("generation_mode") or "sentence",
            "streaming_mode": result.get("streaming_mode") or "sentence-buffered",
            "upstream_backend": result.get("upstream_backend") or "unknown",
        }

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
                synthesis_started_ms = (
                    started - self.started_at
                ) * 1000.0
                sentence_wait_ms = (started - task.queued_at) * 1000.0

                first_pcm_ms: float | None = None
                first_ttfb_ms: float | None = None
                audio_duration_ms = 0.0
                sentence_chunk_count = 0
                terminal: dict[str, Any] = {}
                streaming_mode = "unknown"
                upstream_backend = "unknown"

                for event in self._stream_events(task.text):
                    kind = event.get("type")
                    if kind == "audio":
                        raw = event.get("raw") or b""
                        if not raw:
                            continue
                        now = time.perf_counter()
                        if first_pcm_ms is None:
                            first_pcm_ms = (
                                now - self.started_at
                            ) * 1000.0
                            first_ttfb_ms = event.get("ttfb_ms")

                        duration_ms = float(event.get("duration_ms") or 0.0)
                        if duration_ms <= 0:
                            raise RuntimeError("TTS stream chunk has invalid duration")

                        sentence_chunk_count += 1
                        self.total_stream_chunks += 1
                        audio_duration_ms += duration_ms
                        elapsed_generation_ms = (now - started) * 1000.0
                        rolling_rtf = (
                            elapsed_generation_ms / audio_duration_ms
                            if audio_duration_ms > 0
                            else None
                        )
                        streaming_mode = str(
                            event.get("streaming_mode") or streaming_mode
                        )
                        upstream_backend = str(
                            event.get("upstream_backend") or upstream_backend
                        )

                        self._put_audio(
                            PCMStreamChunk(
                                sentence_index=task.index,
                                sentence=task.text,
                                raw=raw,
                                duration_ms=duration_ms,
                                chunk_index=self.total_stream_chunks,
                                sentence_chunk_index=sentence_chunk_count,
                                queued_ms=task.queued_ms,
                                sentence_wait_ms=sentence_wait_ms,
                                synthesis_started_ms=synthesis_started_ms,
                                first_pcm_ms=float(first_pcm_ms),
                                emitted_ms=(now - self.started_at) * 1000.0,
                                rolling_rtf=rolling_rtf,
                                streaming_mode=streaming_mode,
                                upstream_backend=upstream_backend,
                            )
                        )
                    elif kind == "done":
                        terminal = dict(event)

                finished = time.perf_counter()
                synthesis_finished_ms = (
                    finished - self.started_at
                ) * 1000.0
                if sentence_chunk_count == 0 or first_pcm_ms is None:
                    raise RuntimeError("TTS stream returned no PCM for sentence")

                synthesis_ms = float(
                    terminal.get("total_ms")
                    or ((finished - started) * 1000.0)
                )
                terminal_audio_ms = float(
                    terminal.get("audio_duration_ms") or audio_duration_ms
                )
                rtf = terminal.get("rtf")
                if rtf is None and terminal_audio_ms > 0:
                    rtf = synthesis_ms / terminal_audio_ms

                streaming_mode = str(
                    terminal.get("streaming_mode") or streaming_mode
                )
                upstream_backend = str(
                    terminal.get("upstream_backend") or upstream_backend
                )
                generation_mode = str(
                    terminal.get("generation_mode") or "stream"
                )

                self.results.append(
                    SentenceStreamResult(
                        index=task.index,
                        sentence=task.text,
                        queued_ms=task.queued_ms,
                        sentence_wait_ms=sentence_wait_ms,
                        synthesis_started_ms=synthesis_started_ms,
                        first_pcm_ms=first_pcm_ms,
                        synthesis_finished_ms=synthesis_finished_ms,
                        synthesis_ms=synthesis_ms,
                        audio_duration_ms=terminal_audio_ms,
                        ttfb_ms=(
                            terminal.get("ttfb_ms")
                            if terminal.get("ttfb_ms") is not None
                            else first_ttfb_ms
                        ),
                        rtf=float(rtf) if rtf is not None else None,
                        chunk_count=sentence_chunk_count,
                        generation_mode=generation_mode,
                        streaming_mode=streaming_mode,
                        upstream_backend=upstream_backend,
                        sentence_queue_depth_at_enqueue=(
                            task.queue_depth_at_enqueue
                        ),
                        sentence_queue_depth_at_start=queue_depth_at_start,
                    )
                )
                sentences.task_done()
        except Exception as exc:
            self.error = exc
            self._put_audio(
                {"type": "error", "stage": "tts", "error": str(exc)}
            )
        finally:
            self._put_audio(None)
