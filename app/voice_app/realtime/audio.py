from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_DTYPE_MAP = {
    "float32": np.dtype("<f4"),
    "int16": np.dtype("<i2"),
}


@dataclass(frozen=True)
class ProcessedPCM:
    raw: bytes
    original_duration_ms: float
    duration_ms: float
    trimmed_ms: float
    appended_pause_ms: float


class AudioPostProcessor:
    """Remove excessive sentence-edge silence while keeping natural pauses.

    Sentence-level TTS often includes a variable amount of leading/trailing
    silence. When separately synthesized sentences are queued back-to-back,
    that silence becomes an audible stop-start pattern. V6 trims only *excess*
    silence, keeps small safety pads around speech, then adds a short controlled
    sentence pause. It never crossfades voiced samples, avoiding clipped words
    and phase artefacts.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        dtype: str,
        enabled: bool = True,
        silence_threshold: float = 0.002,
        leading_keep_ms: int = 35,
        trailing_keep_ms: int = 70,
        sentence_pause_ms: int = 45,
    ) -> None:
        if sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"unsupported dtype {dtype!r}")
        if silence_threshold < 0:
            raise ValueError("silence_threshold must be >= 0")
        if min(leading_keep_ms, trailing_keep_ms, sentence_pause_ms) < 0:
            raise ValueError("silence/pause durations must be >= 0")

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.dtype = dtype
        self.enabled = bool(enabled)
        self.silence_threshold = float(silence_threshold)
        self.leading_keep_ms = int(leading_keep_ms)
        self.trailing_keep_ms = int(trailing_keep_ms)
        self.sentence_pause_ms = int(sentence_pause_ms)
        self.np_dtype = _DTYPE_MAP[dtype]

    @property
    def frame_bytes(self) -> int:
        return self.np_dtype.itemsize * self.channels

    def duration_ms(self, raw: bytes) -> float:
        if len(raw) % self.frame_bytes != 0:
            raise ValueError("PCM is not frame-aligned")
        frames = len(raw) / self.frame_bytes
        return frames / self.sample_rate * 1000.0

    def _to_float_frames(self, raw: bytes) -> np.ndarray:
        if len(raw) % self.frame_bytes != 0:
            raise ValueError("PCM is not frame-aligned")
        values = np.frombuffer(raw, dtype=self.np_dtype)
        if self.channels > 1:
            values = values.reshape(-1, self.channels)
        else:
            values = values.reshape(-1, 1)

        if self.dtype == "int16":
            return values.astype(np.float32) / 32768.0
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("PCM contains NaN/Inf")
        return values

    def _silence_bytes(self, ms: int) -> bytes:
        if ms <= 0:
            return b""
        frames = round(self.sample_rate * ms / 1000.0)
        return bytes(frames * self.frame_bytes)

    def process(self, raw: bytes, *, sentence: str = "") -> ProcessedPCM:
        if not raw:
            raise ValueError("PCM is empty")

        original_ms = self.duration_ms(raw)
        processed = raw
        trimmed_ms = 0.0

        if self.enabled:
            frames = self._to_float_frames(raw)
            amplitude = np.max(np.abs(frames), axis=1)
            active = np.flatnonzero(amplitude >= self.silence_threshold)

            if active.size:
                lead_keep = round(self.sample_rate * self.leading_keep_ms / 1000.0)
                tail_keep = round(self.sample_rate * self.trailing_keep_ms / 1000.0)
                start_frame = max(0, int(active[0]) - lead_keep)
                end_frame = min(len(frames), int(active[-1]) + 1 + tail_keep)
                start_byte = start_frame * self.frame_bytes
                end_byte = end_frame * self.frame_bytes
                processed = raw[start_byte:end_byte]
                trimmed_ms = max(0.0, original_ms - self.duration_ms(processed))

        # A little more pause for questions, while keeping the default compact.
        pause_ms = self.sentence_pause_ms
        stripped = (sentence or "").rstrip()
        if stripped.endswith("?"):
            pause_ms = round(pause_ms * 1.25)
        elif stripped.endswith("!"):
            pause_ms = round(pause_ms * 1.10)

        processed += self._silence_bytes(pause_ms)
        duration_ms = self.duration_ms(processed)

        return ProcessedPCM(
            raw=processed,
            original_duration_ms=original_ms,
            duration_ms=duration_ms,
            trimmed_ms=trimmed_ms,
            appended_pause_ms=float(pause_ms),
        )
