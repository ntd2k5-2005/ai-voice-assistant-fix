from __future__ import annotations

import io
import re
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np


_SENTENCE_END_RE = re.compile(r'([.!?…]+(?:["\'”’)]*))(?=\s|$)', re.S)

# Common Vietnamese/academic abbreviations that may appear before whitespace
# without ending a spoken sentence. The list is intentionally conservative;
# it only prevents obvious false sentence boundaries.
_ABBREVIATIONS = {
    "ts.",
    "ths.",
    "pgs.",
    "gs.",
    "bs.",
    "ks.",
    "tp.",
    "q.",
    "p.",
    "v.d.",
    "vd.",
    "v.v.",
    "etc.",
    "mr.",
    "mrs.",
    "dr.",
}


class SentenceChunker:
    """Incrementally build complete, TTS-friendly spoken sentences.

    V5 is sentence-first: normal LLM output is not cut merely because a small
    character threshold was reached. Complete sentence punctuation is the
    primary boundary. Only unusually long sentences are split, first at strong
    clause punctuation and finally at a word boundary as an emergency guard.

    This preserves semantic/prosodic context for TTS while still bounding the
    latency of pathological run-on sentences.
    """

    def __init__(
        self,
        *,
        min_chars: int = 24,
        soft_chars: int = 220,
        hard_chars: int = 360,
    ) -> None:
        self.min_chars = int(min_chars)
        self.soft_chars = int(soft_chars)
        self.hard_chars = int(hard_chars)
        if not (1 <= self.min_chars <= self.soft_chars <= self.hard_chars):
            raise ValueError("expected 1 <= min_chars <= soft_chars <= hard_chars")
        self.buffer = ""

    @staticmethod
    def _last_token(value: str) -> str:
        value = value.rstrip('"\'”’)]} ')
        if not value:
            return ""
        return value.split()[-1].lower()

    def _is_false_period_boundary(self, candidate: str, remainder: str) -> bool:
        """Return True for obvious abbreviation/list-marker periods.

        This deliberately holds an abbreviation even when it is currently at
        the end of the incremental buffer. A later LLM delta may continue the
        same sentence; if generation truly ends there, ``flush()`` releases it.
        """
        token = self._last_token(candidate)
        if token in _ABBREVIATIONS:
            return True
        # Avoid treating simple ordered-list markers ("1. Nội dung") or an
        # isolated initial ("A. Nguyễn") as complete spoken sentences.
        bare = token[:-1] if token.endswith(".") else token
        if bare.isdigit() and len(candidate.strip()) <= 4:
            return True
        if len(bare) == 1 and bare.isalpha():
            return True
        return False

    def _pop_complete_sentence(self) -> str | None:
        for match in _SENTENCE_END_RE.finditer(self.buffer):
            end = match.end()
            candidate = self.buffer[:end].strip()
            remainder = self.buffer[end:]
            punctuation = match.group(1)
            if "." in punctuation and self._is_false_period_boundary(
                candidate,
                remainder,
            ):
                continue
            if candidate:
                self.buffer = remainder
                return candidate
        return None

    def _pop_long_clause(self) -> str | None:
        """Split a long run-on sentence only at a strong semantic boundary."""
        if len(self.buffer) < self.soft_chars:
            return None

        # Search slightly beyond the preferred target so punctuation just after
        # the threshold wins over a hard word split.
        search_end = min(
            len(self.buffer),
            self.soft_chars + max(40, self.soft_chars // 4),
        )
        search_start = max(self.min_chars, self.soft_chars // 2)

        # Prefer stronger clause separators; comma is deliberately excluded
        # here because it often occurs inside noun phrases and enumerations.
        candidates: list[tuple[int, int]] = []
        for delimiter in ("; ", ": ", " — ", " - "):
            idx = self.buffer.rfind(delimiter, search_start, search_end)
            if idx >= search_start:
                candidates.append((idx, len(delimiter.rstrip())))

        if candidates:
            cut, delimiter_len = max(candidates, key=lambda item: item[0])
            end = cut + delimiter_len
            value = self.buffer[:end].strip()
            self.buffer = self.buffer[end:]
            return value or None
        return None

    def _pop_emergency(self) -> str | None:
        if len(self.buffer) < self.hard_chars:
            return None

        # At the hard guard we still prefer punctuation, including comma, then
        # fall back to a word boundary. This path is exceptional and exists to
        # prevent an unpunctuated LLM run-on from delaying speech indefinitely.
        lower = max(self.min_chars, self.hard_chars // 2)
        cut = max(
            self.buffer.rfind(", ", lower, self.hard_chars),
            self.buffer.rfind("; ", lower, self.hard_chars),
            self.buffer.rfind(": ", lower, self.hard_chars),
            self.buffer.rfind(" — ", lower, self.hard_chars),
            self.buffer.rfind(" - ", lower, self.hard_chars),
        )
        if cut >= lower:
            end = cut + 1
        else:
            end = self.buffer.rfind(" ", lower, self.hard_chars)
            if end < lower:
                end = self.hard_chars

        value = self.buffer[:end].strip()
        self.buffer = self.buffer[end:]
        return value or None

    def push(self, delta: str) -> list[str]:
        if not delta:
            return []
        self.buffer += delta
        ready: list[str] = []

        while True:
            self.buffer = self.buffer.lstrip()
            if not self.buffer:
                break

            sentence = self._pop_complete_sentence()
            if sentence is not None:
                ready.append(sentence)
                continue

            clause = self._pop_long_clause()
            if clause is not None:
                ready.append(clause)
                continue

            emergency = self._pop_emergency()
            if emergency is not None:
                ready.append(emergency)
                continue

            break

        return ready

    def flush(self) -> list[str]:
        value = self.buffer.strip()
        self.buffer = ""
        return [value] if value else []


class PCMPlaybackBuffer:
    """Coalesce low-latency PCM frames into browser-friendly media chunks.

    VieNeu can yield audio in much smaller quanta than a browser media player
    wants to consume. Gradio 5 streams output media through an HLS-style
    player and recommends chunks around one second or longer for smooth
    playback. This buffer therefore keeps the internal/native TTS stream small
    while emitting larger, frame-aligned chunks to the UI. Pending PCM is kept
    across sentence boundaries so sentence N+1 can fill the same playback buffer
    instead of forcing a browser-visible gap.
    """

    _ITEM_BYTES = {"float32": 4, "int16": 2}

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        dtype: str,
        target_ms: int,
    ) -> None:
        if sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if dtype not in self._ITEM_BYTES:
            raise ValueError(f"unsupported PCM dtype {dtype!r}")
        if target_ms < 1:
            raise ValueError("target_ms must be >= 1")

        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.dtype = dtype
        self.target_ms = int(target_ms)
        self.frame_bytes = self._ITEM_BYTES[dtype] * self.channels
        target_frames = max(
            1,
            round(self.sample_rate * (self.target_ms / 1000.0)),
        )
        self.target_bytes = target_frames * self.frame_bytes
        self.pending = bytearray()

    def _validate(self, raw: bytes) -> None:
        if len(raw) % self.frame_bytes != 0:
            raise ValueError(
                "PCM chunk is not frame-aligned: "
                f"bytes={len(raw)}, frame_bytes={self.frame_bytes}"
            )

    def duration_ms(self, raw: bytes | bytearray) -> float:
        self._validate(bytes(raw))
        frames = len(raw) / self.frame_bytes
        return (frames / float(self.sample_rate)) * 1000.0

    @property
    def pending_ms(self) -> float:
        return self.duration_ms(self.pending) if self.pending else 0.0

    def push(self, raw: bytes) -> list[bytes]:
        if not raw:
            return []
        self._validate(raw)
        self.pending.extend(raw)
        ready: list[bytes] = []
        while len(self.pending) >= self.target_bytes:
            ready.append(bytes(self.pending[: self.target_bytes]))
            del self.pending[: self.target_bytes]
        return ready

    def flush(self) -> list[bytes]:
        if not self.pending:
            return []
        raw = bytes(self.pending)
        self._validate(raw)
        self.pending.clear()
        return [raw]


def pcm_to_wav_bytes(
    raw: bytes,
    *,
    dtype: str,
    channels: int,
    sample_rate: int,
) -> bytes:
    """Wrap raw PCM as a standalone PCM16 WAV chunk for browser playback."""
    if channels < 1:
        raise ValueError("channels must be >= 1")

    if dtype == "float32":
        frame_bytes = 4 * channels
        if len(raw) % frame_bytes != 0:
            raise ValueError("float32 PCM chunk is not frame-aligned")
        samples = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(samples).all():
            raise ValueError("PCM contains NaN/Inf")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    elif dtype == "int16":
        frame_bytes = 2 * channels
        if len(raw) % frame_bytes != 0:
            raise ValueError("int16 PCM chunk is not frame-aligned")
        pcm = raw
    else:
        raise ValueError(f"unsupported PCM dtype {dtype!r}")

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()


@dataclass
class StreamSnapshot:
    transcript: str = ""
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    status: str = ""
