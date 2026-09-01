from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass


_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*+•○]|\d+[.)])\s+")
_EMPH_RE = re.compile(r"(\*\*|__|\*|_)")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")
_HARD_END_RE = re.compile(r"[.!?…]+(?:[\"'”’)]*)?(?=\s|$)")
_ORDERED_MARKER_RE = re.compile(r"^\s*\d+[.)]\s*$")
_ABBREVIATIONS = (
    "TS.", "ThS.", "PGS.", "GS.", "TP.", "Q.", "P.", "Mr.", "Mrs.", "Dr.", "Prof."
)


def markdown_to_speech(text: str) -> str:
    """Remove presentation-only Markdown while preserving spoken content."""
    value = _FENCE_RE.sub(" ", text)
    value = _LINK_RE.sub(r"\1", value)
    value = _INLINE_CODE_RE.sub(r"\1", value)
    value = _HEADING_RE.sub("", value)
    value = _BULLET_RE.sub("", value)
    value = _EMPH_RE.sub("", value)
    value = value.replace("\r", "\n")
    value = _MULTI_NL_RE.sub(". ", value)
    value = value.replace("\n", " ")
    value = _MULTI_SPACE_RE.sub(" ", value)
    return value.strip()


@dataclass(frozen=True)
class SpeechUnitProfile:
    min_chars: int
    target_chars: int
    max_chars: int

    def __post_init__(self) -> None:
        if not 1 <= self.min_chars <= self.target_chars <= self.max_chars:
            raise ValueError("speech unit profile must satisfy 1 <= min <= target <= max")


class AdaptiveSpeechUnitChunker:
    """Incrementally emit semantic TTS units without blocking the LLM reader.

    V10 uses a smaller *first* profile to reduce time-to-first-audio and a larger
    steady-state profile after playback has been primed. This is deliberately
    phase-adaptive rather than pretending character count predicts model speed:
    V9 runtime data showed both fast and slow units at similar lengths.
    """

    def __init__(
        self,
        *,
        first: SpeechUnitProfile | None = None,
        steady: SpeechUnitProfile | None = None,
    ) -> None:
        self.first = first or SpeechUnitProfile(36, 68, 104)
        self.steady = steady or SpeechUnitProfile(64, 124, 176)
        self.buffer = ""
        self.emitted_count = 0

    @property
    def active_profile(self) -> SpeechUnitProfile:
        return self.first if self.emitted_count == 0 else self.steady

    def push(self, delta: str) -> list[str]:
        if delta:
            self.buffer += delta
        out: list[str] = []
        while True:
            cut = self._find_cut(self.active_profile)
            if cut is None:
                break
            raw = self.buffer[:cut]
            self.buffer = self.buffer[cut:].lstrip()
            spoken = markdown_to_speech(raw)
            if spoken:
                self.emitted_count += 1
                out.append(spoken)
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        while self.buffer.strip():
            profile = self.active_profile
            cut = self._find_cut(profile, allow_short=True)
            if cut is None:
                cut = len(self.buffer)
            raw = self.buffer[:cut]
            self.buffer = self.buffer[cut:].lstrip()
            spoken = markdown_to_speech(raw)
            if spoken:
                self.emitted_count += 1
                out.append(spoken)
        self.buffer = ""
        return out

    def _find_cut(self, profile: SpeechUnitProfile, *, allow_short: bool = False) -> int | None:
        if not self.buffer.strip():
            self.buffer = ""
            return None

        # Newlines are strong semantic boundaries for headings/list items.
        newline = self.buffer.find("\n")
        if newline >= 0:
            prefix = self.buffer[: newline + 1]
            spoken_len = len(markdown_to_speech(prefix))
            if (spoken_len >= profile.min_chars or allow_short) and newline + 1 <= profile.max_chars:
                return newline + 1

        hard_ends: list[int] = []
        for match in _HARD_END_RE.finditer(self.buffer):
            prefix = self.buffer[: match.end()]
            stripped = prefix.strip()
            if _ORDERED_MARKER_RE.fullmatch(stripped):
                continue
            if any(stripped.endswith(abbr) for abbr in _ABBREVIATIONS):
                continue
            hard_ends.append(match.end())
            if match.end() >= profile.max_chars:
                break

        # A complete, reasonably-sized sentence is always a good unit.
        if hard_ends and hard_ends[0] <= profile.max_chars:
            # A genuine complete sentence should never wait merely because it is
            # shorter than the target profile. This also keeps greetings responsive.
            return hard_ends[0]

        if len(self.buffer) < profile.min_chars and not allow_short:
            return None

        scan_limit = min(len(self.buffer), profile.max_chars)
        if hard_ends:
            scan_limit = min(scan_limit, hard_ends[0])

        # Prefer semantic clause boundaries near the active target.
        candidates: list[int] = []
        for token in (", ", "; ", ": ", " — ", " – "):
            start = 0
            while True:
                idx = self.buffer.find(token, start, scan_limit + 1)
                if idx < 0:
                    break
                cut = idx + len(token)
                if cut >= profile.min_chars or allow_short:
                    candidates.append(cut)
                start = idx + 1
        if candidates:
            return min(candidates, key=lambda pos: (abs(pos - profile.target_chars), pos > profile.target_chars))

        # If the buffer is not yet over max and no full sentence exists, keep
        # accumulating. This avoids turning every network delta into a TTS call.
        if len(self.buffer) < profile.max_chars and not hard_ends and not allow_short:
            return None

        # Emergency guard: split on whitespace only, never inside a word.
        limit = min(profile.max_chars, hard_ends[0] if hard_ends else profile.max_chars)
        window = self.buffer[: limit + 1]
        before = window.rfind(" ", max(1, profile.min_chars), limit + 1)
        if before > 0:
            return before + 1
        after = self.buffer.find(" ", limit)
        return after + 1 if after >= 0 else min(limit, len(self.buffer))


# Compatibility alias for older tests/imports. V10 primary code uses
# AdaptiveSpeechUnitChunker explicitly.
class SpeechUnitChunker(AdaptiveSpeechUnitChunker):
    def __init__(self, *, min_chars: int = 48, target_chars: int = 90, max_chars: int = 140):
        profile = SpeechUnitProfile(min_chars, target_chars, max_chars)
        super().__init__(first=profile, steady=profile)


@dataclass(frozen=True)
class SpeechUnitTask:
    index: int
    text: str
    queued_ms: float
    queue_depth_at_enqueue: int


class SpeechUnitScheduler:
    """Bounded TTS queue.

    Blocking here is intentional, but in V10 only the speech-segmentation worker
    can block on it. The LLM network/control reader never calls put() directly.
    """

    def __init__(self, *, maxsize: int, started_at: float, cancel_event: threading.Event):
        self._q: queue.Queue[SpeechUnitTask | None] = queue.Queue(maxsize=maxsize)
        self.started_at = started_at
        self.cancel_event = cancel_event
        self.unit_count = 0
        self.max_depth = 0
        self.total_backpressure_ms = 0.0
        self._closed = False

    def qsize(self) -> int:
        return self._q.qsize()

    def put(self, text: str) -> SpeechUnitTask | None:
        spoken = text.strip()
        if not spoken or self.cancel_event.is_set() or self._closed:
            return None
        self.unit_count += 1
        task = SpeechUnitTask(
            index=self.unit_count,
            text=spoken,
            queued_ms=(time.perf_counter() - self.started_at) * 1000.0,
            queue_depth_at_enqueue=self._q.qsize(),
        )
        before = time.perf_counter()
        while not self.cancel_event.is_set():
            try:
                self._q.put(task, timeout=0.05)
                break
            except queue.Full:
                continue
        self.total_backpressure_ms += (time.perf_counter() - before) * 1000.0
        if self.cancel_event.is_set():
            return None
        self.max_depth = max(self.max_depth, self._q.qsize())
        return task

    def get(self) -> SpeechUnitTask | None:
        # Do not let the TTS worker block forever after an upstream failure.
        # On cancellation we stop consuming queued speech immediately; on a
        # normal close we return once the queue has drained.
        while True:
            if self.cancel_event.is_set():
                return None
            try:
                return self._q.get(timeout=0.05)
            except queue.Empty:
                if self._closed:
                    return None

    def task_done(self) -> None:
        self._q.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # During normal completion, append a sentinel after pending units. On
        # cancellation, get() observes cancel_event and exits without requiring
        # a sentinel, so close never deadlocks on a full queue.
        if self.cancel_event.is_set():
            return
        while True:
            try:
                self._q.put(None, timeout=0.05)
                return
            except queue.Full:
                if self.cancel_event.is_set():
                    return
