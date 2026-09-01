from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class PlayoutMetrics:
    first_dispatch_ms: float | None = None
    dispatched_audio_ms: float = 0.0
    estimated_underrun_count: int = 0
    estimated_max_gap_ms: float = 0.0
    max_buffer_ahead_ms: float = 0.0
    startup_buffer_ms: float = 0.0
    startup_target_ms: float = 0.0


class AdaptiveJitterController:
    """Small server-side jitter/prebuffer controller for browser playout.

    The browser owns the real playback clock, so these numbers are estimates.
    V6 uses them for two things only: (1) hold a little audio before the first
    dispatch when synthesis is close to realtime, and (2) avoid pushing an
    unlimited amount of audio into the browser. It does not pretend to replace
    a browser AudioWorklet clock.
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
        self.metrics = PlayoutMetrics(startup_target_ms=float(base_startup_ms))
        self._first_prepared_at: float | None = None
        self._first_dispatch_at: float | None = None

    def target_for_rtf(self, rtf: float | None) -> float:
        if rtf is None:
            return float(self.base_startup_ms)
        if rtf <= 0.45:
            return float(self.min_startup_ms)
        if rtf <= 0.70:
            return float(max(self.min_startup_ms, self.base_startup_ms))
        if rtf <= 0.95:
            return float(min(self.max_startup_ms, self.base_startup_ms + 700))
        return float(self.max_startup_ms)

    def note_first_prepared(self) -> None:
        if self._first_prepared_at is None:
            self._first_prepared_at = time.perf_counter()

    def startup_ready(
        self,
        *,
        buffered_ms: float,
        first_rtf: float | None,
        source_done: bool,
        prepared_segments: int = 1,
    ) -> bool:
        self.note_first_prepared()
        target = self.target_for_rtf(first_rtf)
        self.metrics.startup_target_ms = target
        self.metrics.startup_buffer_ms = max(self.metrics.startup_buffer_ms, buffered_ms)

        assert self._first_prepared_at is not None
        waited_ms = (time.perf_counter() - self._first_prepared_at) * 1000.0

        # If sentence synthesis is close to realtime, one long first sentence
        # can satisfy a millisecond target while still leaving no look-ahead.
        # Hold briefly for a second prepared segment so sentence B can genuinely
        # sit in the queue before A starts playing.
        risky_rtf = first_rtf is not None and first_rtf >= 0.85
        if (
            risky_rtf
            and not source_done
            and prepared_segments < 2
            and waited_ms < self.startup_max_wait_ms
        ):
            return False

        if source_done or buffered_ms >= target:
            return True

        return waited_ms >= self.startup_max_wait_ms

    def buffer_ahead_ms(self) -> float:
        if self._first_dispatch_at is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._first_dispatch_at) * 1000.0
        return max(0.0, self.metrics.dispatched_audio_ms - elapsed_ms)

    def wait_for_dispatch_budget(self) -> None:
        # Keep enough queued audio for smoothness, but do not flood a long
        # response into the browser at once. This wait naturally propagates
        # backpressure through the bounded audio and sentence queues.
        ahead = self.buffer_ahead_ms()
        if ahead <= self.max_buffer_ahead_limit_ms:
            return
        sleep_ms = ahead - self.max_buffer_ahead_limit_ms
        time.sleep(min(sleep_ms, 250.0) / 1000.0)

    def record_dispatch(self, duration_ms: float) -> tuple[float, float]:
        now = time.perf_counter()
        if self._first_dispatch_at is None:
            self._first_dispatch_at = now
            self.metrics.first_dispatch_ms = (now - self.started_at) * 1000.0
            estimated_gap_ms = 0.0
        else:
            elapsed_ms = (now - self._first_dispatch_at) * 1000.0
            estimated_gap_ms = max(0.0, elapsed_ms - self.metrics.dispatched_audio_ms)
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
