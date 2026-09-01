from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SentenceTask:
    index: int
    text: str
    queued_at: float
    queued_ms: float
    queue_depth_at_enqueue: int


class SentenceScheduler:
    """Bounded sentence queue with measurable producer backpressure.

    Blocking ``put`` is intentional. If TTS and playback cannot keep up, the
    LLM producer eventually blocks on this queue instead of accumulating an
    unbounded amount of text in memory. That is the same backpressure pattern
    used by frame-based realtime voice pipelines, adapted here to sentence
    synthesis because VieNeu is most natural and fastest on complete sentences.
    """

    def __init__(
        self,
        *,
        maxsize: int,
        started_at: float,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._queue: queue.Queue[SentenceTask | None] = queue.Queue(maxsize=maxsize)
        self._started_at = float(started_at)
        self._cancel_event = cancel_event
        self._index = 0
        self._lock = threading.Lock()
        self.max_depth = 0
        self.total_backpressure_ms = 0.0
        self.put_count = 0

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    def qsize(self) -> int:
        return self._queue.qsize()

    def put(self, text: str) -> SentenceTask | None:
        text = (text or "").strip()
        if not text:
            return None

        with self._lock:
            self._index += 1
            index = self._index

        queued_at = time.perf_counter()
        task = SentenceTask(
            index=index,
            text=text,
            queued_at=queued_at,
            queued_ms=(queued_at - self._started_at) * 1000.0,
            queue_depth_at_enqueue=min(self._queue.qsize() + 1, self.maxsize),
        )

        wait_started = time.perf_counter()
        while True:
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise RuntimeError("sentence scheduling cancelled")
            try:
                self._queue.put(task, timeout=0.1)
                break
            except queue.Full:
                continue
        waited_ms = (time.perf_counter() - wait_started) * 1000.0

        with self._lock:
            self.total_backpressure_ms += waited_ms
            self.put_count += 1
            self.max_depth = max(self.max_depth, self._queue.qsize())

        return task

    def get(self, timeout: float | None = None) -> SentenceTask | None:
        if timeout is None:
            return self._queue.get()
        return self._queue.get(timeout=timeout)

    def task_done(self) -> None:
        self._queue.task_done()

    def close(self) -> None:
        # ``None`` is ordered after all sentences during a normal turn. On
        # cancellation we discard remaining work, then enqueue the sentinel so
        # a consumer already blocked in ``get`` can still exit cleanly.
        if self._cancel_event is not None and self._cancel_event.is_set():
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            return

        while True:
            try:
                self._queue.put(None, timeout=0.1)
                return
            except queue.Full:
                continue

    @property
    def sentence_count(self) -> int:
        return self._index
