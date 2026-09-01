import threading
import time

import numpy as np

from voice_app.realtime import (
    AdaptiveJitterController,
    AudioPostProcessor,
    SentenceScheduler,
)


def test_audio_postprocessor_trims_excess_silence_and_keeps_controlled_pause():
    sr = 1000
    silence_a = np.zeros(200, dtype="<f4")
    tone = np.ones(500, dtype="<f4") * 0.25
    silence_b = np.zeros(300, dtype="<f4")
    raw = np.concatenate([silence_a, tone, silence_b]).tobytes()

    processor = AudioPostProcessor(
        sample_rate=sr,
        channels=1,
        dtype="float32",
        enabled=True,
        silence_threshold=0.01,
        leading_keep_ms=20,
        trailing_keep_ms=30,
        sentence_pause_ms=40,
    )
    result = processor.process(raw, sentence="Xin chào.")

    assert result.original_duration_ms == 1000.0
    assert result.trimmed_ms == 450.0
    assert result.appended_pause_ms == 40.0
    assert result.duration_ms == 590.0


def test_audio_postprocessor_question_gets_small_extra_pause():
    raw = (np.ones(500, dtype="<f4") * 0.2).tobytes()
    processor = AudioPostProcessor(
        sample_rate=1000,
        channels=1,
        dtype="float32",
        enabled=False,
        sentence_pause_ms=40,
    )
    result = processor.process(raw, sentence="Bạn khỏe không?")
    assert result.appended_pause_ms == 50.0
    assert result.duration_ms == 550.0


def test_sentence_scheduler_applies_real_backpressure_when_full():
    cancel = threading.Event()
    scheduler = SentenceScheduler(
        maxsize=1,
        started_at=time.perf_counter(),
        cancel_event=cancel,
    )
    scheduler.put("Câu một.")

    def delayed_consumer():
        time.sleep(0.06)
        scheduler.get()
        scheduler.task_done()

    thread = threading.Thread(target=delayed_consumer)
    thread.start()
    scheduler.put("Câu hai.")
    thread.join(timeout=1)

    assert scheduler.max_depth == 1
    assert scheduler.total_backpressure_ms >= 30.0


def test_adaptive_jitter_raises_startup_target_when_rtf_is_risky():
    ctrl = AdaptiveJitterController(
        started_at=time.perf_counter(),
        base_startup_ms=1200,
        min_startup_ms=700,
        max_startup_ms=3000,
        startup_max_wait_ms=500,
        max_buffer_ahead_ms=6000,
    )
    assert ctrl.target_for_rtf(0.3) == 700.0
    assert ctrl.target_for_rtf(0.6) == 1200.0
    assert ctrl.target_for_rtf(0.85) == 1900.0
    assert ctrl.target_for_rtf(1.1) == 3000.0


def test_adaptive_jitter_waits_for_second_segment_when_rtf_is_near_realtime():
    ctrl = AdaptiveJitterController(
        started_at=time.perf_counter(),
        base_startup_ms=1200,
        min_startup_ms=700,
        max_startup_ms=3000,
        startup_max_wait_ms=500,
        max_buffer_ahead_ms=6000,
    )
    # A long first sentence alone exceeds the millisecond target, but RTF 0.9
    # is risky enough that V6 should briefly wait for one look-ahead sentence.
    assert ctrl.startup_ready(
        buffered_ms=3500,
        first_rtf=0.9,
        source_done=False,
        prepared_segments=1,
    ) is False
    assert ctrl.startup_ready(
        buffered_ms=4000,
        first_rtf=0.9,
        source_done=False,
        prepared_segments=2,
    ) is True
