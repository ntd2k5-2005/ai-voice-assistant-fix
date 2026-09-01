import queue
import threading
import time
from pathlib import Path

import numpy as np

from voice_app.config import AppSettings
from voice_app.contracts import (
    RAGContext,
    RAGLatency,
    RAGResult,
    ServiceStatus,
    STTResult,
)
from voice_app.orchestrator import VoiceOrchestrator
from voice_app.realtime import (
    AdaptiveStreamJitterController,
    ContinuousPCMBuffer,
    SentenceScheduler,
    TTSStreamWorker,
)


class FakeSTT:
    def health(self):
        return ServiceStatus(service="STT", reachable=True, ready=True)

    def transcribe(self, path):
        return STTResult(text="Mã là gì?", client_latency_ms=5)

    def close(self):
        pass


class FakeRAG:
    def health(self):
        return ServiceStatus(service="RAG", reachable=True, ready=True)

    def retrieve(self, query, **kwargs):
        return RAGResult(
            query=query,
            mode=kwargs["mode"],
            rerank_requested=False,
            rerank_applied=False,
            contexts=[
                RAGContext(
                    rank=1,
                    chunk_id="c",
                    document_id="d",
                    text="Mã là EUREKA-V7-7421.",
                    score=0.9,
                    score_source="dense",
                )
            ],
            latency=RAGLatency(total_ms=2),
            client_latency_ms=3,
        )

    def close(self):
        pass


class SlowTwoSentenceLLM:
    def health(self):
        return ServiceStatus(service="LLM", reachable=True, ready=True)

    def chat_stream(self, messages, **kwargs):
        yield {
            "type": "delta",
            "content": "Câu A hoàn chỉnh và có đủ ngữ nghĩa.",
            "client_ttft_ms": 1,
        }
        time.sleep(0.12)
        yield {
            "type": "delta",
            "content": " Câu B hoàn chỉnh và tiếp tục ý tiếp theo.",
            "client_ttft_ms": 1,
        }
        yield {
            "type": "done",
            "message": {
                "role": "assistant",
                "content": (
                    "Câu A hoàn chỉnh và có đủ ngữ nghĩa. "
                    "Câu B hoàn chỉnh và tiếp tục ý tiếp theo."
                ),
            },
            "metrics": {"ttft_ms": 1, "total_latency_ms": 130},
            "client_total_ms": 130,
        }

    def close(self):
        pass


class FakeStreamingTTS:
    def __init__(self, output_dir: Path, *, sample_rate=1000, chunk_sleep=0.008):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.chunk_sleep = chunk_sleep
        self.calls = []

    def health(self):
        return ServiceStatus(service="TTS", reachable=True, ready=True)

    def stream_pcm_chunks(self, text, *, voice_index):
        started = time.perf_counter()
        self.calls.append((text, started))
        for _ in range(4):
            time.sleep(self.chunk_sleep)
            frames = round(self.sample_rate * 250 / 1000)
            raw = (np.ones(frames, dtype="<f4") * 0.15).tobytes()
            yield {
                "type": "audio",
                "raw": raw,
                "duration_ms": 250.0,
                "ttfb_ms": self.chunk_sleep * 1000.0,
                "streaming_mode": "frame-pcm",
                "upstream_backend": "test",
            }
        total_ms = (time.perf_counter() - started) * 1000.0
        yield {
            "type": "done",
            "total_ms": total_ms,
            "audio_duration_ms": 1000.0,
            "rtf": total_ms / 1000.0,
            "ttfb_ms": self.chunk_sleep * 1000.0,
            "generation_mode": "stream",
            "streaming_mode": "frame-pcm",
            "upstream_backend": "test",
        }

    def save_pcm_wav(self, raw, *, prefix):
        out = self.output_dir / f"{prefix}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Tests only need a stable final artifact contract.
        out.write_bytes(b"RIFF" + raw)
        return str(out)

    def close(self):
        pass


def settings(tmp_path: Path, **overrides):
    base = dict(
        _env_file=None,
        tts_sample_rate=1000,
        tts_channels=1,
        tts_dtype="float32",
        tts_generation_mode="stream",
        tts_phrase_min_chars=5,
        tts_phrase_soft_chars=120,
        tts_phrase_hard_chars=220,
        tts_stream_chunk_ms=250,
        tts_playback_chunk_ms=1000,
        tts_playback_min_chunk_ms=500,
        tts_sentence_queue_max=4,
        tts_audio_queue_max=8,
        tts_startup_buffer_ms=500,
        tts_startup_buffer_min_ms=250,
        tts_startup_buffer_max_ms=1200,
        tts_startup_max_wait_ms=500,
        tts_max_buffer_ahead_ms=5000,
        tts_output_dir=tmp_path,
    )
    base.update(overrides)
    return AppSettings(**base)


def test_v7_continuous_pcm_buffer_crosses_sentence_boundary():
    buf = ContinuousPCMBuffer(
        sample_rate=1000,
        channels=1,
        dtype="float32",
        target_chunk_ms=1000,
    )
    a = (np.ones(600, dtype="<f4") * 0.1).tobytes()
    b = (np.ones(600, dtype="<f4") * 0.2).tobytes()
    buf.append(a, sentence_index=1)
    assert not buf.ready
    buf.append(b, sentence_index=2)
    dispatch = buf.pop()
    assert dispatch is not None
    assert dispatch.duration_ms == 1000.0
    assert dispatch.sentence_indices == (1, 2)
    assert buf.buffered_ms == 200.0


def test_v7_buffer_supports_smaller_rescue_dispatch():
    buf = ContinuousPCMBuffer(
        sample_rate=1000,
        channels=1,
        dtype="float32",
        target_chunk_ms=1000,
    )
    raw = (np.ones(750, dtype="<f4") * 0.1).tobytes()
    buf.append(raw, sentence_index=1)
    assert not buf.ready
    assert buf.ready_for(500)
    dispatch = buf.pop(target_ms=500)
    assert dispatch is not None
    assert dispatch.duration_ms == 500.0
    assert buf.buffered_ms == 250.0


def test_v7_adaptive_stream_startup_uses_rolling_rtf():
    ctrl = AdaptiveStreamJitterController(
        started_at=time.perf_counter(),
        base_startup_ms=650,
        min_startup_ms=400,
        max_startup_ms=1600,
        startup_max_wait_ms=750,
        max_buffer_ahead_ms=5000,
    )
    assert ctrl.target_for_rtf(0.3) == 400.0
    assert ctrl.target_for_rtf(0.6) == 650.0
    assert ctrl.target_for_rtf(0.9) == 950.0
    assert ctrl.target_for_rtf(1.1) == 1600.0


def test_v7_worker_emits_pcm_before_sentence_generation_finishes(tmp_path):
    scheduler = SentenceScheduler(
        maxsize=2,
        started_at=time.perf_counter(),
    )
    scheduler.put("Một câu hoàn chỉnh.")
    scheduler.close()
    audio_queue = queue.Queue(maxsize=8)
    tts = FakeStreamingTTS(tmp_path, chunk_sleep=0.02)
    worker = TTSStreamWorker(
        tts_client=tts,
        voice_index=2,
        audio_queue=audio_queue,
        started_at=time.perf_counter(),
    )

    thread = threading.Thread(target=worker.run, args=(scheduler,))
    thread.start()
    first = audio_queue.get(timeout=1)
    first_seen = time.perf_counter()
    assert getattr(first, "raw", b"")
    assert thread.is_alive(), "worker should still be generating the sentence"
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert worker.results
    assert worker.results[0].first_pcm_ms < worker.results[0].synthesis_finished_ms
    assert first_seen > 0


def test_v7_native_pcm_reaches_playout_before_llm_done_and_sentence_finish(tmp_path):
    tts = FakeStreamingTTS(tmp_path)
    orchestrator = VoiceOrchestrator(
        settings(tmp_path),
        stt_client=FakeSTT(),
        rag_client=FakeRAG(),
        llm_client=SlowTwoSentenceLLM(),
        tts_client=tts,
    )

    events = list(
        orchestrator.stream_process(
            query_text="Mã là gì?",
            synthesize=True,
            rag_mode="dense",
            rerank=False,
            top_k=1,
        )
    )
    kinds = [event["type"] for event in events]
    assert kinds.index("audio") < kinds.index("llm_done")

    done = events[-1]
    meta = done["details"]["tts"]
    first = meta["segments"][0]

    assert meta["pipeline_version"] == "v7"
    assert meta["generation_strategy"] == "sentence-native-stream-v7"
    assert meta["playout_strategy"] == "continuous-pcm-adaptive-jitter-v7"
    assert meta["native_frame_streaming"] is True
    assert meta["generation_mode"] == "stream"
    assert meta["sentence_count"] == 2
    assert meta["stream_chunk_count"] == 8
    assert meta["pipeline_first_tts_pcm_ms"] < first["synthesis_finished_ms"]
    assert meta["pipeline_first_audio_ms"] < first["synthesis_finished_ms"]
    assert meta["first_audio_before_first_sentence_finished"] is True
    assert meta["first_tts_pcm_before_llm_done"] is True
    assert meta["first_audio_before_llm_done"] is True
    assert meta["sentence_stream_overlap_count"] >= 1
    assert meta["next_sentence_first_pcm_before_previous_end_count"] >= 1
    assert done["final_audio"]
    assert Path(done["final_audio"]).exists()
