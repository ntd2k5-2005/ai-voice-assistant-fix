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
                    text="Mã là EUREKA-7421.",
                    score=0.9,
                    score_source="dense",
                )
            ],
            latency=RAGLatency(total_ms=2),
            client_latency_ms=3,
        )

    def close(self):
        pass


class FakeLLM:
    def health(self):
        return ServiceStatus(service="LLM", reachable=True, ready=True)

    def chat_stream(self, messages, **kwargs):
        yield {
            "type": "delta",
            "content": "Mã kiểm thử là EUREKA-7421. Đây là câu hai.",
            "client_ttft_ms": 1,
        }
        yield {
            "type": "done",
            "message": {
                "role": "assistant",
                "content": "Mã kiểm thử là EUREKA-7421. Đây là câu hai.",
            },
            "metrics": {"ttft_ms": 1, "total_latency_ms": 4},
            "client_total_ms": 5,
        }

    def close(self):
        pass


class FakeSentenceTTS:
    def __init__(self, *, sample_rate=1000, sentence_ms=1000, sleep_s=0.0):
        self.sample_rate = sample_rate
        self.sentence_ms = sentence_ms
        self.sleep_s = sleep_s
        self.calls = []

    def health(self):
        return ServiceStatus(service="TTS", reachable=True, ready=True)

    def synthesize_sentence_pcm(self, text, *, voice_index, generation_mode):
        started = time.perf_counter()
        self.calls.append((text, started, generation_mode))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        frames = round(self.sample_rate * self.sentence_ms / 1000)
        raw = (np.ones(frames, dtype="<f4") * 0.15).tobytes()
        total_ms = self.sleep_s * 1000.0 + 1.0
        return {
            "raw": raw,
            "ttfb_ms": total_ms,
            "total_ms": total_ms,
            "audio_duration_ms": float(self.sentence_ms),
            "rtf": total_ms / self.sentence_ms,
            "generation_mode": generation_mode,
            "streaming_mode": "sentence-buffered",
            "upstream_backend": "test",
        }

    def close(self):
        pass


def settings(tmp_path: Path, **overrides):
    base = dict(
        _env_file=None,
        tts_sample_rate=1000,
        tts_channels=1,
        tts_dtype="float32",
        tts_generation_mode="sentence",
        tts_phrase_min_chars=5,
        tts_phrase_soft_chars=120,
        tts_phrase_hard_chars=220,
        tts_stream_chunk_ms=250,
        tts_playback_chunk_ms=1000,
        tts_sentence_queue_max=2,
        tts_audio_queue_max=2,
        tts_startup_buffer_ms=0,
        tts_startup_buffer_min_ms=0,
        tts_startup_buffer_max_ms=0,
        tts_startup_max_wait_ms=0,
        tts_max_buffer_ahead_ms=5000,
        tts_trim_silence=False,
        tts_inter_sentence_pause_ms=0,
        tts_output_dir=tmp_path,
    )
    base.update(overrides)
    return AppSettings(**base)


def test_v6_stream_process_yields_hidden_sink_audio_and_final_replay(tmp_path):
    tts = FakeSentenceTTS(sentence_ms=1000)
    orchestrator = VoiceOrchestrator(
        settings(tmp_path),
        stt_client=FakeSTT(),
        rag_client=FakeRAG(),
        llm_client=FakeLLM(),
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

    kinds = [e["type"] for e in events]
    assert kinds[0] == "prepared"
    assert "llm_delta" in kinds
    assert "audio" in kinds
    assert "tts_done" in kinds
    assert kinds[-1] == "done"

    done = events[-1]
    assert "EUREKA-7421" in done["answer"]
    assert done["final_audio"]
    assert Path(done["final_audio"]).exists()
    tts_meta = done["details"]["tts"]
    assert tts_meta["generation_strategy"] == "sentence-prefetch"
    assert tts_meta["generation_mode"] == "sentence"
    assert tts_meta["playout_strategy"] == "sentence-prefetch-adaptive-jitter"
    assert tts_meta["sentence_count"] == 2


def test_v6_prefetches_sentence_b_while_sentence_a_can_be_playing(tmp_path):
    class TwoSentenceLLM(FakeLLM):
        def chat_stream(self, messages, **kwargs):
            text = (
                "Câu A giải thích đầy đủ một ý nghĩa hoàn chỉnh. "
                "Câu B tiếp tục bằng một ý nghĩa hoàn chỉnh khác."
            )
            yield {"type": "delta", "content": text, "client_ttft_ms": 1}
            yield {
                "type": "done",
                "message": {"role": "assistant", "content": text},
                "metrics": {"ttft_ms": 1, "total_latency_ms": 2},
                "client_total_ms": 3,
            }

    tts = FakeSentenceTTS(sentence_ms=2000, sleep_s=0.01)
    orchestrator = VoiceOrchestrator(
        settings(tmp_path),
        stt_client=FakeSTT(),
        rag_client=FakeRAG(),
        llm_client=TwoSentenceLLM(),
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
    done = events[-1]
    meta = done["details"]["tts"]

    assert [call[0] for call in tts.calls] == [
        "Câu A giải thích đầy đủ một ý nghĩa hoàn chỉnh.",
        "Câu B tiếp tục bằng một ý nghĩa hoàn chỉnh khác.",
    ]
    assert all(call[2] == "sentence" for call in tts.calls)
    assert meta["prefetch_overlap_count"] >= 1
    assert meta["prefetch_overlap_ratio"] == 1.0
    assert meta["prefetch_ready_before_previous_end_count"] >= 1
    assert meta["prefetch_ready_before_previous_end_ratio"] == 1.0
    assert meta["estimated_underrun_count"] == 0


def test_v6_coalesces_short_complete_sentences_for_browser_hls(tmp_path):
    tts = FakeSentenceTTS(sentence_ms=500)
    orchestrator = VoiceOrchestrator(
        settings(
            tmp_path,
            tts_startup_buffer_ms=1000,
            tts_startup_buffer_min_ms=1000,
            tts_startup_buffer_max_ms=1000,
            tts_startup_max_wait_ms=100,
        ),
        stt_client=FakeSTT(),
        rag_client=FakeRAG(),
        llm_client=FakeLLM(),
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
    audios = [e for e in events if e["type"] == "audio"]
    assert len(audios) == 1
    assert audios[0]["duration_ms"] == 1000.0
    assert audios[0]["sentence_indices"] == [1, 2]
    assert audios[0]["playout_strategy"] == "sentence-prefetch-adaptive-jitter"


def test_v6_audio_can_start_before_llm_stream_finishes(tmp_path):
    audio_seen = False

    class SlowLLM(FakeLLM):
        def chat_stream(self, messages, **kwargs):
            yield {
                "type": "delta",
                "content": "Đây là câu đầu tiên hoàn chỉnh.",
                "client_ttft_ms": 1,
            }
            time.sleep(0.08)
            yield {
                "type": "delta",
                "content": " Đây là câu thứ hai hoàn chỉnh.",
                "client_ttft_ms": 1,
            }
            yield {
                "type": "done",
                "message": {
                    "role": "assistant",
                    "content": "Đây là câu đầu tiên hoàn chỉnh. Đây là câu thứ hai hoàn chỉnh.",
                },
                "metrics": {"ttft_ms": 1, "total_latency_ms": 90},
                "client_total_ms": 90,
            }

    tts = FakeSentenceTTS(sentence_ms=1200, sleep_s=0.005)
    orchestrator = VoiceOrchestrator(
        settings(tmp_path),
        stt_client=FakeSTT(),
        rag_client=FakeRAG(),
        llm_client=SlowLLM(),
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
    kinds = [e["type"] for e in events]
    assert kinds.index("audio") < kinds.index("llm_done")


def test_v6_startup_max_wait_releases_short_first_sentence(tmp_path):
    class DelayedSecondSentenceLLM(FakeLLM):
        def chat_stream(self, messages, **kwargs):
            yield {
                "type": "delta",
                "content": "Câu đầu hoàn chỉnh nhưng ngắn.",
                "client_ttft_ms": 1,
            }
            time.sleep(0.10)
            yield {
                "type": "delta",
                "content": " Câu thứ hai hoàn chỉnh đến sau.",
                "client_ttft_ms": 1,
            }
            yield {
                "type": "done",
                "message": {
                    "role": "assistant",
                    "content": (
                        "Câu đầu hoàn chỉnh nhưng ngắn. "
                        "Câu thứ hai hoàn chỉnh đến sau."
                    ),
                },
                "metrics": {"ttft_ms": 1, "total_latency_ms": 110},
                "client_total_ms": 110,
            }

    tts = FakeSentenceTTS(sentence_ms=500, sleep_s=0.002)
    orchestrator = VoiceOrchestrator(
        settings(
            tmp_path,
            tts_startup_buffer_ms=900,
            tts_startup_buffer_min_ms=900,
            tts_startup_buffer_max_ms=3200,
            tts_startup_max_wait_ms=20,
        ),
        stt_client=FakeSTT(),
        rag_client=FakeRAG(),
        llm_client=DelayedSecondSentenceLLM(),
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
    kinds = [e["type"] for e in events]
    first_audio = next(e for e in events if e["type"] == "audio")

    assert kinds.index("audio") < kinds.index("llm_done")
    assert first_audio["duration_ms"] == 500.0
    assert first_audio["sentence_indices"] == [1]
