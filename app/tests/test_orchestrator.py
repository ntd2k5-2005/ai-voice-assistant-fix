from voice_app.config import AppSettings
from voice_app.contracts import (
    LLMResult,
    RAGContext,
    RAGLatency,
    RAGResult,
    ServiceStatus,
    STTResult,
    TTSResult,
)
from voice_app.orchestrator import VoiceOrchestrator


class FakeSTT:
    def health(self):
        return ServiceStatus(service="STT", reachable=True, ready=True)
    def transcribe(self, path):
        return STTResult(
            text="Câu hỏi thử nghiệm",
            client_latency_ms=10,
            inference_ms=8,
        )
    def close(self):
        pass


class FakeRAG:
    def health(self):
        return ServiceStatus(service="RAG", reachable=True, ready=True)
    def retrieve(self, query, **kwargs):
        return RAGResult(
            query=query,
            mode=kwargs["mode"],
            rerank_requested=kwargs["rerank"],
            rerank_applied=kwargs["rerank"],
            contexts=[
                RAGContext(
                    rank=1,
                    chunk_id="c1",
                    document_id="d1",
                    text="Ngữ cảnh thử nghiệm",
                    score=0.9,
                    score_source="dense",
                )
            ],
            latency=RAGLatency(total_ms=20),
            client_latency_ms=22,
        )
    def close(self):
        pass


class FakeLLM:
    def health(self):
        return ServiceStatus(service="LLM", reachable=True, ready=True)
    def chat(self, messages, **kwargs):
        assert any("Ngữ cảnh thử nghiệm" in m.content for m in messages)
        return LLMResult(
            text="Câu trả lời",
            model="fake",
            client_latency_ms=30,
        )
    def close(self):
        pass


class FakeTTS:
    def health(self):
        return ServiceStatus(service="TTS", reachable=True, ready=None)
    def synthesize(self, text, **kwargs):
        assert text == "Câu trả lời"
        return TTSResult(
            audio_path="/tmp/fake.wav",
            sample_rate=48000,
            channels=1,
            dtype="float32",
            duration_s=1,
            byte_count=192000,
            ttfb_ms=15,
            total_ms=40,
        )
    def close(self):
        pass


def test_full_pipeline_with_fakes():
    settings = AppSettings(_env_file=None)
    orch = VoiceOrchestrator(
        settings,
        stt_client=FakeSTT(),
        rag_client=FakeRAG(),
        llm_client=FakeLLM(),
        tts_client=FakeTTS(),
    )
    result = orch.process(audio_path="dummy.wav")
    assert result.transcript == "Câu hỏi thử nghiệm"
    assert result.answer == "Câu trả lời"
    assert result.audio_path == "/tmp/fake.wav"
    assert len(result.sources) == 1
    assert result.timings.rag_internal_ms == 20
