from pathlib import Path

import httpx
import numpy as np

from voice_app.clients.llm import LLMClient
from voice_app.clients.rag import RAGClient
from voice_app.clients.tts import TTSClient
from voice_app.contracts import ChatMessage


def test_llm_ndjson_stream_client():
    def handler(request: httpx.Request):
        assert request.url.path == "/v1/chat/stream"
        body = (
            '{"type":"delta","content":"Xin "}\n'
            '{"type":"delta","content":"chào."}\n'
            '{"type":"done","message":{"role":"assistant","content":"Xin chào."},"metrics":{"ttft_ms":1}}\n'
        ).encode()
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content=body,
        )

    client = LLMClient("http://test")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    events = list(
        client.chat_stream(
            [ChatMessage(role="user", content="hi")],
            temperature=0.3,
            top_p=0.9,
            max_tokens=32,
        )
    )
    assert [e["type"] for e in events] == ["delta", "delta", "done"]
    assert events[-1]["client_total_ms"] >= 0
    client.close()


def test_tts_stream_client_returns_wav_chunks():
    raw = np.linspace(-0.4, 0.4, 96000, dtype="<f4").tobytes()

    def handler(request: httpx.Request):
        assert request.url.path == "/api/tts"
        return httpx.Response(
            200,
            headers={
                "content-type": "application/octet-stream",
                "X-Audio-Streaming": "frame-pcm",
                "X-TTS-Backend": "pytorch",
            },
            content=raw,
        )

    client = TTSClient("http://test", sample_rate=48000, channels=1, dtype="float32")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    events = list(client.stream_wav_chunks("Xin chào.", voice_index=0))
    audio_events = [e for e in events if e["type"] == "audio"]
    assert len(audio_events) >= 4
    assert all(e["wav"][:4] == b"RIFF" for e in audio_events)
    assert all(e["duration_ms"] > 0 for e in audio_events)
    assert events[-1]["type"] == "done"
    assert events[-1]["peak"] > 0
    assert events[-1]["streaming_mode"] == "frame-pcm"
    assert events[-1]["upstream_backend"] == "pytorch"
    assert events[-1]["audio_duration_ms"] > 0
    client.close()


def test_tts_stream_client_pcm_path_skips_per_chunk_wav_wrapping():
    raw = np.linspace(-0.2, 0.2, 48000, dtype="<f4").tobytes()

    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            headers={
                "content-type": "application/octet-stream",
                "X-Audio-Streaming": "frame-pcm",
                "X-TTS-Backend": "pytorch",
            },
            content=raw,
        )

    client = TTSClient(
        "http://test",
        sample_rate=48000,
        channels=1,
        dtype="float32",
        stream_chunk_ms=250,
    )
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    events = list(client.stream_pcm_chunks("Xin chào.", voice_index=0))
    audio_events = [e for e in events if e["type"] == "audio"]
    assert len(audio_events) == 4
    assert all("wav" not in e for e in audio_events)
    assert sum(e["duration_ms"] for e in audio_events) == 1000.0
    assert events[-1]["streaming_mode"] == "frame-pcm"
    client.close()



def test_v6_tts_sentence_pcm_requests_sentence_mode():
    import json

    raw = np.linspace(-0.3, 0.3, 48000, dtype="<f4").tobytes()

    def handler(request: httpx.Request):
        assert request.url.path == "/api/tts"
        body = json.loads(request.content.decode("utf-8"))
        assert body["pre_chunked"] is True
        assert body["generation_mode"] == "sentence"
        return httpx.Response(
            200,
            headers={
                "content-type": "application/octet-stream",
                "X-Audio-Streaming": "sentence-buffered",
                "X-TTS-Generation-Mode": "sentence",
                "X-TTS-Backend": "pytorch",
            },
            content=raw,
        )

    client = TTSClient(
        "http://test",
        sample_rate=48000,
        channels=1,
        dtype="float32",
        generation_mode="sentence",
    )
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = client.synthesize_sentence_pcm("Một câu hoàn chỉnh.", voice_index=0)
    assert result["generation_mode"] == "sentence"
    assert result["streaming_mode"] == "sentence-buffered"
    assert result["audio_duration_ms"] == 1000.0
    assert result["peak"] > 0
    client.close()

def test_rag_knowledge_management_client(tmp_path: Path):
    document = tmp_path / "knowledge.txt"
    document.write_text("hello", encoding="utf-8")

    def handler(request: httpx.Request):
        path = request.url.path
        if request.method == "GET" and path == "/v1/knowledge/documents":
            return httpx.Response(200, json={"documents": [{"document_id": "d1", "chunk_count": 2}]})
        if request.method == "GET" and path == "/v1/knowledge/documents/d1/chunks":
            return httpx.Response(200, json={"document": {"document_id": "d1"}, "chunks": [{"chunk_id": "c1", "text": "hello"}]})
        if request.method == "POST" and path == "/v1/knowledge/documents":
            return httpx.Response(201, json={"document_id": "d1", "chunk_count": 1, "version": 1})
        if request.method == "POST" and path == "/v1/knowledge/documents/d1/reindex":
            return httpx.Response(200, json={"document_id": "d1", "chunk_count": 1, "version": 2})
        if request.method == "DELETE" and path == "/v1/knowledge/documents/d1":
            return httpx.Response(200, json={"document_id": "d1", "deleted": True, "action": "delete"})
        return httpx.Response(404, json={"detail": "not found"})

    client = RAGClient("http://test")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.list_documents()[0]["document_id"] == "d1"
    assert client.get_document("d1", include_chunks=True)["chunks"][0]["chunk_id"] == "c1"
    assert client.upload_document(document, document_id="d1")["version"] == 1
    assert client.upload_document(document, document_id="d1", reindex=True)["version"] == 2
    assert client.delete_document("d1")["deleted"] is True
    client.close()
