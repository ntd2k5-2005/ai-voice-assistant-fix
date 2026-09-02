from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.contracts import IngestionResult, RetrievalCandidate, RetrievalResult
from rag.errors import SourceTooLargeError


class FakeKnowledge:
    def __init__(self):
        self.calls = []
        self.documents = {
            "guide": {
                "document_id": "guide",
                "filename": "guide.md",
                "hash": "abc",
                "version": 1,
                "status": "active",
                "created_at": 1.0,
                "updated_at": 2.0,
            }
        }

    def ingest(self, source, **kwargs):
        self.calls.append(("ingest", source, kwargs))
        if kwargs.get("document_id") == "too-large":
            raise SourceTooLargeError("too large")
        return IngestionResult(
            kwargs.get("document_id") or "auto",
            "add",
            1,
            2,
            diagnostics={
                "operation_id": kwargs.get("operation_id") or "op-1",
                "cleanup_pending": False,
            },
        )

    def reindex(self, source, **kwargs):
        self.calls.append(("reindex", source, kwargs))
        return IngestionResult(
            kwargs["document_id"], "reindex", 2, 3, diagnostics={"operation_id": "op-r"}
        )

    def inspect(self, document_id=None, *, include_chunks=False, include_chunk_counts=False):
        if document_id is None:
            rows = [dict(value) for value in self.documents.values()]
            if include_chunk_counts:
                for value in rows:
                    value["chunk_count"] = 1
            return rows
        row = self.documents.get(document_id)
        if row is None:
            return None
        if not include_chunks:
            value = dict(row)
            if include_chunk_counts:
                value["chunk_count"] = 1
            return value
        detailed = dict(row)
        detailed["chunk_count"] = 1
        return {
            "documents": [detailed],
            "active_chunks": [
                {
                    "chunk_id": "chunk-1",
                    "text": "Nội dung kiểm thử",
                    "metadata": {"section": "Mục 1"},
                }
            ],
            "active_chunk_count": 1,
            "inspected_chunk_count": 1,
            "chunks_truncated": False,
        }

    def delete(self, document_id, *, operation_id=None):
        self.calls.append(("delete", document_id, operation_id))
        return {"action": "delete", "version": 1, "operation_id": operation_id or "op-d"}

    def recover(self):
        return ["op-recovered"]


class FakeModule:
    def __init__(self):
        self.knowledge = FakeKnowledge()
        self.config = SimpleNamespace(
            parser=SimpleNamespace(max_file_bytes=1024 * 1024),
            observability=SimpleNamespace(max_nested_items=100, max_text_characters_per_item=4000),
        )
        self.closed = False
        self.retrieve_calls = []

    def retrieve(self, query, **kwargs):
        self.retrieve_calls.append((query, kwargs))
        candidate = RetrievalCandidate(
            chunk_id="chunk-1",
            document_id="guide",
            document_version=1,
            text="Context thật",
            metadata={"heading_path": ["Giải pháp", "Cá nhân"]},
            dense_score=0.8,
            fusion_score=0.7,
            rerank_score=0.9,
            final_score=0.9,
            final_score_source="rerank",
            dense_raw_score=0.6,
            fusion_raw_score=0.04,
            rerank_raw_score=2.2,
            source="guide.md",
            page=2,
            section="Giải pháp > Cá nhân",
        )
        return RetrievalResult(
            candidates=[candidate],
            query=query,
            mode=kwargs.get("mode") or "hybrid",
            rerank_requested=bool(kwargs.get("rerank")),
            total_latency_ms=12.0,
            dense_search_latency_ms=3.0,
            sparse_search_latency_ms=2.0,
            fusion_latency_ms=1.0,
            rerank_latency_ms=5.0,
            diagnostics={"rerank_applied": True, "secret_debug": "shown only in debug"},
        )

    def runtime_snapshot(self):
        return {
            "config_fingerprint": "fingerprint",
            "resolved": {"storage": {"active_document_count": 1}},
        }

    def close(self):
        self.closed = True


def test_lifespan_health_and_close():
    module = FakeModule()
    app = create_app(module_factory=lambda: module)
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        assert client.get("/health/ready").json() == {
            "status": "ready",
            "config_fingerprint": "fingerprint",
            "active_document_count": 1,
        }
    assert module.closed is True


def test_retrieval_public_contract_is_small_and_scores_are_zero_to_one():
    module = FakeModule()
    app = create_app(module_factory=lambda: module)
    with TestClient(app) as client:
        response = client.post(
            "/v1/retrieval",
            json={
                "query": "giải pháp là gì?",
                "mode": "hybrid",
                "rerank": True,
                "top_k": 5,
                "candidate_k": 20,
            },
        )
    assert response.status_code == 200
    payload = response.json()
    context = payload["contexts"][0]
    assert context["score"] == 0.9
    assert context["score_source"] == "rerank"
    assert context["heading_path"] == ["Giải pháp", "Cá nhân"]
    assert "scores" not in context
    assert "raw_scores" not in context
    assert "metadata" not in context
    assert "diagnostics" not in payload
    query, kwargs = module.retrieve_calls[-1]
    assert query == "giải pháp là gì?"
    assert kwargs["include_trace"] is False


def test_retrieval_debug_exposes_normalized_and_raw_stage_scores():
    app = create_app(module_factory=FakeModule)
    with TestClient(app) as client:
        response = client.post(
            "/v1/retrieval",
            json={"query": "x", "mode": "hybrid", "rerank": True, "debug": True},
        )
    assert response.status_code == 200
    context = response.json()["contexts"][0]
    assert context["scores"] == {"dense": 0.8, "fusion": 0.7, "rerank": 0.9}
    assert context["raw_scores"]["rerank"] == 2.2
    assert response.json()["diagnostics"]["secret_debug"] == "shown only in debug"


def test_retrieval_validation_rejects_invalid_candidate_window():
    app = create_app(module_factory=FakeModule)
    with TestClient(app) as client:
        response = client.post(
            "/v1/retrieval",
            json={"query": "x", "top_k": 10, "candidate_k": 5},
        )
    assert response.status_code == 422


def test_file_upload_uses_real_file_handle_and_derives_document_id():
    module = FakeModule()
    app = create_app(module_factory=lambda: module)
    with TestClient(app) as client:
        response = client.post(
            "/v1/knowledge/documents",
            files={"file": ("guide.md", b"# Hello\n\nWorld", "text/markdown")},
            data={"metadata_json": '{"language":"vi"}', "operation_id": "op-client"},
        )
    assert response.status_code == 201
    payload = response.json()
    assert payload["document_id"] == "guide"
    assert payload["chunk_count"] == 2
    _, source, kwargs = module.knowledge.calls[-1]
    assert callable(getattr(source, "read", None))
    assert kwargs["file_name"] == "guide.md"
    assert kwargs["mime_type"] == "text/markdown"
    assert kwargs["metadata"] == {"language": "vi"}
    assert kwargs["include_trace"] is False


def test_text_ingestion_and_management_routes():
    module = FakeModule()
    app = create_app(module_factory=lambda: module)
    with TestClient(app) as client:
        added = client.post(
            "/v1/knowledge/text",
            json={"document_id": "text-1", "text": "Xin chào", "metadata": {"lang": "vi"}},
        )
        assert added.status_code == 201
        assert added.json()["document_id"] == "text-1"
        listed = client.get("/v1/knowledge/documents").json()["documents"][0]
        assert listed["document_id"] == "guide"
        assert listed["chunk_count"] == 1
        detail = client.get("/v1/knowledge/documents/guide")
        assert detail.status_code == 200
        assert detail.json()["document"]["chunk_count"] == 1
        chunks = client.get("/v1/knowledge/documents/guide/chunks").json()
        assert chunks["active_chunk_count"] == 1
        assert chunks["chunks"][0]["chunk_id"] == "chunk-1"
        assert client.get("/v1/knowledge/documents/missing").status_code == 404
        deleted = client.delete("/v1/knowledge/documents/guide?operation_id=op-x")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert client.post("/v1/knowledge/recover").json() == {"recovered_operations": ["op-recovered"]}


def test_known_rag_error_is_mapped_to_http_status():
    module = FakeModule()
    app = create_app(module_factory=lambda: module)
    with TestClient(app) as client:
        response = client.post(
            "/v1/knowledge/documents",
            data={"document_id": "too-large"},
            files={"file": ("big.txt", b"123", "text/plain")},
        )
    assert response.status_code == 413
    assert response.json()["code"] == "SourceTooLargeError"


def test_importing_api_does_not_initialize_rag_models_or_stores():
    code = """
import sys
import rag.api.app
assert 'qdrant_client' not in sys.modules
assert 'sentence_transformers' not in sys.modules
assert 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_text_ingestion_respects_central_parser_size_limit():
    module = FakeModule()
    module.config = SimpleNamespace(parser=SimpleNamespace(max_file_bytes=4))
    app = create_app(module_factory=lambda: module)
    with TestClient(app) as client:
        response = client.post(
            "/v1/knowledge/text",
            json={"document_id": "too-big", "text": "12345"},
        )
    assert response.status_code == 413
