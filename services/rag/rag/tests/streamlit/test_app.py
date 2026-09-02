from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from rag.config import RagConfig
from rag.contracts import IngestionResult, RetrievalCandidate, RetrievalResult


APP = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"


class FakeKnowledge:
    def __init__(self):
        self.ingested = []
        self.deleted = []

    def ingest(self, source):
        self.ingested.append(source)
        return IngestionResult(
            document_id="streamlit-demo",
            action="add",
            version=1,
            chunk_count=1,
            diagnostics={
                "operation_id": "operation-1",
                "trace": {
                    "source": {
                        "file_name": "streamlit-demo.md",
                        "file_type": ".md",
                        "size_bytes": 50,
                        "source_identity": "sha256:test",
                        "parser": "markdown",
                        "raw_preview": "# RAG service overview",
                    },
                    "parsed_units": [
                        {"unit_index": 0, "unit_type": "heading", "text": "RAG service overview"}
                    ],
                    "normalized_units": [
                        {"unit_index": 0, "unit_type": "heading", "text": "RAG service overview"}
                    ],
                    "chunks": [
                        {
                            "index": 0,
                            "chunk_id": "streamlit-demo:1:0",
                            "token_count": 4,
                            "text": "RAG service overview",
                            "section": "RAG service overview",
                        }
                    ],
                    "embedding": {
                        "model": "test-embedding",
                        "dimension": 2,
                        "chunk_count": 1,
                        "batch_size": 1,
                        "elapsed_ms": 0.2,
                        "vector_norm": {"min": 1.0, "max": 1.0, "mean": 1.0},
                        "finite": True,
                        "vector_previews": [[1.0, 0.0]],
                    },
                    "sparse": {
                        "chunk_count": 1,
                        "analyzer": "unicode61",
                        "status": "verified",
                        "elapsed_ms": 0.1,
                    },
                    "lifecycle": {
                        "document_id": "streamlit-demo",
                        "version": 1,
                        "operation_id": "operation-1",
                        "previous_version": None,
                        "transitions": [
                            {"state": "staged"},
                            {"state": "verified"},
                            {"state": "active"},
                        ],
                        "verification_status": "verified",
                        "active": True,
                    },
                },
            },
        )

    update = ingest
    reindex = ingest

    def inspect(self, document_id=None, *, include_chunks=False):
        document = {
            "document_id": "streamlit-demo",
            "filename": "streamlit-demo.md",
            "version": 1,
            "hash": "abc",
            "status": "active",
            "chunk_count": 1,
        }
        chunks = [
            {
                "chunk_id": "streamlit-demo:1:0",
                "document_id": "streamlit-demo",
                "text": "RAG service overview",
            }
        ]
        return {
            "documents": [document],
            "active_chunks": chunks if include_chunks else [],
        }

    def delete(self, document_id):
        self.deleted.append(document_id)
        return {"document_id": document_id, "action": "delete"}

    def recover(self):
        return []


class FakeModule:
    def __init__(self):
        self.config = RagConfig()
        self.knowledge = FakeKnowledge()
        self.retrieval_calls = []

    def runtime_snapshot(self):
        return {
            "configured": self.config.snapshot(),
            "resolved": {
                "chunking": {
                    "effective_max_tokens": 448,
                    "tokenizer": "test-tokenizer",
                },
                "embedding": {
                    "model_name": self.config.embedding.model_name,
                    "family": "test",
                    "dimension": 2,
                    "device": "cpu",
                    "dtype": "float32",
                    "max_sequence_length": 512,
                    "batch_size": 2,
                    "normalized": True,
                    "revision": "test-revision",
                },
                "sparse": {"analyzer": "unicode61"},
                "reranker": {"available": False},
                "storage": {"active_document_count": 1, "active_version_count": 1},
            },
            "config_fingerprint": "test-fingerprint",
        }

    def retrieve(self, query, **kwargs):
        self.retrieval_calls.append((query, kwargs))
        candidate = RetrievalCandidate(
            chunk_id="streamlit-demo:1:0",
            document_id="streamlit-demo",
            text="RAG service overview",
            source="streamlit-demo.md",
            dense_score=0.8,
            sparse_score=1.2,
            fusion_score=0.9,
            final_score=0.9,
        )
        return RetrievalResult(
            candidates=[candidate],
            embed_latency_ms=0.1,
            dense_search_latency_ms=0.2,
            sparse_search_latency_ms=0.1,
            fusion_latency_ms=0.01,
            total_latency_ms=0.5,
            mode=kwargs["mode"],
            query=query.strip(),
            rerank_requested=kwargs["rerank"],
            diagnostics={
                "fusion_strategy": "weighted",
                "rerank_applied": False,
                "trace": {
                    "query": {
                        "original": query,
                        "normalized": query.strip(),
                        "mode": kwargs["mode"],
                        "embedding_executed": kwargs["mode"] != "sparse",
                        "sparse_executed": kwargs["mode"] != "dense",
                        "rerank_requested": kwargs["rerank"],
                    },
                    "query_embedding": {
                        "executed": kwargs["mode"] != "sparse",
                        "model": "test-embedding",
                        "dimension": 2,
                        "elapsed_ms": 0.1,
                        "vector_norm": 1.0,
                        "finite": True,
                        "vector_preview": [1.0, 0.0],
                    },
                    "dense_candidates": [candidate.to_dict()],
                    "sparse_candidates": [candidate.to_dict()],
                    "fusion": {
                        "executed": kwargs["mode"] == "hybrid",
                        "strategy": "weighted",
                        "parameters": {"dense_weight": 0.5, "sparse_weight": 0.5},
                        "candidates": [candidate.to_dict()],
                    },
                    "rerank": {"executed": False},
                },
            },
        )

    def close(self):
        pass


def _app(monkeypatch):
    module = FakeModule()
    monkeypatch.setattr(
        "rag.streamlit_app.utils._load_rag_module",
        lambda config_key, config: module,
    )
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.run()
    assert not app.exception
    return app, module


def test_app_import_and_initial_render_have_no_immediate_exception(monkeypatch):
    app, _ = _app(monkeypatch)
    assert app.title[0].value.endswith("RAG Module Demo")
    assert len(app.tabs) >= 3


def test_app_ingestion_path_calls_public_knowledge_service_and_renders_trace(monkeypatch):
    app, module = _app(monkeypatch)
    app.radio(key="source_kind").set_value("Text").run()
    app.text_input(key="ingest_document_id").set_value("streamlit-demo")
    app.text_area[0].set_value("# Streamlit smoke knowledge")
    app.button(key="run_ingestion").click().run()
    assert not app.exception
    assert len(module.knowledge.ingested) == 1
    assert module.knowledge.ingested[0].source_type == "text"
    assert any("Knowledge" in item.value for item in app.success)
    assert any(tab.label == "4. Chunks" for tab in app.tabs)


def test_app_retrieval_path_calls_public_module_and_renders_all_stages(monkeypatch):
    app, module = _app(monkeypatch)
    mode = app.selectbox(key="retrieval_mode")
    mode.select("hybrid")
    app.button(key="run_retrieval").click().run()
    assert not app.exception
    assert len(module.retrieval_calls) == 1
    _, kwargs = module.retrieval_calls[0]
    assert kwargs["mode"] == "hybrid"
    assert kwargs["top_k"] == module.config.retrieval.top_k
    assert kwargs["candidate_k"] == module.config.retrieval.candidate_k
    assert any("Retrieval" in item.value for item in app.success)
    subheaders = {item.value for item in app.subheader}
    assert "Query preparation" in subheaders
    assert "Dense candidates" in subheaders
    assert "Sparse candidates" in subheaders
    assert "Final Top-K contexts" in subheaders
