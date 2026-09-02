from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    PathsConfig,
    RagConfig,
    RerankerConfig,
)
from rag.lifecycle import DocumentRegistry
from rag.retrieval import SQLiteBM25Retriever
from rag.service import RagModule
from rag.storage.qdrant import QdrantStore
from rag.streamlit_app.utils import config_fingerprint
from rag.streamlit_app.view_models import (
    UNAVAILABLE,
    build_config_view,
    build_ingestion_view,
    build_inspection_view,
    build_retrieval_view,
)


APP = Path(__file__).resolve().parents[2] / "streamlit_app" / "app.py"


class DeterministicEmbedder:
    embedding_dim = 2
    model_name = "tests/streamlit-deterministic"
    normalize = True
    resolved_config = {
        "model_name": model_name,
        "family": "deterministic-test",
        "dimension": embedding_dim,
        "device": "cpu",
        "dtype": "float32",
        "max_length": 128,
        "batch_size": 2,
        "normalize": True,
        "revision": "test-revision",
    }

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        values = list(texts)
        self.document_calls.append(values)
        return [[1.0, 0.0] for _ in values]

    def embed_query(self, query):
        self.query_calls.append(query)
        return [1.0, 0.0]


class DeterministicReranker:
    model_name = "tests/streamlit-reranker"
    resolved_config = {
        "model_name": model_name,
        "family": "sequence-classification",
        "adapter": "sequence_classification",
        "score_extraction": "scalar",
        "device": "cpu",
        "dtype": "float32",
        "max_length": 128,
        "batch_size": 2,
        "revision": "test-revision",
    }

    def rerank_candidates(self, candidates, query, top_k):
        reversed_candidates = list(reversed(candidates))
        return [
            (candidate, float(len(reversed_candidates) - index))
            for index, candidate in enumerate(reversed_candidates[:top_k])
        ]


def _module(tmp_path) -> tuple[RagModule, DeterministicEmbedder]:
    config = RagConfig(
        paths=PathsConfig(
            qdrant=tmp_path / "qdrant",
            sparse=tmp_path / "sparse.db",
            registry=tmp_path / "registry.db",
            benchmark_output=tmp_path / "benchmarks",
        ),
        embedding=EmbeddingConfig(
            model_name=DeterministicEmbedder.model_name,
            device="cpu",
            dimension=2,
            max_length=128,
            batch_size=2,
            normalize=True,
        ),
        chunking=ChunkingConfig(
            max_tokens=18,
            fallback_max_tokens=18,
            overlap_tokens=2,
            reserved_tokens=2,
            min_chunk_tokens=1,
        ),
        reranker=RerankerConfig(
            model_name=DeterministicReranker.model_name,
            device="cpu",
            max_length=128,
            batch_size=2,
        ),
    )
    embedder = DeterministicEmbedder()
    dense = QdrantStore(
        config.dense_store.collection_name,
        2,
        str(config.paths.qdrant),
        embedding_model=embedder.model_name,
        normalization=True,
    )
    sparse = SQLiteBM25Retriever(config.paths.sparse)
    registry = DocumentRegistry(config.paths.registry)
    reranker = DeterministicReranker()
    module = RagModule.from_components(
        embedder=embedder,
        dense_store=dense,
        sparse_store=sparse,
        registry=registry,
        reranker=reranker,
        config=config,
    )
    return module, embedder


@pytest.mark.integration
def test_app_executes_ingestion_and_retrieval_through_real_public_facade(
    monkeypatch,
    tmp_path,
):
    module, embedder = _module(tmp_path)
    monkeypatch.setattr(
        "rag.streamlit_app.utils._load_rag_module",
        lambda config_key, config: module,
    )
    try:
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not app.exception

        app.radio(key="source_kind").set_value("Text").run()
        app.text_input(key="ingest_document_id").set_value("streamlit-demo")
        app.text_area[0].set_value(
            "Registry commits activate indexed documents only after dense and sparse verification. "
            "This second sentence ensures the configured chunker produces enough evidence."
        )
        app.button(key="run_ingestion").click().run()
        assert not app.exception
        assert any("Knowledge" in item.value for item in app.success)
        fingerprint = config_fingerprint(module.config)
        ingestion_result = app.session_state[f"rag_last_ingestion:{fingerprint}"]
        ingestion_trace = ingestion_result.diagnostics["trace"]
        ingestion_view = build_ingestion_view(ingestion_result)[0]
        assert ingestion_view["trace_available"] is True
        assert all(
            ingestion_view[field]
            for field in (
                "source",
                "parsed_units",
                "normalized_units",
                "chunks",
                "embedding",
                "sparse",
                "lifecycle",
            )
        )
        assert ingestion_trace["source"]["parser"] in {"text", "markdown"}
        assert ingestion_trace["parsed_units"]
        assert ingestion_trace["normalized_units"]
        assert len(ingestion_trace["chunks"]) >= 2
        assert ingestion_trace["embedding"]["finite"] is True
        assert ingestion_trace["sparse"]["status"] == "verified_active"
        assert ingestion_trace["lifecycle"]["active"] is True
        inspected = module.knowledge.inspect("streamlit-demo")
        assert inspected is not None
        assert inspected["status"] == "active"
        public_inspection = module.knowledge.inspect(
            "streamlit-demo", include_chunks=True
        )
        assert public_inspection["active_chunks"]
        assert build_inspection_view(public_inspection)["chunks_by_document"][
            "streamlit-demo"
        ]
        config_view = build_config_view(module.config, module.runtime_snapshot())
        required_resolved_rows = {
            "Parser / OCR": ("OCR ordered fallback",),
            "Chunking": ("Effective maximum tokens", "Counting tokenizer"),
            "Embedding": (
                "Family / profile",
                "Dimension",
                "Device",
                "Dtype",
                "Maximum sequence length",
                "Batch size",
                "Normalization",
            ),
            "Sparse": ("Analyzer",),
            "Reranker": (
                "Runtime available",
                "Device",
                "Maximum length",
                "Batch size",
            ),
            "Storage / lifecycle": ("Active documents", "Active versions"),
        }
        for section, labels in required_resolved_rows.items():
            rows = {row["setting"]: row for row in config_view[section]}
            assert all(rows[label]["resolved"] != UNAVAILABLE for label in labels)

        app.text_input(key="retrieval_query").set_value("registry commits")
        app.selectbox(key="retrieval_mode").select("hybrid")
        app.button(key="run_retrieval").click().run()
        assert not app.exception
        assert any("Retrieval" in item.value for item in app.success)
        _, retrieval_result = app.session_state[
            f"rag_last_retrieval:{fingerprint}"
        ]
        retrieval_trace = retrieval_result.diagnostics["trace"]
        retrieval_view = build_retrieval_view(
            retrieval_result,
            original_query="registry commits",
        )
        assert retrieval_view["trace_available"] is True
        assert retrieval_view["dense"]["available"] is True
        assert retrieval_view["sparse"]["available"] is True
        assert retrieval_trace["query_embedding"]["executed"] is True
        assert retrieval_trace["dense_candidates"]
        assert retrieval_trace["sparse_candidates"]
        assert retrieval_trace["fusion"]["executed"] is True
        assert retrieval_trace["fusion"]["candidates"]
        assert retrieval_trace["rerank"]["executed"] is True
        assert retrieval_trace["rerank"]["applied"] is True
        assert retrieval_trace["rerank"]["fallback"] is False
        assert retrieval_trace["rerank"]["before"]
        assert retrieval_trace["rerank"]["after"]
        assert (
            retrieval_trace["rerank"]["before"][0]["chunk_id"]
            != retrieval_trace["rerank"]["after"][0]["chunk_id"]
        )
        assert not any(
            "UNAVAILABLE" in warning.value for warning in app.warning
        )
        assert retrieval_result.candidates
        assert retrieval_result.candidates[0].document_id == "streamlit-demo"

        dense_calls = len(embedder.query_calls)
        app.selectbox(key="retrieval_mode").select("sparse")
        app.button(key="run_retrieval").click().run()
        assert not app.exception
        assert len(embedder.query_calls) == dense_calls
        _, sparse_result = app.session_state[f"rag_last_retrieval:{fingerprint}"]
        sparse_trace = sparse_result.diagnostics["trace"]
        assert sparse_trace["query_embedding"]["executed"] is False
        assert sparse_trace["query_embedding"]["reason"]
        assert sparse_trace["dense_candidates"] == []
        assert sparse_trace["sparse_candidates"]
        assert sparse_trace["fusion"]["executed"] is False
        assert any(
            "Query embedding: NOT EXECUTED" in item.value for item in app.info
        )

        app.text_area(key="inspection_update_text").set_value(
            "Updated lifecycle evidence contains the exact token emerald-update."
        )
        app.button(key="inspection_update").click().run()
        assert not app.exception
        updated = module.knowledge.inspect("streamlit-demo")
        assert updated["status"] == "active"
        assert updated["version"] == 2
        updated_result = app.session_state[f"rag_last_ingestion:{fingerprint}"]
        assert updated_result.action == "update"
        assert updated_result.diagnostics["trace"]["lifecycle"]["active"] is True

        app.checkbox(key="confirm_delete").check().run()
        app.button(key="delete_document").click().run()
        assert not app.exception
        deleted = module.knowledge.inspect("streamlit-demo")
        assert deleted["status"] == "deleted"
        assert not module.retrieve(
            "emerald-update",
            mode="sparse",
            rerank=False,
        ).candidates
    finally:
        module.close()
