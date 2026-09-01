from __future__ import annotations

import json
import math

import pytest

from rag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    ObservabilityConfig,
    PathsConfig,
    RagConfig,
    RerankerConfig,
    RetrievalConfig,
)
from rag.contracts import DocumentSource, IngestionResult, RetrievalCandidate
from rag.lifecycle import DocumentRegistry
from rag.retrieval import SQLiteBM25Retriever
from rag.service import RagModule
from rag.storage.qdrant import QdrantStore


class ObservableEmbedder:
    """Small deterministic embedding component with truthful runtime facts."""

    embedding_dim = 5
    model_name = "tests/observable-embedder"
    max_length = 32
    normalize = True
    resolved_config = {
        "model_name": model_name,
        "family": "deterministic-test",
        "dimension": embedding_dim,
        "device": "cpu",
        "dtype": "float32",
        "max_length": max_length,
        "batch_size": 2,
        "normalize": normalize,
        "revision": "test-revision",
    }

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        # All five dimensions are deliberately non-zero. A two-dimensional
        # trace preview therefore cannot accidentally equal the full vector.
        seed = 1.0 + float(sum(text.encode("utf-8")) % 7)
        raw = [seed, seed + 1.0, seed + 2.0, seed + 3.0, seed + 4.0]
        norm = math.sqrt(sum(value * value for value in raw))
        return [value / norm for value in raw]

    def embed_documents(self, texts):
        values = list(texts)
        self.document_calls.append(values)
        return [self._vector(value) for value in values]

    def embed_query(self, query):
        self.query_calls.append(query)
        return self._vector(query)


class ObservableReranker:
    model_name = "tests/observable-reranker"
    resolved_config = {
        "model_name": model_name,
        "family": "deterministic-test-reranker",
        "device": "cpu",
        "dtype": "float32",
        "max_length": 64,
        "batch_size": 2,
        "revision": "test-reranker-revision",
    }

    def rerank_candidates(self, candidates, query, *, top_k):
        del query
        # Reverse a stable pool using explicit, finite scores. Returning
        # (candidate, score) exercises the real public rerank orchestration.
        values = list(candidates)
        return [
            (candidate, float(position))
            for position, candidate in enumerate(values, 1)
        ][-top_k:][::-1]


def _build_module(tmp_path) -> tuple[RagModule, ObservableEmbedder]:
    config = RagConfig(
        paths=PathsConfig(
            qdrant=tmp_path / "qdrant",
            sparse=tmp_path / "sparse.db",
            registry=tmp_path / "registry.db",
            benchmark_output=tmp_path / "benchmarks",
        ),
        chunking=ChunkingConfig(
            max_tokens=12,
            overlap_tokens=1,
            reserved_tokens=1,
            min_chunk_tokens=1,
        ),
        embedding=EmbeddingConfig(
            model_name=ObservableEmbedder.model_name,
            revision="test-revision",
            device="cpu",
            dtype="float32",
            batch_size=2,
            max_length=ObservableEmbedder.max_length,
            dimension=ObservableEmbedder.embedding_dim,
            normalize=True,
        ),
        reranker=RerankerConfig(
            model_name=ObservableReranker.model_name,
            revision="test-reranker-revision",
            trust_remote_code=False,
            device="cpu",
            dtype="float32",
            batch_size=2,
            max_length=64,
        ),
        retrieval=RetrievalConfig(
            default_mode="hybrid",
            top_k=2,
            candidate_k=4,
            parallel_hybrid=False,
        ),
        observability=ObservabilityConfig(
            max_trace_items=2,
            max_text_characters_per_item=24,
            vector_preview_dimensions=2,
            max_inspection_chunks=2,
        ),
    )
    embedder = ObservableEmbedder()
    dense = QdrantStore(
        config.dense_store.collection_name,
        embedder.embedding_dim,
        str(config.paths.qdrant),
        embedding_model=embedder.model_name,
        normalization=True,
        embedding_revision="test-revision",
    )
    sparse = SQLiteBM25Retriever(config.paths.sparse)
    registry = DocumentRegistry(config.paths.registry)
    module = RagModule.from_components(
        embedder=embedder,
        dense_store=dense,
        sparse_store=sparse,
        registry=registry,
        reranker=ObservableReranker(),
        config=config,
    )
    return module, embedder


def _assert_json_safe(value) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    )
    assert json.loads(encoded) == value


def _walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _assert_no_full_vectors(value) -> None:
    for item in _walk(value):
        if not isinstance(item, dict):
            continue
        assert not ({"vector", "vectors", "embedding_vector"} & set(item))


@pytest.fixture
def observable_module(tmp_path):
    module, embedder = _build_module(tmp_path)
    try:
        yield module, embedder
    finally:
        module.close()


def _long_source() -> str:
    return (
        "Quan sát pipeline giúp kiểm chứng dense sparse fusion. "
        "Từ khóa hồng ngọc xuất hiện trong phần đầu của tài liệu.\n\n"
        "Mỗi giai đoạn cần provenance rõ ràng và kết quả hữu hạn. "
        "Từ khóa lục bảo xuất hiện trong phần giữa của tài liệu.\n\n"
        "Lifecycle chỉ công bố phiên bản đã sẵn sàng hoàn toàn. "
        "Từ khóa lam ngọc xuất hiện trong phần cuối của tài liệu."
    )


@pytest.mark.integration
def test_ingestion_trace_is_exact_json_safe_bounded_and_vector_redacted(
    observable_module,
):
    module, _ = observable_module

    result = module.knowledge.ingest(
        _long_source(),
        source_type="text",
        document_id="observable-guide",
        file_name="observable-guide.txt",
        include_trace=True,
    )

    assert isinstance(result, IngestionResult)
    assert result.chunk_count > 2, "the trace-bound assertion must be non-vacuous"
    trace = result.diagnostics["trace"]
    assert set(trace) == {
        "source",
        "parsed_units",
        "normalized_units",
        "chunks",
        "embedding",
        "sparse",
        "lifecycle",
        "truncation",
    }
    assert set(trace["source"]) == {
        "source",
        "source_type",
        "file_name",
        "file_type",
        "mime_type",
        "size_bytes",
        "source_identity",
        "source_identity_kind",
        "parser",
        "parse_elapsed_ms",
        "parsed_unit_count",
        "raw_preview",
    }
    assert trace["source"]["file_name"] == "observable-guide.txt"
    assert trace["source"]["file_type"] == ".txt"
    assert trace["source"]["size_bytes"] == len(_long_source().encode("utf-8"))
    assert trace["source"]["source_identity"].startswith("sha256:")
    assert trace["source"]["parser"] == "text"

    assert len(trace["parsed_units"]) <= 2
    assert len(trace["normalized_units"]) <= 2
    assert len(trace["chunks"]) == 2
    assert trace["truncation"] == {
        "max_items": 2,
        "max_text_characters_per_item": 24,
        "parsed_units_truncated": False,
        "normalized_units_truncated": False,
        "chunks_truncated": True,
        "sources_truncated": False,
    }
    for group in (trace["parsed_units"], trace["normalized_units"], trace["chunks"]):
        assert all(len(row["text"]) <= 24 for row in group)
        assert all(isinstance(row["text_truncated"], bool) for row in group)

    embedding = trace["embedding"]
    assert set(embedding) == {
        "model",
        "family",
        "dimension",
        "chunk_count",
        "batch_size",
        "normalized",
        "elapsed_ms",
        "vector_norm",
        "finite",
        "vector_previews",
        "preview_dimensions",
        "vectors_omitted",
        "previews_truncated",
    }
    assert embedding["dimension"] == 5
    assert embedding["finite"] is True
    assert embedding["vectors_omitted"] is True
    assert embedding["preview_dimensions"] == 2
    assert len(embedding["vector_previews"]) == 2
    assert all(len(preview) == 2 for preview in embedding["vector_previews"])
    assert embedding["previews_truncated"] is True
    assert set(embedding["vector_norm"]) == {"min", "max", "mean"}

    sparse = trace["sparse"]
    assert sparse["chunk_count"] == result.chunk_count
    assert sparse["analyzer"] == "unicode61"
    assert sparse["status"] == "verified_active"
    assert sparse["elapsed_ms"] >= 0
    assert sparse["timing_scope"] == "sparse_stage_and_verification"

    lifecycle = trace["lifecycle"]
    assert lifecycle["document_id"] == "observable-guide"
    assert lifecycle["version"] == result.version == 1
    assert lifecycle["previous_version"] == 0
    assert lifecycle["verification_status"] == "dense_sparse_parity_verified"
    assert lifecycle["active"] is True
    assert [transition["stage"] for transition in lifecycle["transitions"]] == [
        "registry_begin",
        "dense_write",
        "dense_verify",
        "sparse_write",
        "sparse_verify",
        "physical_readiness",
        "registry_cas",
        "old_version_cleanup",
    ]

    _assert_json_safe(result.diagnostics)
    _assert_no_full_vectors(result.diagnostics)


@pytest.mark.integration
def test_ingestion_trace_bounds_multi_source_details_and_reports_omission(
    observable_module,
):
    module, _ = observable_module
    sources = [
        DocumentSource(
            value=f"Nguồn số {index} chứa dữ liệu quan sát riêng biệt.",
            source_type="text",
            metadata={"document_id": "multi-source-guide"},
            file_name=f"source-{index}.txt",
        )
        for index in range(5)
    ]

    result = module.knowledge.ingest(sources, include_trace=True)

    assert isinstance(result, IngestionResult)
    source = result.diagnostics["trace"]["source"]
    assert source["source_count"] == 5
    assert len(source["items"]) == 2
    assert [item["file_name"] for item in source["items"]] == [
        "source-0.txt",
        "source-1.txt",
    ]
    assert source["sources_truncated"] is True
    assert result.diagnostics["trace"]["truncation"]["sources_truncated"] is True
    _assert_json_safe(result.diagnostics)


@pytest.mark.integration
def test_runtime_snapshot_and_active_chunk_inspection_are_public_and_bounded(
    observable_module,
):
    module, _ = observable_module
    result = module.knowledge.ingest(
        _long_source(),
        source_type="text",
        document_id="observable-guide",
        file_name="observable-guide.txt",
    )
    assert result.chunk_count > 2, "the inspection limit must actually truncate rows"

    snapshot = module.runtime_snapshot()
    assert set(snapshot) == {"configured", "resolved", "config_fingerprint"}
    assert len(snapshot["config_fingerprint"]) == 64
    assert snapshot["configured"] == module.config.snapshot()
    assert set(snapshot["resolved"]) == {
        "parser",
        "chunking",
        "embedding",
        "sparse",
        "retrieval",
        "reranker",
        "storage",
        "lifecycle",
        "observability",
    }
    resolved = snapshot["resolved"]
    assert resolved["chunking"]["effective_max_tokens"] == 11
    assert resolved["chunking"]["min_chunk_tokens"] == 1
    assert resolved["embedding"] == {
        **ObservableEmbedder.resolved_config,
        "max_sequence_length": ObservableEmbedder.max_length,
        "normalized": True,
    }
    assert resolved["sparse"]["analyzer"] == "unicode61"
    assert resolved["retrieval"]["default_mode"] == "hybrid"
    assert resolved["reranker"]["model_name"] == ObservableReranker.model_name
    assert resolved["reranker"]["available"] is True
    assert resolved["storage"]["active_document_count"] == 1
    assert resolved["storage"]["active_version_count"] == 1
    assert resolved["lifecycle"]["visibility_epoch"] is not None
    assert resolved["observability"] == module.config.snapshot()["observability"]

    ordinary = module.knowledge.inspect("observable-guide")
    assert ordinary["document_id"] == "observable-guide"
    assert "active_chunks" not in ordinary

    inspection = module.knowledge.inspect("observable-guide", include_chunks=True)
    assert set(inspection) == {
        "documents",
        "active_chunks",
        "active_chunk_count",
        "inspected_chunk_count",
        "chunk_limit",
        "chunk_limit_reached",
        "chunks_truncated",
    }
    inspected_document = inspection["documents"][0]
    assert {
        key: value
        for key, value in inspected_document.items()
        if key not in {"chunk_count", "inspected_chunk_count"}
    } == ordinary
    assert inspected_document["chunk_count"] == result.chunk_count
    assert inspected_document["inspected_chunk_count"] == 2
    assert inspection["active_chunk_count"] == result.chunk_count
    assert inspection["inspected_chunk_count"] == 2
    assert inspection["chunk_limit"] == 2
    assert inspection["chunk_limit_reached"] is True
    assert inspection["chunks_truncated"] is True
    assert len(inspection["active_chunks"]) == 2
    for chunk in inspection["active_chunks"]:
        assert set(chunk) == {
            "chunk_id",
            "document_id",
            "version",
            "state",
            "text",
            "metadata",
            "text_truncated",
            "diagnostic_bounds",
        }
        assert chunk["document_id"] == "observable-guide"
        assert chunk["version"] == 1
        assert chunk["state"] == "active"
        assert chunk["text"]
        assert chunk["text_truncated"] is True
        assert chunk["diagnostic_bounds"]["strings_truncated"] >= 1

    _assert_json_safe(snapshot)
    _assert_json_safe(inspection)
    _assert_no_full_vectors(snapshot)
    _assert_no_full_vectors(inspection)


@pytest.mark.integration
def test_unchanged_ingestion_trace_truthfully_reports_skipped_work(
    observable_module,
):
    module, embedder = observable_module
    first = module.knowledge.ingest(
        _long_source(),
        source_type="text",
        document_id="unchanged-guide",
        file_name="unchanged-guide.txt",
    )
    document_embedding_calls = len(embedder.document_calls)

    unchanged = module.knowledge.update(
        _long_source(),
        source_type="text",
        document_id="unchanged-guide",
        file_name="unchanged-guide.txt",
        include_trace=True,
    )

    assert unchanged.action == "unchanged"
    assert unchanged.version == first.version == 1
    assert unchanged.chunk_count == first.chunk_count
    assert len(embedder.document_calls) == document_embedding_calls
    trace = unchanged.diagnostics["trace"]
    assert trace["embedding"]["status"] == "skipped"
    assert trace["embedding"]["elapsed_ms"] == 0.0
    assert trace["embedding"]["vector_previews"] == []
    assert trace["sparse"]["status"] == "unchanged"
    assert trace["sparse"]["elapsed_ms"] == 0.0
    assert trace["lifecycle"]["previous_version"] == 1
    assert trace["lifecycle"]["version"] == 1
    assert trace["lifecycle"]["active"] is True
    assert trace["lifecycle"]["transitions"] == [
        {"stage": "content_hash_match", "status": "unchanged"}
    ]
    assert len(trace["chunks"]) == 2
    assert trace["truncation"]["chunks_truncated"] is True
    _assert_json_safe(unchanged.diagnostics)
    _assert_no_full_vectors(unchanged.diagnostics)


@pytest.mark.integration
def test_retrieval_trace_has_mode_fusion_and_rerank_semantics(observable_module):
    module, embedder = observable_module
    module.knowledge.ingest(
        _long_source(),
        source_type="text",
        document_id="observable-guide",
        file_name="observable-guide.txt",
    )

    embedder.query_calls.clear()
    sparse = module.retrieve(
        "hồng ngọc",
        mode="sparse",
        top_k=1,
        candidate_k=4,
        include_trace=True,
    )
    assert sparse.candidates
    assert embedder.query_calls == []
    sparse_trace = sparse.diagnostics["trace"]
    assert sparse_trace["query"]["embedding_executed"] is False
    assert sparse_trace["query"]["sparse_executed"] is True
    assert sparse_trace["query_embedding"] == {
        "executed": False,
        "reason": "Sparse retrieval does not execute query embedding.",
    }
    assert sparse_trace["dense_candidates"] == []
    assert sparse_trace["sparse_candidates"]
    assert sparse_trace["fusion"]["executed"] is False
    assert sparse_trace["fusion"]["candidates"] == []
    assert sparse_trace["latency"]["fusion_ms"] == 0.0
    assert sparse_trace["truncation"]["fusion_candidates_truncated"] is False
    assert sparse_trace["rerank"]["executed"] is False
    assert sparse_trace["rerank"]["before"] == []
    assert sparse_trace["rerank"]["after"] == []

    dense = module.retrieve(
        "hồng ngọc",
        mode="dense",
        top_k=1,
        candidate_k=4,
        include_trace=True,
    )
    dense_trace = dense.diagnostics["trace"]
    assert dense_trace["query"]["embedding_executed"] is True
    assert dense_trace["query"]["sparse_executed"] is False
    assert dense_trace["query_embedding"]["executed"] is True
    assert dense_trace["query_embedding"]["dimension"] == 5
    assert len(dense_trace["query_embedding"]["vector_preview"]) == 2
    assert dense_trace["query_embedding"]["vector_omitted"] is True
    assert dense_trace["dense_candidates"]
    assert dense_trace["sparse_candidates"] == []
    assert dense_trace["fusion"]["executed"] is False
    assert dense_trace["fusion"]["candidates"] == []
    assert dense_trace["latency"]["fusion_ms"] == 0.0
    assert dense_trace["truncation"]["fusion_candidates_truncated"] is False
    assert dense_trace["rerank"]["before"] == []
    assert dense_trace["rerank"]["after"] == []

    hybrid = module.retrieve(
        "  ho\u0302\u0300ng ngo\u0323c  ",
        mode="hybrid",
        top_k=2,
        candidate_k=4,
        rerank=True,
        include_trace=True,
    )
    trace = hybrid.diagnostics["trace"]
    assert set(trace) == {
        "query",
        "query_embedding",
        "dense_candidates",
        "sparse_candidates",
        "fusion",
        "rerank",
        "latency",
        "truncation",
    }
    assert set(trace["query"]) == {
        "original",
        "normalized",
        "mode",
        "embedding_executed",
        "sparse_executed",
        "rerank_requested",
        "normalization_changed",
        "preparation_elapsed_ms",
    }
    assert trace["query"]["mode"] == "hybrid"
    assert trace["query"]["original"] == "  ho\u0302\u0300ng ngo\u0323c  "
    assert trace["query"]["normalized"] == "hồng ngọc"
    assert trace["query"]["normalization_changed"] is True
    assert trace["query"]["embedding_executed"] is True
    assert trace["query"]["sparse_executed"] is True
    assert trace["query"]["rerank_requested"] is True
    assert len(trace["dense_candidates"]) <= 2
    assert len(trace["sparse_candidates"]) <= 2
    assert all(len(row["text"]) <= 24 for row in trace["dense_candidates"])
    assert all(len(row["text"]) <= 24 for row in trace["sparse_candidates"])

    fusion = trace["fusion"]
    assert set(fusion) == {
        "executed",
        "reason",
        "strategy",
        "parameters",
        "candidates",
    }
    assert fusion["executed"] is True
    assert fusion["reason"] is None
    assert fusion["strategy"] == "weighted"
    assert fusion["parameters"] == {"dense_weight": 0.5, "sparse_weight": 0.5}
    assert fusion["candidates"]
    dense_ranks = {
        row["chunk_id"]: row["dense_rank"] for row in trace["dense_candidates"]
    }
    sparse_ranks = {
        row["chunk_id"]: row["sparse_rank"] for row in trace["sparse_candidates"]
    }
    for row in fusion["candidates"]:
        if row["chunk_id"] in dense_ranks:
            assert row["dense_rank"] == dense_ranks[row["chunk_id"]]
        else:
            assert row["dense_rank"] is None or row["dense_rank"] > 2
        if row["chunk_id"] in sparse_ranks:
            assert row["sparse_rank"] == sparse_ranks[row["chunk_id"]]
        else:
            assert row["sparse_rank"] is None or row["sparse_rank"] > 2
        assert row["fusion_rank"] is not None

    rerank = trace["rerank"]
    assert set(rerank) == {
        "executed",
        "reason",
        "model",
        "elapsed_ms",
        "applied",
        "fallback",
        "error",
        "before",
        "after",
    }
    assert rerank["executed"] is True
    assert rerank["reason"] is None
    assert rerank["model"] == ObservableReranker.model_name
    assert rerank["applied"] is True
    assert rerank["fallback"] is False
    assert rerank["error"] is None
    assert rerank["before"]
    assert rerank["after"]
    assert [row["chunk_id"] for row in rerank["after"]] == [
        candidate.chunk_id for candidate in hybrid.candidates
    ]
    assert all(isinstance(candidate, RetrievalCandidate) for candidate in hybrid.candidates)
    assert set(trace["latency"]) == {
        "query_embedding_ms",
        "dense_search_ms",
        "sparse_search_ms",
        "fusion_ms",
        "candidate_selection_ms",
        "rerank_ms",
        "total_ms",
        "query_preparation_ms",
    }
    assert all(value >= 0 for value in trace["latency"].values())

    _assert_json_safe(hybrid.to_dict())
    _assert_no_full_vectors(hybrid.to_dict())


@pytest.mark.integration
def test_include_trace_false_keeps_public_results_lean(observable_module):
    module, _ = observable_module
    ingested = module.knowledge.ingest(
        _long_source(),
        source_type="text",
        document_id="lean-guide",
        file_name="lean-guide.txt",
        include_trace=False,
    )
    assert "trace" not in ingested.diagnostics
    assert "ingestion_trace" not in ingested.diagnostics

    retrieved = module.retrieve(
        "hồng ngọc",
        mode="hybrid",
        top_k=1,
        candidate_k=4,
        include_trace=False,
    )
    assert retrieved.candidates
    assert "trace" not in retrieved.diagnostics
    assert "retrieval_trace" not in retrieved.diagnostics
    assert not any(key.startswith("trace_") for key in retrieved.diagnostics)
    _assert_json_safe(retrieved.to_dict())
