from __future__ import annotations

from dataclasses import replace

import pytest

from rag.config import RagConfig
from rag.contracts import IngestionResult, RetrievalCandidate, RetrievalResult
from rag.streamlit_app.utils import config_fingerprint, resolve_config
from rag.streamlit_app.view_models import (
    UNAVAILABLE,
    build_config_view,
    build_ingestion_view,
    build_inspection_view,
    build_retrieval_view,
    candidate_row,
    chunk_statistics,
    finite_vector_summary,
)


def test_streamlit_uses_rag_config_unchanged_without_explicit_overrides():
    central = RagConfig()
    effective, overrides = resolve_config({}, base=central)
    assert effective is central
    assert effective.snapshot() == central.snapshot()
    assert overrides == ()


def test_streamlit_overrides_replace_only_explicit_central_fields(tmp_path):
    central = RagConfig()
    effective, overrides = resolve_config(
        {
            "RAG_STREAMLIT_DATA_DIR": str(tmp_path),
            "RAG_EMBEDDING_DEVICE": "cpu",
            "RAG_EMBEDDING_LOCAL_FILES_ONLY": "true",
            "RAG_RERANKER_MODEL": "org/explicit-model",
        },
        base=central,
    )
    assert effective.parser is central.parser
    assert effective.chunking is central.chunking
    assert effective.retrieval is central.retrieval
    assert effective.embedding.model_name == central.embedding.model_name
    assert effective.embedding.device == "cpu"
    assert effective.embedding.local_files_only is True
    assert effective.reranker.model_name == "org/explicit-model"
    assert effective.paths.qdrant == tmp_path / "qdrant"
    assert {item.environment for item in overrides} == {
        "RAG_STREAMLIT_DATA_DIR",
        "RAG_EMBEDDING_DEVICE",
        "RAG_EMBEDDING_LOCAL_FILES_ONLY",
        "RAG_RERANKER_MODEL",
    }


def test_invalid_boolean_override_fails_before_runtime_construction():
    with pytest.raises(ValueError, match="RAG_EMBEDDING_LOCAL_FILES_ONLY"):
        resolve_config({"RAG_EMBEDDING_LOCAL_FILES_ONLY": "sometimes"})


def test_reranker_can_be_explicitly_disabled_without_a_hidden_ui_default():
    central = RagConfig()
    effective, overrides = resolve_config(
        {"RAG_RERANKER_MODEL": "disabled"},
        base=central,
    )
    assert effective.reranker.model_name is None
    assert overrides[0].setting == "reranker.model_name"
    assert overrides[0].value is None


def test_resource_fingerprint_covers_the_complete_effective_config():
    first = RagConfig()
    second = replace(
        first,
        retrieval=replace(first.retrieval, candidate_k=first.retrieval.candidate_k + 1),
    )
    assert config_fingerprint(first) == config_fingerprint(RagConfig())
    assert config_fingerprint(first) != config_fingerprint(second)


def test_config_view_distinguishes_configured_resolved_and_unavailable():
    config = RagConfig()
    view = build_config_view(
        config,
        {
            "configured": config.snapshot(),
            "resolved": {
                "chunking": {"effective_max_tokens": 448, "tokenizer": "model-tokenizer"},
                "embedding": {
                    "model_name": config.embedding.model_name,
                    "family": "e5",
                    "dimension": 384,
                    "device": "cpu",
                    "dtype": "float32",
                    "max_sequence_length": 512,
                    "batch_size": 16,
                    "normalized": True,
                    "revision": "abc123",
                },
            },
        },
    )
    chunk_rows = {row["setting"]: row for row in view["Chunking"]}
    embedding_rows = {row["setting"]: row for row in view["Embedding"]}
    storage_rows = {row["setting"]: row for row in view["Storage / lifecycle"]}
    assert chunk_rows["Effective maximum tokens"]["resolved"] == 448
    assert embedding_rows["Dimension"]["resolved"] == 384
    assert storage_rows["Active documents"]["resolved"] == UNAVAILABLE
    assert storage_rows["Active documents"]["status"] == "unavailable"


def test_ingestion_view_exposes_every_flow1_stage_and_exact_chunk_statistics():
    result = IngestionResult(
        document_id="guide",
        action="update",
        version=2,
        chunk_count=2,
        diagnostics={
            "operation_id": "op-2",
            "trace": {
                "source": {
                    "file_name": "guide.pdf",
                    "file_type": ".pdf",
                    "size_bytes": 1234,
                    "source_identity": "sha256:abc",
                    "parser": "pdf",
                },
                "parsed_units": [{"unit_index": 0, "text": "Raw", "page": 1}],
                "normalized_units": [{"unit_index": 0, "text": "Normalized", "page": 1}],
                "chunks": [
                    {"index": 0, "chunk_id": "c1", "token_count": 10, "text": "one"},
                    {"index": 1, "chunk_id": "c2", "token_count": 20, "text": "two"},
                ],
                "embedding": {
                    "model": "embed",
                    "dimension": 3,
                    "chunk_count": 2,
                    "finite": True,
                },
                "sparse": {"chunk_count": 2, "analyzer": "unicode61", "status": "verified"},
                "lifecycle": {
                    "document_id": "guide",
                    "previous_version": 1,
                    "version": 2,
                    "operation_id": "op-2",
                    "transitions": ["staged", "verified", "active"],
                    "active": True,
                },
            },
        },
    )
    view = build_ingestion_view(result)[0]
    assert view["trace_available"] is True
    assert view["source"]["parser"] == "pdf"
    assert view["parsed_units"][0]["text"] == "Raw"
    assert view["normalized_units"][0]["text"] == "Normalized"
    assert view["chunk_statistics"] == {
        "chunk_count": 2,
        "token_counts_available": True,
        "min_tokens": 10,
        "max_tokens": 20,
        "mean_tokens": 15.0,
        "median_tokens": 15.0,
    }
    assert view["embedding"]["finite"] is True
    assert view["sparse"]["status"] == "verified"
    assert view["lifecycle"]["previous_version"] == 1


def test_chunk_statistics_never_invents_missing_token_counts():
    assert chunk_statistics([{"chunk_id": "a"}]) == {
        "chunk_count": 1,
        "token_counts_available": False,
        "min_tokens": None,
        "max_tokens": None,
        "mean_tokens": None,
        "median_tokens": None,
    }


def test_sparse_retrieval_view_states_query_embedding_was_not_executed():
    candidate = RetrievalCandidate(
        chunk_id="s1",
        document_id="doc",
        text="lexical hit",
        sparse_score=1.2,
        final_score=1.2,
    )
    result = RetrievalResult(
        candidates=[candidate],
        mode="sparse",
        query="normalized query",
        diagnostics={
            "trace": {
                "query": {
                    "original": "  normalized query ",
                    "normalized": "normalized query",
                    "mode": "sparse",
                    "embedding_executed": False,
                    "sparse_executed": True,
                    "rerank_requested": False,
                },
                "query_embedding": {
                    "executed": False,
                    "reason": "sparse retrieval mode",
                },
                "sparse_candidates": [candidate.to_dict()],
                "fusion": {"executed": False},
                "rerank": {"executed": False},
            }
        },
    )
    view = build_retrieval_view(result, original_query="  normalized query ")
    assert view["query_embedding"]["executed"] is False
    assert view["dense"]["executed"] is False
    assert view["sparse"]["executed"] is True
    assert view["sparse"]["available"] is True
    assert view["fusion"]["executed"] is False
    assert view["final"][0]["dense_score"] is None
    assert view["final"][0]["sparse_score"] == 1.2


def test_hybrid_retrieval_view_preserves_branch_fusion_and_rerank_order():
    final = RetrievalCandidate(
        chunk_id="b",
        document_id="doc",
        text="relevant",
        dense_score=0.7,
        sparse_score=2.0,
        fusion_score=0.9,
        rerank_score=4.0,
        final_score=4.0,
    )
    result = RetrievalResult(
        candidates=[final],
        mode="hybrid",
        query="query",
        rerank_requested=True,
        diagnostics={
            "rerank_applied": True,
            "trace": {
                "query": {"mode": "hybrid"},
                "query_embedding": {"executed": True, "dimension": 3},
                "dense_candidates": [{"rank": 1, "chunk_id": "a", "dense_score": 0.8}],
                "sparse_candidates": [{"rank": 1, "chunk_id": "b", "sparse_score": 2.0}],
                "fusion": {
                    "executed": True,
                    "strategy": "weighted",
                    "candidates": [
                        {"rank": 1, "chunk_id": "a", "fusion_score": 0.95},
                        {"rank": 2, "chunk_id": "b", "fusion_score": 0.9},
                    ],
                },
                "rerank": {
                    "executed": True,
                    "applied": True,
                    "before": [
                        {"rank": 1, "chunk_id": "a", "fusion_score": 0.95},
                        {"rank": 2, "chunk_id": "b", "fusion_score": 0.9},
                    ],
                    "after": [
                        {"rank": 1, "chunk_id": "b", "rerank_score": 4.0},
                        {"rank": 2, "chunk_id": "a", "rerank_score": 1.0},
                    ],
                },
            },
        },
    )
    view = build_retrieval_view(result)
    assert view["dense"]["candidates"][0]["chunk_id"] == "a"
    assert view["sparse"]["candidates"][0]["chunk_id"] == "b"
    assert view["fusion"]["candidates"][1]["chunk_id"] == "b"
    assert view["rerank"]["before"][0]["chunk_id"] == "a"
    assert view["rerank"]["after"][0]["chunk_id"] == "b"
    assert view["final"][0]["chunk_id"] == "b"


def test_candidate_scores_preserve_not_applicable_separately_from_zero():
    row = candidate_row(
        {
            "chunk_id": "c",
            "dense_score": 0.0,
            "sparse_score": None,
            "metadata": {"page": 3},
        }
    )
    assert row["dense_score"] == 0.0
    assert row["sparse_score"] is None
    assert row["page"] == 3


def test_inspection_view_supports_documents_and_active_chunks():
    view = build_inspection_view(
        {
            "documents": [
                {
                    "document_id": "doc",
                    "filename": "doc.md",
                    "version": 2,
                    "hash": "abc",
                    "status": "active",
                    "chunk_count": 1,
                }
            ],
            "active_chunks": [
                {"chunk_id": "c1", "document_id": "doc", "text": "knowledge"}
            ],
        }
    )
    assert view["empty"] is False
    assert view["summary_rows"][0]["state"] == "active"
    assert view["chunks_by_document"]["doc"][0]["chunk_id"] == "c1"


def test_finite_vector_summary_reports_dimension_norm_and_nonfinite_values():
    assert finite_vector_summary([3.0, 4.0]) == {
        "dimension": 2,
        "finite": True,
        "norm": 5.0,
    }
    assert finite_vector_summary([float("nan")])["finite"] is False
