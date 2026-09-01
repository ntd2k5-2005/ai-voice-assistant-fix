import time

import pytest

from rag.contracts import MetadataFilter
from rag.retrieval import HybridRetriever, SparseRetriever
from rag.storage.qdrant import QdrantStore


def test_dense_store_persists_checkpoint_and_profile_identity(tmp_path):
    store = QdrantStore(
        "identity-v3",
        2,
        str(tmp_path / "identity"),
        embedding_model="model/name",
        normalization=True,
        embedding_revision="commit-a",
        embedding_profile_hash="profile-a",
    )
    try:
        identity = store.index_identity
        assert identity["embedding_revision"] == "commit-a"
        assert identity["embedding_profile_hash"] == "profile-a"
        store.ensure_embedding_identity(
            "model/name",
            2,
            True,
            revision="commit-a",
            profile_hash="profile-a",
        )
        with pytest.raises(ValueError, match="revision mismatch"):
            store.ensure_embedding_identity(
                "model/name",
                2,
                True,
                revision="commit-b",
                profile_hash="profile-a",
            )
    finally:
        store.close()


def test_existing_vectors_without_strict_model_identity_require_reindex(tmp_path):
    store = QdrantStore(
        "legacy-identity",
        2,
        str(tmp_path / "legacy"),
        embedding_model="model/name",
        normalization=True,
    )
    try:
        store.upsert_chunks(
            [
                {
                    "chunk_id": "legacy-chunk",
                    "document_id": "legacy-doc",
                    "version": 1,
                    "text": "legacy vector",
                    "vector": [1.0, 0.0],
                    "metadata": {
                        "chunk_id": "legacy-chunk",
                        "document_id": "legacy-doc",
                        "version": 1,
                    },
                }
            ]
        )
        with pytest.raises(ValueError, match="reindex is required"):
            store.ensure_embedding_identity(
                "model/name",
                2,
                True,
                revision="commit-a",
                profile_hash="profile-a",
            )
    finally:
        store.close()


def test_reranker_timeout_returns_deterministic_fusion_fallback():
    chunks = [{"_id": "a", "text": "alpha beta", "metadata": {}}]

    class SlowReranker:
        def rerank_candidates(self, candidates, query, top_k=10):
            time.sleep(0.05)
            return candidates[:top_k]

    result = HybridRetriever(
        sparse=SparseRetriever(chunks),
        reranker=SlowReranker(),
        rerank_timeout_ms=1,
    ).retrieve("alpha", top_k=1, candidate_k=1, rerank=True)
    assert result.diagnostics["fallback"] is True
    assert result.diagnostics["rerank_error"] == "TimeoutError"
    assert result.diagnostics["rerank_applied"] is False
    assert result.candidates[0].final_score is not None


def test_qdrant_persists_embedding_identity(tmp_path):
    first = QdrantStore("identity", 2, str(tmp_path / "qdrant"), embedding_model="model-a")
    first.client.close()
    with pytest.raises(ValueError, match="embedding model"):
        QdrantStore("identity", 2, str(tmp_path / "qdrant"), embedding_model="model-b")


def test_qdrant_applies_contract_filter_before_top_k(tmp_path):
    store = QdrantStore("filters", 2, str(tmp_path / "qdrant"))
    store.upsert_chunks([
        {"_id": "a", "document_id": "a", "text": "alpha", "vector": [1.0, 0.0], "metadata": {"group": "a"}},
        {"_id": "b", "document_id": "b", "text": "beta", "vector": [0.9, 0.1], "metadata": {"group": "b"}},
    ])
    hits = store.search([1.0, 0.0], top_k=1, query_filter=MetadataFilter("group", "=", "b"))
    assert [hit["document_id"] for hit in hits] == ["b"]
    store.client.close()


def test_qdrant_contains_filter_uses_exact_complete_fallback(tmp_path):
    store = QdrantStore("contains", 2, str(tmp_path / "qdrant"))
    store.upsert_chunks([
        {"_id": "a", "document_id": "a", "text": "alpha", "vector": [1.0, 0.0], "metadata": {"tags": ["private"]}},
        {"_id": "b", "document_id": "b", "text": "beta", "vector": [0.9, 0.1], "metadata": {"tags": ["allowed"]}},
    ])
    hits = store.search([1.0, 0.0], top_k=1, query_filter=MetadataFilter("tags", "contains", "allowed"))
    assert [hit["document_id"] for hit in hits] == ["b"]
    store.client.close()
