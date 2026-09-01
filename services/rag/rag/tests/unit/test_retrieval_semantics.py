from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from rag.contracts import MetadataFilter, RetrievalCandidate
from rag.retrieval import (
    DenseRetriever,
    ReciprocalRankFusion,
    RetrievalEngine,
    WeightedScoreFusion,
)


class CountingEmbedder:
    model_name = "fake/model"
    embedding_dim = 2

    def __init__(self):
        self.query_calls = []

    def embed_query(self, query):
        self.query_calls.append(query)
        return [1.0, 0.0]


class DenseStore:
    dim = 2

    def __init__(self, hits=None):
        self.hits = list(hits or [])
        self.calls = []
        self.identities = []

    def ensure_embedding_identity(self, model_name, dimension):
        self.identities.append((model_name, dimension))

    def search(self, vector, top_k=10, query_filter=None):
        self.calls.append((list(vector), top_k, query_filter))
        return self.hits[:top_k]


class SparseSpy:
    def __init__(self, candidates=None):
        self.candidates = list(candidates or [])
        self.calls = []

    def search(self, query, top_k=10, filters=None):
        self.calls.append((query, top_k, filters))
        return self.candidates[:top_k]


class CountingReranker:
    def __init__(self):
        self.calls = []

    def rerank_candidates(self, candidates, query, top_k=10):
        self.calls.append((query, len(candidates), top_k))
        return [
            (candidate, 10.0 if "best" in candidate.text else float(index))
            for index, candidate in enumerate(candidates)
        ][:top_k]


def dense_hits():
    return [
        {"_id": "a", "text": "best dense", "score": 0.8, "metadata": {"group": "x"}},
        {"_id": "b", "text": "other dense", "score": -0.2, "metadata": {"group": "y"}},
    ]


def sparse_hits():
    return [
        RetrievalCandidate(
            chunk_id="a", text="best dense", retrieval_score=20.0, sparse_score=20.0
        ),
        RetrievalCandidate(
            chunk_id="c", text="lexical", retrieval_score=5.0, sparse_score=5.0
        ),
    ]


def make_engine(*, reranker=None, **kwargs):
    embedder = CountingEmbedder()
    store = DenseStore(dense_hits())
    dense = DenseRetriever(embedder, store)
    sparse = SparseSpy(sparse_hits())
    engine = RetrievalEngine(dense=dense, sparse=sparse, reranker=reranker, **kwargs)
    return engine, embedder, store, sparse


def test_sparse_mode_never_embeds_and_dense_mode_never_reranks_implicitly():
    reranker = CountingReranker()
    engine, embedder, _, sparse = make_engine(reranker=reranker)

    sparse_result = engine.retrieve("lexical", mode="sparse", top_k=1, candidate_k=2)
    assert sparse_result.mode == "sparse"
    assert embedder.query_calls == []
    assert len(sparse.calls) == 1

    dense_result = engine.retrieve(
        "semantic", mode="dense", top_k=1, candidate_k=2, rerank=False
    )
    assert dense_result.mode == "dense"
    assert embedder.query_calls == ["semantic"]
    assert reranker.calls == []
    assert dense_result.diagnostics["rerank_applied"] is False


def test_hybrid_normalizes_public_scores_and_preserves_raw_diagnostics():
    engine, embedder, _, sparse = make_engine()
    result = engine.retrieve("query", mode="hybrid", top_k=3, candidate_k=3)
    by_id = {candidate.chunk_id: candidate for candidate in result.candidates}
    assert embedder.query_calls == ["query"]
    assert len(sparse.calls) == 1
    assert by_id["a"].dense_score == pytest.approx(0.9)
    assert by_id["a"].sparse_score == pytest.approx(1.0)
    assert by_id["b"].dense_score == pytest.approx(0.4)
    assert by_id["c"].sparse_score == pytest.approx(0.0)
    assert by_id["a"].dense_raw_score == pytest.approx(0.8)
    assert by_id["a"].sparse_raw_score == pytest.approx(20.0)
    assert by_id["b"].dense_raw_score == pytest.approx(-0.2)
    assert by_id["c"].sparse_raw_score == pytest.approx(5.0)
    assert all(0.0 <= candidate.final_score <= 1.0 for candidate in result.candidates)
    assert all(candidate.fusion_score is not None for candidate in result.candidates)
    assert result.diagnostics["fusion_strategy"] == "rrf"
    assert result.diagnostics["score_semantics"]["public_range"] == [0.0, 1.0]
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3]


def test_single_mode_public_scores_are_bounded_and_identify_final_source():
    engine, _, _, _ = make_engine()

    dense = engine.retrieve("q", mode="dense", top_k=2, candidate_k=2)
    assert [candidate.final_score_source for candidate in dense.candidates] == ["dense", "dense"]
    assert dense.candidates[0].dense_score == pytest.approx(0.9)
    assert dense.candidates[0].dense_raw_score == pytest.approx(0.8)
    assert dense.candidates[0].final_score == pytest.approx(dense.candidates[0].dense_score)

    sparse = engine.retrieve("q", mode="sparse", top_k=2, candidate_k=2)
    assert [candidate.final_score_source for candidate in sparse.candidates] == ["sparse", "sparse"]
    assert [candidate.sparse_score for candidate in sparse.candidates] == pytest.approx([1.0, 0.0])
    assert [candidate.sparse_raw_score for candidate in sparse.candidates] == pytest.approx([20.0, 5.0])
    assert all(0.0 <= candidate.final_score <= 1.0 for candidate in [*dense.candidates, *sparse.candidates])


def test_candidate_k_top_k_and_filter_propagation_are_exact():
    engine, _, store, sparse = make_engine()
    filters = [MetadataFilter("group", "=", "x")]
    result = engine.retrieve(
        "q", mode="hybrid", top_k=1, candidate_k=2, filters=filters
    )
    assert len(result.candidates) == 1
    assert store.calls[0][1:] == (2, filters)
    assert sparse.calls[0][1:] == (2, filters)
    assert result.diagnostics["candidate_k"] == 2
    with pytest.raises(ValueError, match="candidate_k"):
        engine.retrieve("q", mode="hybrid", top_k=3, candidate_k=2)


def test_rerank_is_independent_for_dense_sparse_and_hybrid_modes():
    for mode in ("dense", "sparse", "hybrid"):
        reranker = CountingReranker()
        engine, _, _, _ = make_engine(reranker=reranker)
        result = engine.retrieve("q", mode=mode, top_k=1, candidate_k=2, rerank=True)
        assert reranker.calls == [("q", 2, 1)]
        assert result.mode == mode
        assert result.diagnostics["rerank_requested"] is True
        assert result.diagnostics["rerank_applied"] is True
        assert result.diagnostics["rerank_fallback"] is False
        assert result.candidates[0].rerank_score is not None
        assert 0.0 <= result.candidates[0].rerank_score <= 1.0
        assert result.candidates[0].rerank_raw_score is not None
        assert result.candidates[0].final_score_source == "rerank"
        assert result.candidates[0].final_score == pytest.approx(result.candidates[0].rerank_score)


def test_requested_but_unavailable_reranker_reports_truthful_fallback():
    engine, _, _, _ = make_engine(reranker=None)
    result = engine.retrieve("q", mode="dense", top_k=1, candidate_k=2, rerank=True)
    assert result.diagnostics["rerank_requested"] is True
    assert result.diagnostics["rerank_available"] is False
    assert result.diagnostics["rerank_applied"] is False
    assert result.diagnostics["rerank_fallback"] is True
    assert result.diagnostics["rerank_error"] == "RuntimeError"
    assert result.candidates[0].rerank_score is None


def test_empty_pool_does_not_claim_rerank_execution():
    engine = RetrievalEngine(dense=DenseRetriever(CountingEmbedder(), DenseStore([])))
    result = engine.retrieve("q", mode="dense", top_k=1, candidate_k=1, rerank=True)
    assert result.candidates == []
    assert result.diagnostics["rerank_requested"] is True
    assert result.diagnostics["rerank_applied"] is False
    assert result.diagnostics["rerank_fallback"] is False
    assert result.diagnostics["rerank_candidates"] == 0


def test_timeout_and_invalid_scores_fall_back_to_identical_pre_rerank_order():
    class SlowReranker:
        def rerank_candidates(self, candidates, query, top_k=10):
            time.sleep(0.05)
            return [(candidate, 1.0) for candidate in candidates[:top_k]]

    engine, _, _, _ = make_engine(reranker=SlowReranker(), rerank_timeout_ms=1)
    baseline = engine.retrieve("q", mode="hybrid", top_k=2, candidate_k=3)
    timed_out = engine.retrieve(
        "q", mode="hybrid", top_k=2, candidate_k=3, rerank=True
    )
    assert [item.chunk_id for item in timed_out.candidates] == [
        item.chunk_id for item in baseline.candidates
    ]
    assert timed_out.diagnostics["rerank_error"] == "TimeoutError"
    assert timed_out.diagnostics["rerank_applied"] is False

    class InvalidReranker:
        def rerank_candidates(self, candidates, query, top_k=10):
            return [(candidates[0], float("nan"))]

    invalid, _, _, _ = make_engine(reranker=InvalidReranker())
    result = invalid.retrieve("q", mode="hybrid", top_k=1, candidate_k=2, rerank=True)
    assert result.diagnostics["rerank_fallback"] is True
    assert result.diagnostics["rerank_error"] == "RuntimeError"


def test_rerank_failure_can_be_configured_to_fail_request():
    class Broken:
        def rerank_candidates(self, candidates, query, top_k=10):
            raise LookupError("bad model")

    engine, _, _, _ = make_engine(reranker=Broken(), rerank_failure="raise")
    with pytest.raises(LookupError, match="bad model"):
        engine.retrieve("q", mode="dense", top_k=1, candidate_k=2, rerank=True)


def test_weighted_and_rrf_fusion_handle_equal_negative_and_empty_branches():
    dense = [
        RetrievalCandidate(chunk_id="b", retrieval_score=0.4, dense_score=0.4, dense_raw_score=-0.2),
        RetrievalCandidate(chunk_id="a", retrieval_score=0.4, dense_score=0.4, dense_raw_score=-0.2),
    ]
    weighted = WeightedScoreFusion().fuse(dense, [], limit=2)
    assert [candidate.chunk_id for candidate in weighted] == ["a", "b"]
    assert all(candidate.fusion_score == pytest.approx(0.4) for candidate in weighted)
    assert all(candidate.dense_score == pytest.approx(0.4) for candidate in weighted)

    sparse = [
        RetrievalCandidate(chunk_id="b", retrieval_score=1.0, sparse_score=1.0, sparse_raw_score=100),
        RetrievalCandidate(chunk_id="c", retrieval_score=0.0, sparse_score=0.0, sparse_raw_score=50),
    ]
    rrf = ReciprocalRankFusion(rank_constant=10).fuse(dense, sparse)
    assert rrf[0].chunk_id == "b"
    assert all(candidate.fusion_score is not None for candidate in rrf)


def test_rrf_strategy_is_configurable_and_deterministic():
    engine, _, _, _ = make_engine(fusion_strategy="rrf", rrf_k=10)
    first = engine.retrieve("q", mode="hybrid", top_k=3, candidate_k=3)
    second = engine.retrieve("q", mode="hybrid", top_k=3, candidate_k=3)
    assert [candidate.chunk_id for candidate in first.candidates] == [
        candidate.chunk_id for candidate in second.candidates
    ]
    assert first.diagnostics["fusion_strategy"] == "rrf"


def test_parallel_hybrid_and_query_length_validation():
    engine, embedder, _, sparse = make_engine(parallel_branches=True, max_query_chars=5)
    result = engine.retrieve("short", mode="hybrid", top_k=1, candidate_k=2)
    assert result.diagnostics["parallel_branches"] is True
    assert len(embedder.query_calls) == len(sparse.calls) == 1
    with pytest.raises(ValueError, match="maximum"):
        engine.retrieve("too long", mode="hybrid", top_k=1, candidate_k=2)
    with pytest.raises(TypeError, match="string"):
        engine.retrieve(123, mode="hybrid", top_k=1, candidate_k=2)


def test_sparse_rerank_policy_is_enforced():
    engine, _, _, _ = make_engine(
        reranker=CountingReranker(), allow_sparse_rerank=False
    )
    with pytest.raises(ValueError, match="disabled"):
        engine.retrieve("q", mode="sparse", top_k=1, candidate_k=2, rerank=True)


def test_dense_retriever_propagates_full_embedding_identity_when_supported():
    embedder = CountingEmbedder()
    embedder.identity = SimpleNamespace(
        model_name="fake/model", dimension=2, normalized=True
    )

    class IdentityStore(DenseStore):
        def ensure_embedding_identity(self, model_name, dimension, normalization=None):
            self.identities.append((model_name, dimension, normalization))

    store = IdentityStore([])
    DenseRetriever(embedder, store)
    assert store.identities == [("fake/model", 2, True)]


def test_public_serialization_hides_raw_scores_but_trace_keeps_them():
    engine, _, _, _ = make_engine()
    result = engine.retrieve(
        "query", mode="hybrid", top_k=2, candidate_k=3, include_trace=True
    )
    public_candidate = result.to_dict()["candidates"][0]
    assert "dense_raw_score" not in public_candidate
    assert "sparse_raw_score" not in public_candidate
    assert "fusion_raw_score" not in public_candidate
    assert 0.0 <= public_candidate["final_score"] <= 1.0

    trace_rows = result.diagnostics["trace"]["fusion"]["candidates"]
    assert trace_rows
    assert any(
        row.get("dense_raw_score") is not None
        or row.get("sparse_raw_score") is not None
        for row in trace_rows
    )
