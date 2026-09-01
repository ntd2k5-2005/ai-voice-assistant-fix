import json
from dataclasses import dataclass

from rag.benchmark import (
    BenchmarkCase,
    evaluate_embedding_model,
    evaluate_reranker_model,
    evaluate_retrieval_mode,
    make_artifact,
    write_artifact,
)
from rag.contracts import RetrievalCandidate, RetrievalResult


def test_benchmark_records_raw_rankings_metrics_timings_and_artifacts(tmp_path):
    def retrieve(query, top_k, candidate_k):
        order = ["capital", "coast"] if "thủ đô" in query else ["coast", "capital"]
        return RetrievalResult(
            candidates=[RetrievalCandidate(chunk_id=item) for item in order[:top_k]],
            total_latency_ms=1.0,
            mode="dense",
            query=query,
        )

    cases = [BenchmarkCase("thủ đô", frozenset({"capital"}), case_id="q1")]
    report = evaluate_retrieval_mode(retrieve, cases, top_k=2, candidate_k=5)
    assert report["quality"]["recall@1"] == 1.0
    assert report["quality"]["mrr"] == 1.0
    assert report["queries"][0]["retrieved"] == ["capital", "coast"]

    artifact = make_artifact(
        benchmark_kind="end_to_end",
        dataset={"name": "tiny-vi", "cases": 1},
        modes={"dense": report},
        config_snapshot={"retrieval": {"top_k": 2}},
        seed=42,
        model_identity={"embedding": "deterministic-test"},
    )
    paths = write_artifact(artifact, tmp_path / "result")
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["environment"]["seed"] == 42
    assert "dense,q1" in paths["csv"].read_text(encoding="utf-8-sig")
    assert "Generated from" in paths["markdown"].read_text(encoding="utf-8")


def test_benchmark_can_score_document_identity():
    @dataclass
    class Candidate:
        chunk_id: str
        document_id: str

    report = evaluate_retrieval_mode(
        lambda query, **kwargs: [Candidate("chunk-1", "doc-1")],
        [{"query": "q", "relevant": ["doc-1"]}],
        top_k=1,
        candidate_k=1,
        identity_field="document_id",
    )
    assert report["quality"]["recall@1"] == 1.0


def test_model_benchmark_uses_distinct_document_and_query_apis_in_order():
    class Embedder:
        def __init__(self):
            self.document_calls = []
            self.query_calls = []

        def embed_documents(self, texts):
            self.document_calls.append(list(texts))
            return [[1.0, 0.0], [0.0, 1.0]]

        def embed_queries_batch(self, queries):
            self.query_calls.append(list(queries))
            return [[1.0, 0.0]]

    embedder = Embedder()
    report = evaluate_embedding_model(
        embedder,
        {"capital": "Hà Nội", "coast": "Đà Nẵng"},
        [{"query": "thủ đô", "relevant": ["capital"]}],
        top_k=2,
    )
    assert report["quality"]["recall@1"] == 1.0
    assert report["dimension"] == 2
    assert embedder.document_calls == [["Hà Nội", "Đà Nẵng"]]
    assert embedder.query_calls == [["thủ đô"]]


def test_reranker_benchmark_uses_fixed_candidate_pool():
    class Reranker:
        def rerank_candidates(self, candidates, query, top_k):
            return sorted(candidates, key=lambda item: item.chunk_id)[:top_k]

    report = evaluate_reranker_model(
        Reranker(),
        [
            {
                "query": "capital",
                "relevant": ["a"],
                "candidates": [
                    {"id": "b", "text": "coast"},
                    {"id": "a", "text": "capital"},
                ],
            }
        ],
        top_k=2,
    )
    assert report["quality"]["recall@1"] == 1.0
