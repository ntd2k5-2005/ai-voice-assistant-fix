"""Reproducible model and end-to-end retrieval benchmark primitives.

The module deliberately does not choose a winning model.  It records raw
rankings, quality metrics, stage timings, exact configuration, and environment
metadata so a report can be regenerated from evidence.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics import aggregate_metrics, hit_rate_at_k, mrr, ndcg_at_k, recall_at_k


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    relevant: frozenset[str]
    relevance: Mapping[str, float] = field(default_factory=dict)
    case_id: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int = 0) -> "BenchmarkCase":
        relevant = frozenset(str(item) for item in value.get("relevant", ()))
        if not relevant:
            raise ValueError("each benchmark case requires at least one relevant id")
        query = str(value.get("query", "")).strip()
        if not query:
            raise ValueError("each benchmark case requires a query")
        return cls(
            query=query,
            relevant=relevant,
            relevance={str(key): float(score) for key, score in (value.get("relevance") or {}).items()},
            case_id=str(value.get("case_id") or index),
        )


@dataclass(frozen=True)
class Timing:
    query_embedding_ms: float | None = None
    dense_search_ms: float = 0.0
    sparse_search_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


def percentiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}

    def nearest_rank(percentile: float) -> float:
        index = max(0, min(len(ordered) - 1, int(percentile * len(ordered) + 0.999999) - 1))
        return ordered[index]

    return {
        "mean": statistics.fmean(ordered),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
    }


def _candidate_id(candidate: Any, identity_field: str) -> str:
    if isinstance(candidate, Mapping):
        value = candidate.get(identity_field)
        if value is None and identity_field == "chunk_id":
            value = candidate.get("_id", candidate.get("id"))
    else:
        value = getattr(candidate, identity_field, None)
        if value is None and identity_field == "chunk_id":
            value = getattr(candidate, "id", None)
    if value is None:
        raise ValueError(f"candidate does not expose {identity_field!r}")
    return str(value)


def _timing_from_result(result: Any, measured_total_ms: float) -> Timing:
    latency = getattr(result, "latency", None)
    if latency is not None:
        return Timing(
            query_embedding_ms=getattr(latency, "query_embedding_ms", None),
            dense_search_ms=float(getattr(latency, "dense_search_ms", 0.0) or 0.0),
            sparse_search_ms=float(getattr(latency, "sparse_search_ms", 0.0) or 0.0),
            fusion_ms=float(getattr(latency, "fusion_ms", 0.0) or 0.0),
            rerank_ms=float(getattr(latency, "rerank_ms", 0.0) or 0.0),
            total_ms=measured_total_ms,
        )
    return Timing(
        query_embedding_ms=getattr(result, "embed_latency_ms", None),
        dense_search_ms=float(getattr(result, "search_latency_ms", 0.0) or 0.0),
        rerank_ms=float(getattr(result, "rerank_latency_ms", 0.0) or 0.0),
        total_ms=measured_total_ms,
    )


def evaluate_retrieval_mode(
    retrieve: Callable[..., Any],
    cases: Sequence[BenchmarkCase | Mapping[str, Any]],
    *,
    top_k: int = 10,
    candidate_k: int = 50,
    identity_field: str = "chunk_id",
    retrieve_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one retrieval configuration against fixed relevance labels."""

    if candidate_k < top_k or top_k <= 0:
        raise ValueError("candidate_k must be >= top_k > 0")
    normalized = [
        case if isinstance(case, BenchmarkCase) else BenchmarkCase.from_mapping(case, index)
        for index, case in enumerate(cases)
    ]
    kwargs = dict(retrieve_kwargs or {})
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    timings: list[Timing] = []
    for case in normalized:
        started = time.perf_counter()
        result = retrieve(case.query, top_k=top_k, candidate_k=candidate_k, **kwargs)
        total_ms = (time.perf_counter() - started) * 1000.0
        candidates = getattr(result, "candidates", result)
        retrieved = [_candidate_id(candidate, identity_field) for candidate in list(candidates)[:top_k]]
        relevance = dict(case.relevance) or {item: 1.0 for item in case.relevant}
        timing = _timing_from_result(result, total_ms)
        timings.append(timing)
        metric_rows.append({"retrieved": retrieved, "relevant": set(case.relevant), "relevance": relevance})
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "retrieved": retrieved,
                "relevant": sorted(case.relevant),
                "recall@1": recall_at_k(retrieved, case.relevant, 1),
                "recall@5": recall_at_k(retrieved, case.relevant, 5),
                "recall@10": recall_at_k(retrieved, case.relevant, 10),
                "hit@5": hit_rate_at_k(retrieved, case.relevant, 5),
                "mrr": mrr(retrieved, case.relevant),
                "ndcg@10": ndcg_at_k(retrieved, relevance, 10),
                "timing": asdict(timing),
            }
        )
    quality = aggregate_metrics(metric_rows, ks=(1, 5, 10)) if metric_rows else {}
    latency = {
        name: percentiles(
            getattr(timing, name)
            for timing in timings
            if getattr(timing, name) is not None
        )
        for name in Timing.__dataclass_fields__
    }
    return {"quality": quality, "latency_ms": latency, "queries": rows}


def benchmark_modes(
    retrievers: Mapping[str, Callable[..., Any]],
    evaluations: Sequence[BenchmarkCase | Mapping[str, Any]],
    top_k: int = 10,
    candidate_k: int = 50,
    *,
    identity_field: str = "chunk_id",
) -> dict[str, Any]:
    return {
        mode: evaluate_retrieval_mode(
            retriever,
            evaluations,
            top_k=top_k,
            candidate_k=candidate_k,
            identity_field=identity_field,
        )
        for mode, retriever in retrievers.items()
    }


def evaluate_embedding_model(
    embedder: Any,
    documents: Mapping[str, str],
    cases: Sequence[BenchmarkCase | Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Benchmark an embedding model against a fixed passage corpus.

    Document and query APIs are exercised independently, which catches models
    whose retrieval instructions differ between indexing and querying.
    """

    import numpy as np

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    document_ids = [str(item) for item in documents]
    document_texts = [str(documents[item]) for item in documents]
    normalized_cases = [
        case if isinstance(case, BenchmarkCase) else BenchmarkCase.from_mapping(case, index)
        for index, case in enumerate(cases)
    ]
    document_started = time.perf_counter()
    document_vectors = np.asarray(embedder.embed_documents(document_texts), dtype=np.float32)
    document_ms = (time.perf_counter() - document_started) * 1000.0
    if document_vectors.ndim != 2 or document_vectors.shape[0] != len(document_ids):
        raise ValueError("embed_documents returned an invalid or reordered batch")

    queries = [case.query for case in normalized_cases]
    query_started = time.perf_counter()
    if hasattr(embedder, "embed_queries_batch"):
        query_vectors = np.asarray(embedder.embed_queries_batch(queries), dtype=np.float32)
    else:
        query_vectors = np.asarray([embedder.embed_query(query) for query in queries], dtype=np.float32)
    query_ms = (time.perf_counter() - query_started) * 1000.0
    if query_vectors.ndim != 2 or query_vectors.shape[0] != len(normalized_cases):
        raise ValueError("query embedding returned an invalid or reordered batch")
    if query_vectors.shape[1] != document_vectors.shape[1]:
        raise ValueError("query/document embedding dimensions differ")

    document_norms = np.linalg.norm(document_vectors, axis=1)
    rows = []
    metric_rows = []
    for case, vector in zip(normalized_cases, query_vectors):
        denominator = document_norms * np.linalg.norm(vector)
        similarities = np.divide(
            document_vectors @ vector,
            denominator,
            out=np.full(len(document_ids), -np.inf, dtype=np.float32),
            where=denominator != 0,
        )
        indices = sorted(range(len(document_ids)), key=lambda index: (-float(similarities[index]), document_ids[index]))[:top_k]
        retrieved = [document_ids[index] for index in indices]
        relevance = dict(case.relevance) or {item: 1.0 for item in case.relevant}
        metric_rows.append({"retrieved": retrieved, "relevant": set(case.relevant), "relevance": relevance})
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "retrieved": retrieved,
                "scores": [float(similarities[index]) for index in indices],
                "relevant": sorted(case.relevant),
                "recall@1": recall_at_k(retrieved, case.relevant, 1),
                "recall@5": recall_at_k(retrieved, case.relevant, 5),
                "recall@10": recall_at_k(retrieved, case.relevant, 10),
                "hit@5": hit_rate_at_k(retrieved, case.relevant, 5),
                "mrr": mrr(retrieved, case.relevant),
                "ndcg@10": ndcg_at_k(retrieved, relevance, 10),
            }
        )
    return {
        "quality": aggregate_metrics(metric_rows, ks=(1, 5, 10)) if metric_rows else {},
        "latency_ms": {
            "document_embedding_total": document_ms,
            "query_embedding_total": query_ms,
            "query_embedding_mean": query_ms / max(len(normalized_cases), 1),
        },
        "dimension": int(document_vectors.shape[1]),
        "documents": len(document_ids),
        "queries": rows,
    }


def evaluate_reranker_model(
    reranker: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Benchmark the canonical reranker over a fixed candidate pool."""

    from .contracts import RetrievalCandidate

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    timings: list[float] = []
    for index, raw_case in enumerate(cases):
        case = BenchmarkCase.from_mapping(raw_case, index)
        candidates = []
        for candidate_index, raw_candidate in enumerate(raw_case.get("candidates", ())):
            candidate_id = raw_candidate.get("chunk_id", raw_candidate.get("id"))
            if candidate_id is None:
                raise ValueError(f"candidate {candidate_index} requires an id")
            candidates.append(
                RetrievalCandidate(
                    chunk_id=str(candidate_id),
                    text=str(raw_candidate.get("text", "")),
                    document_id=str(raw_candidate.get("document_id", "")),
                )
            )
        started = time.perf_counter()
        if hasattr(reranker, "rerank_candidates"):
            ranked = reranker.rerank_candidates(candidates, case.query, top_k=top_k)
        else:
            raw_ranked = reranker.rerank(case.query, candidates, top_k=top_k)
            ranked = [item[0] if isinstance(item, tuple) else item for item in raw_ranked]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings.append(elapsed_ms)
        retrieved = [_candidate_id(candidate, "chunk_id") for candidate in ranked]
        relevance = dict(case.relevance) or {item: 1.0 for item in case.relevant}
        metric_rows.append({"retrieved": retrieved, "relevant": set(case.relevant), "relevance": relevance})
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "retrieved": retrieved,
                "relevant": sorted(case.relevant),
                "latency_ms": elapsed_ms,
                "recall@1": recall_at_k(retrieved, case.relevant, 1),
                "recall@5": recall_at_k(retrieved, case.relevant, 5),
                "recall@10": recall_at_k(retrieved, case.relevant, 10),
                "hit@5": hit_rate_at_k(retrieved, case.relevant, 5),
                "mrr": mrr(retrieved, case.relevant),
                "ndcg@10": ndcg_at_k(retrieved, relevance, 10),
            }
        )
    return {
        "quality": aggregate_metrics(metric_rows, ks=(1, 5, 10)) if metric_rows else {},
        "latency_ms": percentiles(timings),
        "queries": rows,
    }


def reproducibility_metadata(*, seed: int = 0, **extra: Any) -> dict[str, Any]:
    packages = {}
    for package in (
        "numpy", "qdrant-client", "sentence-transformers", "torch",
        "transformers", "pdfplumber", "openpyxl",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        **extra,
    }


def make_artifact(
    *,
    benchmark_kind: str,
    dataset: Mapping[str, Any],
    modes: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    seed: int = 0,
    model_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_kind": benchmark_kind,
        "dataset": dict(dataset),
        "configuration": dict(config_snapshot),
        "models": dict(model_identity or {}),
        "environment": reproducibility_metadata(seed=seed),
        "modes": dict(modes),
    }


def write_artifact(artifact: Mapping[str, Any], output_prefix: str | Path) -> dict[str, Path]:
    """Write raw JSON, flat per-query CSV, and generated Markdown report."""

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    flat_rows = []
    for mode, report in (artifact.get("modes") or {}).items():
        for row in report.get("queries", []):
            timing = row.get("timing", {})
            flat_rows.append(
                {
                    "mode": mode,
                    "case_id": row.get("case_id"),
                    "query": row.get("query"),
                    "retrieved": json.dumps(row.get("retrieved", []), ensure_ascii=False),
                    "relevant": json.dumps(row.get("relevant", []), ensure_ascii=False),
                    "recall@1": row.get("recall@1"),
                    "recall@5": row.get("recall@5"),
                    "recall@10": row.get("recall@10"),
                    "mrr": row.get("mrr"),
                    "ndcg@10": row.get("ndcg@10"),
                    **timing,
                }
            )
    fieldnames = list(flat_rows[0]) if flat_rows else ["mode", "case_id", "query"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    markdown_path.write_text(render_markdown(artifact), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def render_markdown(artifact: Mapping[str, Any]) -> str:
    dataset = artifact.get("dataset", {})
    environment = artifact.get("environment", {})
    lines = [
        "# Retrieval benchmark report",
        "",
        "> Generated from the adjacent machine-readable artifact. Values are not hand-authored.",
        "",
        f"- Kind: `{artifact.get('benchmark_kind', 'unknown')}`",
        f"- Dataset: `{dataset.get('name', 'unknown')}` ({dataset.get('cases', 'unknown')} cases)",
        f"- Seed: `{environment.get('seed', 'unknown')}`",
        f"- Timestamp (UTC): `{environment.get('timestamp_utc', 'unknown')}`",
        "",
    ]
    if artifact.get("benchmark_kind") == "model_selection":
        lines.extend(
            [
                "| Candidate | Component | Recall@1 | Recall@5 | MRR | nDCG@10 | Mean latency ms | Dimension | Max length | Device |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for name, report in (artifact.get("modes") or {}).items():
            quality = report.get("quality", {})
            resolved = report.get("resolved_config", {})
            component = report.get("component", "unknown")
            latency = report.get("latency_ms", {})
            mean_latency = (
                latency.get("query_embedding_mean")
                if component == "embedding"
                else latency.get("mean", 0.0)
            )
            lines.append(
                "| {name} | {component} | {r1:.4f} | {r5:.4f} | {mrr:.4f} | {ndcg:.4f} | {latency:.3f} | {dimension} | {max_length} | {device} |".format(
                    name=name,
                    component=component,
                    r1=float(quality.get("recall@1", 0.0)),
                    r5=float(quality.get("recall@5", 0.0)),
                    mrr=float(quality.get("mrr", 0.0)),
                    ndcg=float(quality.get("ndcg@10", 0.0)),
                    latency=float(mean_latency or 0.0),
                    dimension=report.get("dimension", resolved.get("dimension", "n/a")),
                    max_length=resolved.get("max_length", "n/a"),
                    device=resolved.get("device", "n/a"),
                )
            )
        lines.extend(
            [
                "",
                "Model-selection values use fixed document/query or candidate pools; they are not end-to-end service timings.",
                "See the JSON and CSV files for raw rankings and per-query evidence.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 total ms | p95 total ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, report in (artifact.get("modes") or {}).items():
        quality = report.get("quality", {})
        total = report.get("latency_ms", {}).get("total_ms", {})
        lines.append(
            "| {mode} | {r1:.4f} | {r5:.4f} | {r10:.4f} | {mrr:.4f} | {ndcg:.4f} | {p50:.3f} | {p95:.3f} |".format(
                mode=mode,
                r1=float(quality.get("recall@1", 0.0)),
                r5=float(quality.get("recall@5", 0.0)),
                r10=float(quality.get("recall@10", 0.0)),
                mrr=float(quality.get("mrr", 0.0)),
                ndcg=float(quality.get("ndcg@10", 0.0)),
                p50=float(total.get("p50", 0.0)),
                p95=float(total.get("p95", 0.0)),
            )
        )
    lines.extend(["", "See the JSON and CSV files for raw rankings and per-query timings.", ""])
    return "\n".join(lines)


# Compatibility helpers retained for existing benchmark callers.
def run_queries(retriever: Any, queries: Sequence[str], top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
    rows = []
    for query in queries:
        started = time.perf_counter()
        result = retriever.retrieve(query, top_k=top_k, **kwargs)
        rows.append({"result": result, "timing": _timing_from_result(result, (time.perf_counter() - started) * 1000.0)})
    return rows


def timing_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        name: percentiles(
            getattr(row["timing"], name)
            for row in rows
            if getattr(row["timing"], name) is not None
        )
        for name in Timing.__dataclass_fields__
    }


__all__ = [
    "BenchmarkCase", "Timing", "benchmark_modes", "evaluate_embedding_model",
    "evaluate_reranker_model", "evaluate_retrieval_mode",
    "make_artifact", "percentiles", "render_markdown", "reproducibility_metadata",
    "run_queries", "timing_summary", "write_artifact",
]
