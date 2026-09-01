"""Canonical dense/sparse/hybrid retrieval with optional independent reranking."""
from __future__ import annotations

import math
import inspect
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import fields
from enum import Enum
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence

from ..contracts import RetrievalCandidate, RetrievalResult
from ..errors import RerankerTimeoutError, RetrievalError
from ..observability import bounded_json

from .fusion import FusionStrategy, resolve_fusion_strategy
from .scoring import cosine_to_unit, minmax_unit, normalize_reranker_score


def _copy_candidate(candidate: RetrievalCandidate) -> RetrievalCandidate:
    values = {field.name: getattr(candidate, field.name) for field in fields(RetrievalCandidate)}
    return RetrievalCandidate(**values)


def _coerce_candidates(values: Sequence[Any], source: str) -> list[RetrievalCandidate]:
    """Coerce backend hits and expose bounded public branch scores.

    Dense backends return cosine similarity while sparse FTS/BM25 values have
    query/corpus-dependent magnitude. Raw values are retained for diagnostics;
    public branch/retrieval scores are monotonic relevance scores in [0, 1].
    """

    result: list[RetrievalCandidate] = []
    seen = set()
    raw_scores: list[float] = []
    for value in values:
        candidate = (
            _copy_candidate(value)
            if isinstance(value, RetrievalCandidate)
            else RetrievalCandidate.from_mapping(value, source)
        )
        if candidate.chunk_id in seen:
            raise ValueError(f"{source} retriever returned duplicate chunk_id {candidate.chunk_id!r}")
        seen.add(candidate.chunk_id)
        candidate.retrieval_source = source
        raw_score = float(candidate.retrieval_score)
        if not math.isfinite(raw_score):
            raise RuntimeError(f"{source} retriever returned a non-finite score")
        raw_scores.append(raw_score)
        candidate.retrieval_raw_score = raw_score
        if source == "dense":
            candidate.dense_raw_score = raw_score
        else:
            candidate.sparse_raw_score = raw_score
        result.append(candidate)

    if source == "dense":
        public_scores = [cosine_to_unit(value) for value in raw_scores]
    else:
        public_scores = minmax_unit(raw_scores)

    score_field = "dense_score" if source == "dense" else "sparse_score"
    for candidate, public_score in zip(result, public_scores, strict=True):
        setattr(candidate, score_field, float(public_score))
        candidate.retrieval_score = float(public_score)

    return sorted(
        result,
        key=lambda item: (-float(getattr(item, score_field)), item.chunk_id),
    )


def _json_safe(value: Any) -> Any:
    """Convert trace values without importing presentation dependencies."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_safe(scalar())
        except (TypeError, ValueError):
            pass
    return str(value)


def _resolved_config(component: Any) -> dict[str, Any]:
    value = getattr(component, "resolved_config", {}) if component is not None else {}
    if callable(value):
        value = value()
    return _json_safe(value) if isinstance(value, Mapping) else {}


class RetrievalEngine:
    """Execute a retrieval mode and optionally rerank its candidate pool.

    The old :class:`HybridRetriever` name remains an alias-compatible subclass,
    while ``mode`` makes dense, sparse, and hybrid semantics explicit.
    """

    VALID_MODES = frozenset({"dense", "sparse", "hybrid"})

    def __init__(
        self,
        dense=None,
        sparse=None,
        reranker=None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rerank_timeout_ms: int | None = None,
        *,
        fusion_strategy: str | FusionStrategy = "rrf",
        rrf_k: int = 60,
        rerank_failure: str = "fallback",
        parallel_branches: bool = False,
        max_query_chars: int = 32_768,
        allow_sparse_rerank: bool = True,
        visibility_registry=None,
        visibility_retries: int = 2,
        trace_max_items: int = 2_000,
        trace_max_nested_items: int = 100,
        trace_text_characters: int = 4_000,
        trace_vector_dimensions: int = 8,
    ) -> None:
        if rerank_timeout_ms is not None and rerank_timeout_ms <= 0:
            raise ValueError("rerank_timeout_ms must be positive")
        if rerank_failure not in {"fallback", "raise"}:
            raise ValueError("rerank_failure must be 'fallback' or 'raise'")
        if max_query_chars <= 0:
            raise ValueError("max_query_chars must be positive")
        if isinstance(visibility_retries, bool) or visibility_retries < 0:
            raise ValueError("visibility_retries cannot be negative")
        for name, value in (
            ("trace_max_items", trace_max_items),
            ("trace_max_nested_items", trace_max_nested_items),
            ("trace_text_characters", trace_text_characters),
            ("trace_vector_dimensions", trace_vector_dimensions),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.fusion = resolve_fusion_strategy(
            fusion_strategy,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k,
        )
        self.rerank_timeout_ms = rerank_timeout_ms
        self.rerank_failure = rerank_failure
        self.parallel_branches = bool(parallel_branches)
        self.max_query_chars = int(max_query_chars)
        self.allow_sparse_rerank = bool(allow_sparse_rerank)
        self.visibility_registry = visibility_registry
        self.visibility_retries = int(visibility_retries)
        self.trace_max_items = int(trace_max_items)
        self.trace_max_nested_items = int(trace_max_nested_items)
        self.trace_text_characters = int(trace_text_characters)
        self.trace_vector_dimensions = int(trace_vector_dimensions)

    def _candidate_trace(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        rank_field: str,
    ) -> list[dict[str, Any]]:
        result = []
        for rank, candidate in enumerate(candidates[: self.trace_max_items], 1):
            row = candidate.to_dict(include_raw_scores=True)
            text = str(row.get("text") or "")
            row["text"] = text[: self.trace_text_characters]
            row["text_truncated"] = len(text) > self.trace_text_characters
            row[rank_field] = rank
            safe_metadata, counters = bounded_json(
                row.get("metadata") or {},
                max_items=self.trace_max_nested_items,
                max_string_characters=self.trace_text_characters,
            )
            row["metadata"] = safe_metadata
            for key in ("source", "section", "sheet"):
                value = row.get(key)
                if isinstance(value, str) and len(value) > self.trace_text_characters:
                    row[key] = value[: self.trace_text_characters]
                    counters["strings_truncated"] += 1
            nonzero = {key: value for key, value in counters.items() if value}
            if nonzero:
                row["diagnostic_bounds"] = nonzero
            result.append(_json_safe(row))
        return result

    def _query_embedding_trace(self, mode: str, elapsed_ms: float) -> dict[str, Any]:
        if mode == "sparse":
            return {
                "executed": False,
                "reason": "Sparse retrieval does not execute query embedding.",
            }
        vector = getattr(self.dense, "last_query_vector", None)
        resolved = _resolved_config(getattr(self.dense, "embedder", None))
        if vector is None:
            return {
                "executed": False,
                "reason": "Dense retriever did not expose a query-vector summary.",
                "elapsed_ms": float(elapsed_ms),
            }
        values = list(vector)
        norm = math.sqrt(sum(float(value) ** 2 for value in values))
        return {
            "executed": True,
            "model": resolved.get("model_name")
            or getattr(getattr(self.dense, "embedder", None), "model_name", None),
            "dimension": len(values),
            "elapsed_ms": float(elapsed_ms),
            "vector_norm": float(norm),
            "finite": all(math.isfinite(float(value)) for value in values),
            "vector_preview": [
                float(value) for value in values[: self.trace_vector_dimensions]
            ],
            "preview_dimensions": min(len(values), self.trace_vector_dimensions),
            "vector_omitted": len(values) > self.trace_vector_dimensions,
        }

    def _resolve_mode(self, mode: str | None) -> str:
        if mode is None:
            if self.dense is not None and self.sparse is not None:
                return "hybrid"
            if self.dense is not None:
                return "dense"
            if self.sparse is not None:
                return "sparse"
            raise ValueError("at least one retrieval branch is required")
        normalized = mode.casefold()
        if normalized == "bm25":
            normalized = "sparse"
        if normalized not in self.VALID_MODES:
            raise ValueError("mode must be dense, sparse, or hybrid")
        if normalized in {"dense", "hybrid"} and self.dense is None:
            raise ValueError(f"mode={normalized!r} requires a dense retriever")
        if normalized in {"sparse", "hybrid"} and self.sparse is None:
            raise ValueError(f"mode={normalized!r} requires a sparse retriever")
        return normalized

    @staticmethod
    def _supports_parameter(callable_value, name: str) -> bool:
        parameters = inspect.signature(callable_value).parameters
        return name in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _search(
        self,
        query: str,
        mode: str,
        candidate_k: int,
        filters,
        *,
        visibility_snapshot=None,
        query_vector=None,
    ):
        dense: list[RetrievalCandidate] = []
        sparse: list[RetrievalCandidate] = []
        sparse_latency_ms = 0.0

        def dense_search():
            kwargs = {}
            if visibility_snapshot is not None:
                if not self._supports_parameter(self.dense.search, "visibility_snapshot"):
                    raise TypeError(
                        "dense retriever cannot enforce a shared lifecycle visibility snapshot"
                    )
                kwargs["visibility_snapshot"] = visibility_snapshot
            if query_vector is not None:
                if not self._supports_parameter(self.dense.search, "query_vector"):
                    raise TypeError("dense retriever cannot reuse a prepared query embedding")
                kwargs["query_vector"] = query_vector
            return self.dense.search(query, candidate_k, filters, **kwargs)

        def sparse_search():
            started = time.perf_counter()
            kwargs = {}
            if visibility_snapshot is not None:
                if not self._supports_parameter(self.sparse.search, "visibility_snapshot"):
                    raise TypeError(
                        "sparse retriever cannot enforce a shared lifecycle visibility snapshot"
                    )
                kwargs["visibility_snapshot"] = visibility_snapshot
            values = self.sparse.search(query, candidate_k, filters, **kwargs)
            return values, (time.perf_counter() - started) * 1000

        if mode == "hybrid" and self.parallel_branches:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-retrieval") as executor:
                dense_future = executor.submit(dense_search)
                sparse_future = executor.submit(sparse_search)
                dense = _coerce_candidates(dense_future.result(), "dense")
                sparse_values, sparse_latency_ms = sparse_future.result()
                sparse = _coerce_candidates(sparse_values, "sparse")
        else:
            if mode in {"dense", "hybrid"}:
                dense = _coerce_candidates(dense_search(), "dense")
            if mode in {"sparse", "hybrid"}:
                sparse_values, sparse_latency_ms = sparse_search()
                sparse = _coerce_candidates(sparse_values, "sparse")
        return dense, sparse, sparse_latency_ms

    def _visibility_snapshot(self):
        resolver = getattr(self.visibility_registry, "visibility_snapshot", None)
        return resolver() if callable(resolver) else None

    def _visibility_snapshot_current(self, snapshot) -> bool:
        if snapshot is None:
            return True
        checker = getattr(
            self.visibility_registry, "is_visibility_snapshot_current", None
        )
        if not callable(checker):
            raise TypeError(
                "visibility registry supplies snapshots but cannot validate their epoch"
            )
        return bool(checker(snapshot))

    def _candidate_pool(
        self,
        mode: str,
        dense: list[RetrievalCandidate],
        sparse: list[RetrievalCandidate],
        candidate_k: int,
    ) -> list[RetrievalCandidate]:
        if mode == "hybrid":
            return self.fusion.fuse(dense, sparse, limit=candidate_k)
        branch = dense if mode == "dense" else sparse
        score_name = "dense_score" if mode == "dense" else "sparse_score"
        result = []
        for original in branch[:candidate_k]:
            candidate = _copy_candidate(original)
            score = float(getattr(candidate, score_name))
            raw_name = "dense_raw_score" if mode == "dense" else "sparse_raw_score"
            raw_score = getattr(candidate, raw_name, None)
            candidate.final_score = score
            candidate.final_score_source = mode
            candidate.retrieval_score = score
            candidate.retrieval_raw_score = float(raw_score) if raw_score is not None else None
            candidate.retrieval_source = mode
            result.append(candidate)
        return result

    def _execute_reranker(self, query: str, pool: list[RetrievalCandidate], top_k: int):
        if hasattr(self.reranker, "rerank_candidates"):
            return self.reranker.rerank_candidates(pool, query, top_k=top_k)
        if hasattr(self.reranker, "rerank"):
            return self.reranker.rerank(query, pool, top_k=top_k)
        raise TypeError("reranker must implement rerank_candidates() or rerank()")

    def _rerank_with_timeout(
        self, query: str, pool: list[RetrievalCandidate], top_k: int
    ) -> Sequence[Any]:
        timeout_ms = self.rerank_timeout_ms
        if timeout_ms is None:
            timeout_ms = getattr(self.reranker, "timeout_ms", None)
        if timeout_ms is None:
            return self._execute_reranker(query, pool, top_k)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-rerank")
        future = executor.submit(self._execute_reranker, query, pool, top_k)
        try:
            return future.result(timeout=timeout_ms / 1000)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RerankerTimeoutError("reranker timeout") from exc
        finally:
            # A running Python/model call cannot be force-killed safely. Do not
            # block the retrieval fallback waiting for it to finish.
            executor.shutdown(wait=False, cancel_futures=True)

    def _coerce_reranked(
        self, values: Sequence[Any], pool: list[RetrievalCandidate]
    ) -> list[RetrievalCandidate]:
        by_id = {candidate.chunk_id: candidate for candidate in pool}
        result = []
        seen = set()
        for value in values:
            if isinstance(value, tuple) and len(value) == 2:
                item, raw_score = value
            else:
                item = value
                raw_score = getattr(item, "rerank_score", None)
            candidate = (
                item
                if isinstance(item, RetrievalCandidate)
                else RetrievalCandidate.from_mapping(item, "rerank")
            )
            if candidate.chunk_id not in by_id:
                raise RuntimeError(
                    f"reranker returned unknown chunk_id {candidate.chunk_id!r}"
                )
            if candidate.chunk_id in seen:
                raise RuntimeError(
                    f"reranker returned duplicate chunk_id {candidate.chunk_id!r}"
                )
            if raw_score is None:
                raise RuntimeError("reranker did not provide an explicit rerank score")
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("reranker returned a non-finite score")
            resolved = _copy_candidate(by_id[candidate.chunk_id])
            extraction = _resolved_config(self.reranker).get("score_extraction")
            public_score = normalize_reranker_score(score, extraction)
            resolved.rerank_raw_score = score
            resolved.rerank_score = public_score
            resolved.final_score = public_score
            resolved.final_score_source = "rerank"
            resolved.retrieval_raw_score = score
            resolved.retrieval_score = public_score
            seen.add(resolved.chunk_id)
            result.append(resolved)
        if pool and not result:
            raise RuntimeError("reranker returned no scored candidates")
        return sorted(result, key=lambda item: (-float(item.rerank_score), item.chunk_id))

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
        filters=None,
        rerank: bool = False,
        *,
        mode: str | None = None,
        include_trace: bool = False,
    ) -> RetrievalResult:
        started = time.perf_counter()
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        if len(query) > self.max_query_chars:
            raise ValueError(
                f"query exceeds configured maximum of {self.max_query_chars} characters"
            )
        if top_k <= 0 or candidate_k < top_k:
            raise ValueError("candidate_k must be >= top_k > 0")
        actual_mode = self._resolve_mode(mode)
        if rerank and actual_mode == "sparse" and not self.allow_sparse_rerank:
            raise ValueError("reranking sparse-only retrieval is disabled by configuration")

        search_started = time.perf_counter()
        query_vector = None
        if actual_mode in {"dense", "hybrid"} and hasattr(self.dense, "embed_query"):
            query_vector = self.dense.embed_query(query)
        visibility_retries_used = 0
        visibility_epoch = None
        for attempt in range(self.visibility_retries + 1):
            visibility_snapshot = self._visibility_snapshot()
            visibility_epoch = getattr(visibility_snapshot, "epoch", None)
            dense, sparse, sparse_latency_ms = self._search(
                query,
                actual_mode,
                candidate_k,
                filters,
                visibility_snapshot=visibility_snapshot,
                query_vector=query_vector,
            )
            if self._visibility_snapshot_current(visibility_snapshot):
                visibility_retries_used = attempt
                break
            if attempt >= self.visibility_retries:
                raise RetrievalError(
                    "knowledge visibility changed repeatedly during retrieval; retry the query"
                )
        search_wall_ms = (time.perf_counter() - search_started) * 1000
        fusion_started = time.perf_counter()
        pool = self._candidate_pool(actual_mode, dense, sparse, candidate_k)
        candidate_selection_ms = (time.perf_counter() - fusion_started) * 1000
        fusion_latency_ms = (
            candidate_selection_ms if actual_mode == "hybrid" else 0.0
        )

        rerank_requested = bool(rerank)
        rerank_applied = False
        rerank_fallback = False
        rerank_error = None
        rerank_latency_ms = 0.0
        ranked = pool
        rerank_input_count = len(pool) if rerank_requested else 0
        if rerank_requested and pool:
            rerank_started = time.perf_counter()
            try:
                if self.reranker is None:
                    raise RuntimeError("reranker unavailable")
                reranked = self._rerank_with_timeout(query, pool, top_k)
                ranked = self._coerce_reranked(reranked, pool)
                rerank_applied = True
            except Exception as exc:
                if self.rerank_failure == "raise":
                    raise
                rerank_fallback = True
                rerank_error = "TimeoutError" if isinstance(exc, TimeoutError) else type(exc).__name__
                ranked = pool
            finally:
                rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000

        ranked = ranked[:top_k]
        for rank, candidate in enumerate(ranked, 1):
            candidate.rank = rank

        dense_embedding_ms = (
            float(getattr(self.dense, "last_embed_latency_ms", 0.0))
            if actual_mode in {"dense", "hybrid"}
            else 0.0
        )
        dense_search_ms = (
            float(getattr(self.dense, "last_search_latency_ms", 0.0))
            if actual_mode in {"dense", "hybrid"}
            else 0.0
        )
        total_ms = (time.perf_counter() - started) * 1000
        diagnostics = {
            "dense_candidates": len(dense),
            "sparse_candidates": len(sparse),
            "candidate_pool": len(pool),
            "fusion_candidates": len(pool) if actual_mode == "hybrid" else 0,
            "candidate_k": candidate_k,
            "top_k": top_k,
            "fusion_strategy": self.fusion.name if actual_mode == "hybrid" else None,
            "rerank_requested": rerank_requested,
            "rerank_available": self.reranker is not None,
            "rerank_candidates": rerank_input_count,
            "rerank_applied": rerank_applied,
            "rerank_fallback": rerank_fallback,
            "rerank_error": rerank_error,
            # Backward-compatible alias; unlike the previous implementation it
            # is true only when a requested rerank actually failed/unavailable.
            "fallback": rerank_fallback,
            "parallel_branches": self.parallel_branches and actual_mode == "hybrid",
            "visibility_epoch": visibility_epoch,
            "visibility_retries": visibility_retries_used,
            "score_semantics": {
                "public_range": [0.0, 1.0],
                "meaning": "monotonic relevance score; not a calibrated probability",
                "dense": "cosine similarity mapped from [-1,1] to [0,1]",
                "sparse": "query-local min-max normalization of FTS5/BM25 score",
                "fusion": (
                    "RRF normalized by its theoretical per-query maximum"
                    if actual_mode == "hybrid" and self.fusion.name == "rrf"
                    else "weighted combination of normalized branch scores"
                    if actual_mode == "hybrid"
                    else None
                ),
                "rerank": "probability-like model outputs preserved; scalar/logit outputs mapped with sigmoid",
                "raw_scores_available_in_trace": bool(include_trace),
            },
            "latency_ms": {
                "query_embedding": dense_embedding_ms,
                "dense_search": dense_search_ms,
                "sparse_search": sparse_latency_ms,
                "fusion": fusion_latency_ms,
                "candidate_selection": candidate_selection_ms,
                "rerank": rerank_latency_ms,
                "total": total_ms,
            },
        }
        if include_trace:
            dense_ranks = {
                candidate.chunk_id: rank for rank, candidate in enumerate(dense, 1)
            }
            sparse_ranks = {
                candidate.chunk_id: rank for rank, candidate in enumerate(sparse, 1)
            }
            fusion_ranks = {
                candidate.chunk_id: rank for rank, candidate in enumerate(pool, 1)
            }

            def enrich_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
                for row in rows:
                    chunk_id = str(row.get("chunk_id") or "")
                    row["dense_rank"] = dense_ranks.get(chunk_id)
                    row["sparse_rank"] = sparse_ranks.get(chunk_id)
                    row["fusion_rank"] = fusion_ranks.get(chunk_id)
                return rows

            fusion_parameters = {
                key: _json_safe(getattr(self.fusion, key))
                for key in ("dense_weight", "sparse_weight", "rank_constant")
                if hasattr(self.fusion, key)
            }
            before_trace = (
                enrich_ranks(
                    self._candidate_trace(pool, rank_field="pre_rerank_rank")
                )
                if rerank_requested
                else []
            )
            after_trace = (
                enrich_ranks(
                    self._candidate_trace(ranked, rank_field="post_rerank_rank")
                )
                if rerank_requested
                else []
            )
            if not rerank_requested:
                rerank_reason = "Reranking was not requested."
            elif not pool:
                rerank_reason = "The candidate pool was empty."
            elif rerank_applied:
                rerank_reason = None
            else:
                rerank_reason = "Reranking failed or was unavailable; retrieval order was retained."
            reranker_resolved = _resolved_config(self.reranker)
            diagnostics["trace"] = {
                "query": {
                    "original": query,
                    "normalized": query,
                    "mode": actual_mode,
                    "embedding_executed": actual_mode in {"dense", "hybrid"},
                    "sparse_executed": actual_mode in {"sparse", "hybrid"},
                    "rerank_requested": rerank_requested,
                },
                "query_embedding": self._query_embedding_trace(
                    actual_mode, dense_embedding_ms
                ),
                "dense_candidates": self._candidate_trace(
                    dense, rank_field="dense_rank"
                ),
                "sparse_candidates": self._candidate_trace(
                    sparse, rank_field="sparse_rank"
                ),
                "fusion": {
                    "executed": actual_mode == "hybrid",
                    "reason": (
                        None
                        if actual_mode == "hybrid"
                        else "Fusion is only executed for hybrid retrieval."
                    ),
                    "strategy": self.fusion.name if actual_mode == "hybrid" else None,
                    "parameters": fusion_parameters,
                    "candidates": (
                        enrich_ranks(
                            self._candidate_trace(pool, rank_field="fusion_rank")
                        )
                        if actual_mode == "hybrid"
                        else []
                    ),
                },
                "rerank": {
                    "executed": rerank_requested and bool(pool),
                    "reason": rerank_reason,
                    "model": reranker_resolved.get("model_name")
                    or getattr(self.reranker, "model_name", None),
                    "elapsed_ms": float(rerank_latency_ms),
                    "applied": rerank_applied,
                    "fallback": rerank_fallback,
                    "error": rerank_error,
                    "before": before_trace,
                    "after": after_trace,
                },
                "latency": {
                    "query_embedding_ms": dense_embedding_ms,
                    "dense_search_ms": dense_search_ms,
                    "sparse_search_ms": sparse_latency_ms,
                    "fusion_ms": fusion_latency_ms,
                    "candidate_selection_ms": candidate_selection_ms,
                    "rerank_ms": rerank_latency_ms,
                    "total_ms": total_ms,
                },
                "truncation": {
                    "max_items": self.trace_max_items,
                    "max_nested_items": self.trace_max_nested_items,
                    "max_text_characters_per_item": self.trace_text_characters,
                    "vector_preview_dimensions": self.trace_vector_dimensions,
                    "dense_candidates_truncated": len(dense) > self.trace_max_items,
                    "sparse_candidates_truncated": len(sparse) > self.trace_max_items,
                    "fusion_candidates_truncated": (
                        actual_mode == "hybrid"
                        and len(pool) > self.trace_max_items
                    ),
                },
            }
        return RetrievalResult(
            candidates=ranked,
            embed_latency_ms=dense_embedding_ms if actual_mode != "sparse" else None,
            search_latency_ms=search_wall_ms,
            rerank_latency_ms=rerank_latency_ms,
            total_latency_ms=total_ms,
            mode=actual_mode,
            diagnostics=diagnostics,
            query=query,
            rerank_requested=rerank_requested,
            dense_search_latency_ms=dense_search_ms,
            sparse_search_latency_ms=sparse_latency_ms,
            fusion_latency_ms=fusion_latency_ms,
        )


class HybridRetriever(RetrievalEngine):
    """Backward-compatible name for :class:`RetrievalEngine`."""


__all__ = ["RetrievalEngine", "HybridRetriever"]
