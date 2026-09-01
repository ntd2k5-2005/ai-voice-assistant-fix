"""Deterministic dense/sparse fusion strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

from ..contracts import RetrievalCandidate
from .scoring import clamp01


def minmax_normalize(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize finite values while preserving deterministic equal-score ties."""

    if not values:
        return {}
    converted = {str(key): float(value) for key, value in values.items()}
    low, high = min(converted.values()), max(converted.values())
    if high == low:
        return {key: 1.0 for key in converted}
    scale = high - low
    return {key: (value - low) / scale for key, value in converted.items()}


class FusionStrategy(Protocol):
    name: str

    def fuse(
        self,
        dense: Sequence[RetrievalCandidate],
        sparse: Sequence[RetrievalCandidate],
        *,
        limit: int | None = None,
    ) -> list[RetrievalCandidate]: ...


def _copy_candidate(candidate: RetrievalCandidate) -> RetrievalCandidate:
    # Use the shared public type while avoiding mutation of backend results.
    return RetrievalCandidate(**candidate.__dict__)


def _candidate_score(candidate: RetrievalCandidate, branch: str) -> float:
    value = getattr(candidate, f"{branch}_score", None)
    return float(candidate.retrieval_score if value is None else value)


def _active_weights(
    dense_weight: float,
    sparse_weight: float,
    dense: Sequence[RetrievalCandidate],
    sparse: Sequence[RetrievalCandidate],
) -> tuple[float, float]:
    dense_active = dense_weight if dense else 0.0
    sparse_active = sparse_weight if sparse else 0.0
    total = dense_active + sparse_active
    if total <= 0:
        return 0.0, 0.0
    return dense_active / total, sparse_active / total


def _merge_candidates(
    dense: Sequence[RetrievalCandidate], sparse: Sequence[RetrievalCandidate]
) -> dict[str, RetrievalCandidate]:
    merged: dict[str, RetrievalCandidate] = {}
    for branch, candidates in (("dense", dense), ("sparse", sparse)):
        for candidate in candidates:
            chunk_id = str(candidate.chunk_id)
            if chunk_id not in merged:
                merged[chunk_id] = _copy_candidate(candidate)
            target = merged[chunk_id]
            if target.document_id and candidate.document_id and target.document_id != candidate.document_id:
                raise ValueError(
                    f"chunk_id {chunk_id!r} maps to conflicting document IDs during fusion"
                )
            if not target.document_id:
                target.document_id = candidate.document_id
            if not target.text:
                target.text = candidate.text
            target.metadata = {**dict(candidate.metadata), **dict(target.metadata)}
            score = _candidate_score(candidate, branch)
            setattr(target, f"{branch}_score", score)
            raw_value = getattr(candidate, f"{branch}_raw_score", None)
            if raw_value is not None:
                setattr(target, f"{branch}_raw_score", float(raw_value))
    return merged


@dataclass(frozen=True)
class WeightedScoreFusion:
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    name: str = "weighted"

    def __post_init__(self) -> None:
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("fusion weights cannot be negative")
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")

    def fuse(self, dense, sparse, *, limit: int | None = None):
        dense = list(dense)
        sparse = list(sparse)
        merged = _merge_candidates(dense, sparse)
        # Branch candidates already carry public relevance scores in [0, 1].
        # Weighted fusion therefore combines those directly instead of
        # re-normalizing them a second time.
        dense_normalized = {
            candidate.chunk_id: clamp01(_candidate_score(candidate, "dense"))
            for candidate in dense
        }
        sparse_normalized = {
            candidate.chunk_id: clamp01(_candidate_score(candidate, "sparse"))
            for candidate in sparse
        }
        dense_weight, sparse_weight = _active_weights(
            self.dense_weight, self.sparse_weight, dense, sparse
        )
        for chunk_id, candidate in merged.items():
            score = (
                dense_weight * dense_normalized.get(chunk_id, 0.0)
                + sparse_weight * sparse_normalized.get(chunk_id, 0.0)
            )
            score = clamp01(score)
            candidate.fusion_raw_score = float(score)
            candidate.fusion_score = float(score)
            candidate.final_score = float(score)
            candidate.final_score_source = "fusion"
            candidate.retrieval_raw_score = float(score)
            candidate.retrieval_score = float(score)
            candidate.retrieval_source = "hybrid"
        ranked = sorted(merged.values(), key=lambda item: (-float(item.fusion_score), item.chunk_id))
        return ranked if limit is None else ranked[:limit]


@dataclass(frozen=True)
class ReciprocalRankFusion:
    """Rank-only fusion, robust to incomparable backend score scales."""

    rank_constant: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    name: str = "rrf"

    def __post_init__(self) -> None:
        if self.rank_constant <= 0:
            raise ValueError("rank_constant must be positive")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("fusion weights cannot be negative")
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")

    def fuse(self, dense, sparse, *, limit: int | None = None):
        dense = sorted(
            list(dense), key=lambda item: (-_candidate_score(item, "dense"), item.chunk_id)
        )
        sparse = sorted(
            list(sparse), key=lambda item: (-_candidate_score(item, "sparse"), item.chunk_id)
        )
        merged = _merge_candidates(dense, sparse)
        dense_weight, sparse_weight = _active_weights(
            self.dense_weight, self.sparse_weight, dense, sparse
        )
        dense_ranks = {candidate.chunk_id: rank for rank, candidate in enumerate(dense, 1)}
        sparse_ranks = {candidate.chunk_id: rank for rank, candidate in enumerate(sparse, 1)}
        for chunk_id, candidate in merged.items():
            score = 0.0
            if chunk_id in dense_ranks:
                score += dense_weight / (self.rank_constant + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += sparse_weight / (self.rank_constant + sparse_ranks[chunk_id])
            raw_score = float(score)
            # With active weights normalized to sum to one, the theoretical
            # maximum is 1/(k+1) when a chunk ranks first in every active
            # branch. Scale by (k+1) for a stable public [0, 1] relevance score.
            public_score = clamp01(raw_score * (self.rank_constant + 1))
            candidate.fusion_raw_score = raw_score
            candidate.fusion_score = public_score
            candidate.final_score = public_score
            candidate.final_score_source = "fusion"
            candidate.retrieval_raw_score = raw_score
            candidate.retrieval_score = public_score
            candidate.retrieval_source = "hybrid"
        ranked = sorted(merged.values(), key=lambda item: (-float(item.fusion_score), item.chunk_id))
        return ranked if limit is None else ranked[:limit]


def resolve_fusion_strategy(
    strategy: str | FusionStrategy,
    *,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    rrf_k: int = 60,
) -> FusionStrategy:
    if not isinstance(strategy, str):
        if not hasattr(strategy, "fuse"):
            raise TypeError("fusion strategy must be a name or implement fuse()")
        return strategy
    normalized = strategy.casefold().replace("-", "_")
    if normalized in {"weighted", "weighted_sum", "minmax", "normalized"}:
        return WeightedScoreFusion(dense_weight=dense_weight, sparse_weight=sparse_weight)
    if normalized in {"rrf", "reciprocal_rank", "reciprocal_rank_fusion"}:
        return ReciprocalRankFusion(
            rank_constant=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
    raise ValueError(f"unknown fusion strategy: {strategy!r}")


__all__ = [
    "FusionStrategy",
    "WeightedScoreFusion",
    "ReciprocalRankFusion",
    "resolve_fusion_strategy",
    "minmax_normalize",
]
