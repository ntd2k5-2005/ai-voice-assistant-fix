"""Score normalization helpers for public retrieval contracts.

Backend/model scores live on different mathematical scales.  The public API
therefore exposes monotonic relevance scores in ``[0, 1]`` while diagnostics
retain the raw values used by the corresponding backend/model.

The normalized scores are *not probabilities* unless the underlying model
explicitly emits probabilities.  They are ranking-friendly relevance scores.
"""
from __future__ import annotations

import math
from typing import Iterable


def clamp01(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return min(1.0, max(0.0, value))


def cosine_to_unit(value: float) -> float:
    """Map cosine similarity ``[-1, 1]`` monotonically to ``[0, 1]``."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("cosine score must be finite")
    # Some vector engines can exceed the theoretical interval by tiny floating
    # point error; clamp before mapping rather than exposing >1 public scores.
    value = min(1.0, max(-1.0, value))
    return (value + 1.0) / 2.0


def minmax_unit(values: Iterable[float]) -> list[float]:
    """Query-local monotonic min-max normalization.

    BM25/FTS scores do not have a corpus-independent bounded scale, so sparse
    public scores are intentionally relative to the candidate pool for the
    current query. Equal-score pools map to 1.0 to preserve the fact that every
    returned candidate ties for the best observed lexical score.
    """

    converted = [float(value) for value in values]
    if not converted:
        return []
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("score pool contains non-finite values")
    low, high = min(converted), max(converted)
    if high == low:
        return [1.0 for _ in converted]
    scale = high - low
    return [(value - low) / scale for value in converted]


def sigmoid_unit(value: float) -> float:
    """Stable logistic mapping used for raw scalar/logit reranker scores."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("reranker score must be finite")
    if value >= 0:
        z = math.exp(-min(value, 80.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -80.0))
    return z / (1.0 + z)


def normalize_reranker_score(value: float, extraction: str | None) -> float:
    """Normalize a reranker output without changing its order.

    ``positive_probability``, ``yes_no_probability`` and ``sigmoid`` profiles
    already emit bounded scores inside the model adapter. Scalar/logit profiles
    (including ``Alibaba-NLP/gte-multilingual-reranker-base``) are mapped with
    sigmoid. ``auto`` is accepted defensively: values already in [0, 1] are
    preserved, otherwise sigmoid is used.
    """

    value = float(value)
    normalized = (extraction or "auto").casefold()
    if normalized in {"scalar", "positive_logit"}:
        return sigmoid_unit(value)
    if normalized in {"sigmoid", "positive_probability", "yes_no_probability"}:
        return clamp01(value)
    if 0.0 <= value <= 1.0:
        return value
    return sigmoid_unit(value)


__all__ = [
    "clamp01",
    "cosine_to_unit",
    "minmax_unit",
    "normalize_reranker_score",
    "sigmoid_unit",
]
