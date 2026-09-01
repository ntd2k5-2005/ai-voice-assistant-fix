"""Dependency-free bounds and redaction for opt-in public diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, time
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any


_VECTOR_KEYS = frozenset(
    {
        "vector",
        "vectors",
        "embedding_vector",
        "embedding_vectors",
        "tensor",
        "tensors",
    }
)


def bounded_json(
    value: Any,
    *,
    max_items: int,
    max_string_characters: int,
    max_depth: int = 16,
    preserve_root_mapping: bool = False,
) -> tuple[Any, dict[str, int]]:
    """Return bounded JSON data plus truthful truncation/redaction counters."""

    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or max_items <= 0
        or isinstance(max_string_characters, bool)
        or not isinstance(max_string_characters, int)
        or max_string_characters <= 0
        or isinstance(max_depth, bool)
        or not isinstance(max_depth, int)
        or max_depth <= 0
    ):
        raise ValueError("diagnostic bounds must be positive integers")
    counters = {
        "strings_truncated": 0,
        "items_truncated": 0,
        "values_redacted": 0,
        "depth_truncated": 0,
        "cycles_redacted": 0,
        "non_finite_replaced": 0,
    }
    active: set[int] = set()

    def convert(item: Any, *, depth: int, key: str | None = None) -> Any:
        normalized_key = str(key or "").casefold()
        if normalized_key in _VECTOR_KEYS or (
            normalized_key.endswith("_vector")
            and normalized_key not in {"vector_preview", "query_vector_preview"}
        ):
            counters["values_redacted"] += 1
            return {"redacted": True, "reason": "vector payload omitted from diagnostics"}
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            if math.isfinite(item):
                return item
            counters["non_finite_replaced"] += 1
            return None
        if isinstance(item, str):
            if len(item) > max_string_characters:
                counters["strings_truncated"] += 1
                return item[:max_string_characters]
            return item
        if isinstance(item, bytes):
            return {
                "type": "bytes",
                "length": len(item),
                "sha256": hashlib.sha256(item).hexdigest(),
            }
        if isinstance(item, Path):
            return convert(str(item), depth=depth, key=key)
        if isinstance(item, Enum):
            return convert(item.value, depth=depth, key=key)
        if isinstance(item, (datetime, date, time)):
            return item.isoformat()
        if depth >= max_depth:
            counters["depth_truncated"] += 1
            return {"truncated": True, "reason": "maximum diagnostic depth reached"}
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                counters["cycles_redacted"] += 1
                return {"redacted": True, "reason": "cyclic diagnostic value"}
            active.add(identity)
            try:
                pairs = list(item.items())
                mapping_limit = (
                    len(pairs) if depth == 0 and preserve_root_mapping else max_items
                )
                if len(pairs) > mapping_limit:
                    counters["items_truncated"] += len(pairs) - mapping_limit
                return {
                    str(raw_key): convert(
                        child,
                        depth=depth + 1,
                        key=str(raw_key),
                    )
                    for raw_key, child in pairs[:mapping_limit]
                }
            finally:
                active.remove(identity)
        if isinstance(item, (list, tuple, set, frozenset)):
            identity = id(item)
            if identity in active:
                counters["cycles_redacted"] += 1
                return {"redacted": True, "reason": "cyclic diagnostic value"}
            active.add(identity)
            try:
                values = list(item)
                if len(values) > max_items:
                    counters["items_truncated"] += len(values) - max_items
                return [
                    convert(child, depth=depth + 1)
                    for child in values[:max_items]
                ]
            finally:
                active.remove(identity)
        scalar = getattr(item, "item", None)
        if callable(scalar):
            try:
                return convert(scalar(), depth=depth, key=key)
            except (TypeError, ValueError):
                pass
        return convert(str(item), depth=depth, key=key)

    return convert(value, depth=0), counters


__all__ = ["bounded_json"]
