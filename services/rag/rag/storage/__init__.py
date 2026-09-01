"""Storage APIs exposed without importing optional backends eagerly."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


_EXPORTS = {
    "QdrantStore": (".qdrant", "QdrantStore"),
    "SQLiteBM25Retriever": (".fts5", "SQLiteBM25Retriever"),
    "ChunkIdentityError": (".identity", "ChunkIdentityError"),
    "ChunkManifestEntry": (".identity", "ChunkManifestEntry"),
}


if TYPE_CHECKING:
    from .fts5 import SQLiteBM25Retriever
    from .identity import ChunkIdentityError, ChunkManifestEntry
    from .qdrant import QdrantStore


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
