"""Standalone knowledge-ingestion and retrieval package.

The package root intentionally stays lightweight. Model and storage stacks are
loaded only when their public classes are requested or a :class:`RagModule` is
constructed.
"""

from __future__ import annotations

from importlib import import_module

from .config import RagConfig
from .contracts import (
    DocumentSource,
    IngestionResult,
    MetadataFilter,
    ParsedDocument,
    RetrievalCandidate,
    RetrievalResult,
)


_LAZY_IMPORTS = {
    "RagModule": (".service", "RagModule"),
    "KnowledgeService": (".service", "KnowledgeService"),
    "RetrievalPipeline": (".pipeline", "RetrievalPipeline"),
    "create_canonical_pipeline": (".pipeline", "create_canonical_pipeline"),
    "DenseRetriever": (".retrieval", "DenseRetriever"),
    "SparseRetriever": (".retrieval", "SparseRetriever"),
    "SQLiteBM25Retriever": (".retrieval", "SQLiteBM25Retriever"),
    "HybridRetriever": (".retrieval", "HybridRetriever"),
    "RetrievalEngine": (".retrieval", "RetrievalEngine"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


__all__ = [
    "DocumentSource", "HybridRetriever", "IngestionResult", "KnowledgeService",
    "MetadataFilter", "ParsedDocument", "RagConfig", "RagModule",
    "RetrievalCandidate", "RetrievalEngine", "RetrievalPipeline",
    "RetrievalResult", "DenseRetriever", "SparseRetriever",
    "SQLiteBM25Retriever", "create_canonical_pipeline",
]
