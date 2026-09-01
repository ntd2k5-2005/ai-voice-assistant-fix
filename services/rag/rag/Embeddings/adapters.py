"""Dependency-light canonical embedding interfaces and compatibility adapters."""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from .capabilities import EmbeddingProfile, resolve_embedding_profile


@runtime_checkable
class EmbeddingModel(Protocol):
    model_name: str
    embedding_dim: int

    def embed_documents(self, texts: Sequence[str], **kwargs: Any): ...
    def embed_query(self, text: str, **kwargs: Any): ...
    def embed_queries_batch(self, queries: Sequence[str], **kwargs: Any): ...


class EngineEmbeddingAdapter:
    """Expose one stable interface without owning or reloading the model."""

    def __init__(self, engine: Any, profile: EmbeddingProfile | None = None):
        if not hasattr(engine, "embed_documents") or not hasattr(engine, "embed_query"):
            raise TypeError("embedding engine must implement embed_documents() and embed_query()")
        self.engine = engine
        self.model_name = str(getattr(engine, "model_name", ""))
        self.embedding_dim = getattr(engine, "embedding_dim", None)
        self.profile = profile or getattr(engine, "profile", None) or resolve_embedding_profile(
            self.model_name or "generic"
        )

    @property
    def identity(self):
        return getattr(self.engine, "identity", None)

    @property
    def resolved_config(self):
        return getattr(self.engine, "resolved_config", {})

    def embed_documents(self, texts: Sequence[str], **kwargs: Any):
        return self.engine.embed_documents(texts, **kwargs)

    def embed_query(self, text: str, **kwargs: Any):
        return self.engine.embed_query(text, **kwargs)

    def embed_queries_batch(self, queries: Sequence[str], **kwargs: Any):
        method = getattr(self.engine, "embed_queries_batch", None)
        if method is not None:
            return method(queries, **kwargs)
        return [self.engine.embed_query(query, **kwargs) for query in queries]

    def close(self) -> None:
        close = getattr(self.engine, "close", None)
        if close is not None:
            close()


class E5EmbeddingAdapter(EngineEmbeddingAdapter):
    """Compatibility marker for a resolved E5 profile."""


class BGEEmbeddingAdapter(EngineEmbeddingAdapter):
    """Compatibility marker for a resolved BGE profile."""


class QwenEmbeddingAdapter(EngineEmbeddingAdapter):
    """Compatibility marker for a resolved Qwen embedding profile."""


__all__ = [
    "EmbeddingModel",
    "EngineEmbeddingAdapter",
    "E5EmbeddingAdapter",
    "BGEEmbeddingAdapter",
    "QwenEmbeddingAdapter",
]
