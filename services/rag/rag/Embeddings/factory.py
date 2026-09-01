"""Embedding adapter factory backed by the centralized capability resolver."""
from __future__ import annotations

from .adapters import (
    BGEEmbeddingAdapter,
    E5EmbeddingAdapter,
    EngineEmbeddingAdapter,
    QwenEmbeddingAdapter,
)
from .capabilities import resolve_embedding_profile


_ADAPTERS = {
    "e5": E5EmbeddingAdapter,
    "e5-instruct": E5EmbeddingAdapter,
    "bge": BGEEmbeddingAdapter,
    "bge-m3": BGEEmbeddingAdapter,
    "qwen-embedding": QwenEmbeddingAdapter,
}


def adapter_for(engine, model_name: str = ""):
    name = model_name or getattr(engine, "model_name", "") or "generic"
    profile = getattr(engine, "profile", None) or resolve_embedding_profile(name)
    return _ADAPTERS.get(profile.family, EngineEmbeddingAdapter)(engine, profile=profile)


__all__ = ["adapter_for"]
