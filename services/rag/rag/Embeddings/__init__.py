"""Embedding model capabilities and canonical engine."""
from .adapters import (
    BGEEmbeddingAdapter,
    E5EmbeddingAdapter,
    EmbeddingModel,
    EngineEmbeddingAdapter,
    QwenEmbeddingAdapter,
)
from .capabilities import (
    CapabilityResolutionError,
    EmbeddingProfile,
    EmbeddingProfileOverrides,
    resolve_embedding_profile,
)

__all__ = [
    "EmbeddingModel",
    "EngineEmbeddingAdapter",
    "E5EmbeddingAdapter",
    "BGEEmbeddingAdapter",
    "QwenEmbeddingAdapter",
    "CapabilityResolutionError",
    "EmbeddingProfile",
    "EmbeddingProfileOverrides",
    "resolve_embedding_profile",
]
