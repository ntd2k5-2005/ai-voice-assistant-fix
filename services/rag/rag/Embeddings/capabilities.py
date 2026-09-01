"""Embedding model capability resolution.

The resolver is deliberately the only place where model identifiers are
interpreted. Encoding code consumes a resolved profile and therefore does not
need family-specific conditionals.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..errors import ModelCapabilityError


EmbeddingInputKind = Literal["query", "document"]


class CapabilityResolutionError(ModelCapabilityError, ValueError):
    """Raised when a model capability cannot be resolved safely."""


@dataclass(frozen=True)
class EmbeddingProfile:
    family: str
    query_template: str = "{text}"
    document_template: str = "{text}"
    query_instruction: str = ""
    document_instruction: str = ""
    normalize: bool = True
    requires_remote_code: bool = False

    def __post_init__(self) -> None:
        for name, template in (
            ("query_template", self.query_template),
            ("document_template", self.document_template),
        ):
            if "{text}" not in template:
                raise CapabilityResolutionError(f"{name} must contain {{text}}")

    def format(self, text: str, kind: EmbeddingInputKind) -> str:
        if kind not in {"query", "document"}:
            raise ValueError("embedding input kind must be 'query' or 'document'")
        template = self.query_template if kind == "query" else self.document_template
        instruction = self.query_instruction if kind == "query" else self.document_instruction
        return template.format(text=text, instruction=instruction)


@dataclass(frozen=True)
class EmbeddingProfileOverrides:
    """Explicit escape hatch for an unknown or newly released model."""

    family: str | None = None
    query_template: str | None = None
    document_template: str | None = None
    query_instruction: str | None = None
    document_instruction: str | None = None
    normalize: bool | None = None
    requires_remote_code: bool | None = None


_GENERIC = EmbeddingProfile(family="sentence-transformer")
_E5 = EmbeddingProfile(
    family="e5",
    query_template="query: {text}",
    document_template="passage: {text}",
)
_E5_INSTRUCT = EmbeddingProfile(
    family="e5-instruct",
    query_template="Instruct: {instruction}\nQuery: {text}",
    query_instruction="Given a query, retrieve relevant passages",
)
_BGE = EmbeddingProfile(
    family="bge",
    query_template="Represent this sentence for searching relevant passages: {text}",
)
_BGE_M3 = EmbeddingProfile(family="bge-m3")
_QWEN = EmbeddingProfile(
    family="qwen-embedding",
    query_template="Instruct: {instruction}\nQuery:{text}",
    query_instruction="Given a web search query, retrieve relevant passages that answer the query",
)
_NOMIC = EmbeddingProfile(
    family="nomic",
    query_template="search_query: {text}",
    document_template="search_document: {text}",
    requires_remote_code=True,
)


def _base_profile(model_name: str) -> EmbeddingProfile:
    name = model_name.casefold()
    # Ordered from the most specific family to the most general one.
    if "nomic-embed" in name:
        return _NOMIC
    if "qwen" in name and ("embed" in name or "embedding" in name):
        return _QWEN
    if "bge-m3" in name:
        return _BGE_M3
    if "bge" in name and "rerank" not in name:
        return _BGE
    if "e5" in name and "instruct" in name:
        return _E5_INSTRUCT
    if "e5" in name:
        return _E5
    return _GENERIC


def resolve_embedding_profile(
    model_name: str,
    overrides: EmbeddingProfileOverrides | None = None,
) -> EmbeddingProfile:
    if not isinstance(model_name, str) or not model_name.strip():
        raise CapabilityResolutionError("embedding model_name must be a non-empty string")
    profile = _base_profile(model_name.strip())
    if overrides is None:
        return profile
    changes = {
        field: value
        for field, value in vars(overrides).items()
        if value is not None
    }
    return replace(profile, **changes)


def valid_model_max_length(value: object) -> int | None:
    """Reject Hugging Face's huge 'unknown length' sentinel values."""

    if isinstance(value, bool):
        return None
    try:
        resolved = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if 1 < resolved <= 1_000_000 else None


__all__ = [
    "CapabilityResolutionError",
    "EmbeddingInputKind",
    "EmbeddingProfile",
    "EmbeddingProfileOverrides",
    "resolve_embedding_profile",
    "valid_model_max_length",
]
