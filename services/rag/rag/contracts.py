"""Dependency-free contracts shared by ingestion, storage, and retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    page: Optional[int] = None
    heading: Optional[str] = None
    url: Optional[str] = None
    parser_method: str = "unknown"
    ocr_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentSource:
    """A local file, in-memory text, or byte payload accepted by ingestion."""

    value: str | bytes | Path
    source_type: str = "file"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mime_type: str | None = None
    file_name: str | None = None

    def __post_init__(self) -> None:
        if self.source_type not in {"file", "text", "bytes"}:
            raise ValueError("source_type must be 'file', 'text', or 'bytes'")
        if self.source_type == "bytes" and not isinstance(self.value, bytes):
            raise TypeError("a bytes source requires a bytes value")
        if self.source_type != "bytes" and not isinstance(self.value, (str, Path)):
            raise TypeError(f"a {self.source_type} source requires str or Path")


@dataclass(frozen=True)
class ParsedDocument:
    """One ordered, provenance-bearing unit emitted by a parser."""

    document_id: str
    source: str
    source_type: str
    content: str
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    title: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unit_type: str = "text"
    unit_index: int = 0


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    document_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    document_version: int = 1
    source: str = ""

    def to_mapping(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.setdefault("chunk_id", self.chunk_id)
        metadata.setdefault("document_id", self.document_id)
        metadata.setdefault("version", self.document_version)
        if self.source:
            metadata.setdefault("source", self.source)
        return {
            "_id": self.chunk_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_id": self.document_id,
            "metadata": metadata,
        }


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: Any
    dimension: int
    model_name: str
    normalized: bool
    revision: str | None = None


@dataclass(frozen=True)
class MetadataFilter:
    """Backend-neutral, fail-closed metadata predicate."""

    field: str
    operator: str
    value: Any

    def __post_init__(self) -> None:
        if not self.field or self.operator not in {"=", "!=", ">", ">=", "<", "<=", "in", "contains"}:
            raise ValueError("invalid metadata filter")
        if self.operator == "in" and (
            isinstance(self.value, (str, bytes)) or not isinstance(self.value, Iterable)
        ):
            raise ValueError("the 'in' operator requires a non-string iterable")

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        actual = metadata.get(self.field)
        op, expected = self.operator, self.value
        if op == "=":
            return actual == expected
        if op == "!=":
            return actual != expected
        if op == "in":
            try:
                return actual in expected
            except TypeError:
                return False
        if op == "contains":
            try:
                return expected in actual if actual is not None else False
            except TypeError:
                return False
        if actual is None:
            return False
        try:
            if op == ">":
                return actual > expected
            if op == ">=":
                return actual >= expected
            if op == "<":
                return actual < expected
            return actual <= expected
        except TypeError:
            return False


@dataclass
class RetrievalCandidate:
    chunk_id: str
    document_id: str = ""
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    retrieval_score: float = 0.0
    retrieval_source: str = ""
    # Public relevance scores are normalized to [0, 1]. They are not
    # calibrated probabilities unless the underlying model explicitly emits
    # probabilities. Raw backend/model scores are kept separately for
    # diagnostics and omitted from normal public serialization.
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    final_score_source: str = ""
    retrieval_raw_score: Optional[float] = None
    dense_raw_score: Optional[float] = None
    sparse_raw_score: Optional[float] = None
    fusion_raw_score: Optional[float] = None
    rerank_raw_score: Optional[float] = None
    document_version: int | None = None
    source: str = ""
    page: int | None = None
    section: str | None = None
    sheet: str | None = None
    row: int | None = None
    slide: int | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        self.metadata = metadata
        if self.document_version is None:
            raw_version = metadata.get("version", metadata.get("document_version"))
            self.document_version = int(raw_version) if raw_version not in (None, "") else None
        self.source = self.source or str(metadata.get("source") or metadata.get("source_path") or "")
        self.page = self.page if self.page is not None else metadata.get("page", metadata.get("page_number"))
        self.section = self.section or metadata.get("section") or metadata.get("heading")
        self.sheet = self.sheet or metadata.get("sheet") or metadata.get("sheet_name")
        self.row = self.row if self.row is not None else metadata.get("row", metadata.get("row_number"))
        self.slide = self.slide if self.slide is not None else metadata.get("slide", metadata.get("slide_number"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], source: str = "") -> "RetrievalCandidate":
        cid = value.get("chunk_id", value.get("_id", value.get("id")))
        if not cid:
            raise ValueError("candidate requires chunk_id or _id")
        metadata = dict(value.get("metadata") or {})
        return cls(
            chunk_id=str(cid),
            document_id=str(value.get("document_id", metadata.get("document_id", ""))),
            text=str(value.get("text", "")),
            metadata=metadata,
            retrieval_score=float(value.get("retrieval_score", value.get("score", 0.0))),
            retrieval_source=str(source or value.get("retrieval_source", "")),
            dense_score=value.get("dense_score"),
            sparse_score=value.get("sparse_score"),
            fusion_score=value.get("fusion_score"),
            rerank_score=value.get("rerank_score", value.get("reranker_score")),
            final_score=value.get("final_score"),
            final_score_source=str(value.get("final_score_source", "")),
            retrieval_raw_score=value.get("retrieval_raw_score"),
            dense_raw_score=value.get("dense_raw_score"),
            sparse_raw_score=value.get("sparse_raw_score"),
            fusion_raw_score=value.get("fusion_raw_score"),
            rerank_raw_score=value.get("rerank_raw_score"),
            document_version=value.get("document_version"),
            source=str(value.get("source", "")),
            page=value.get("page"),
            section=value.get("section"),
            sheet=value.get("sheet"),
            row=value.get("row"),
            slide=value.get("slide"),
            rank=value.get("rank"),
        )

    def to_dict(self, *, include_raw_scores: bool = False) -> dict[str, Any]:
        raw_fields = {
            "retrieval_raw_score",
            "dense_raw_score",
            "sparse_raw_score",
            "fusion_raw_score",
            "rerank_raw_score",
        }
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None and (include_raw_scores or key not in raw_fields)
        }


@dataclass(frozen=True)
class LatencyBreakdown:
    query_embedding_ms: float | None = None
    dense_search_ms: float = 0.0
    sparse_search_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float | None]:
        return {
            "query_embedding_ms": self.query_embedding_ms,
            "dense_search_ms": self.dense_search_ms,
            "sparse_search_ms": self.sparse_search_ms,
            "fusion_ms": self.fusion_ms,
            "rerank_ms": self.rerank_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class RetrievalResult:
    # Field order preserves the established constructor used by low-level code.
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    embed_latency_ms: Optional[float] = None
    search_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    mode: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    query: str = ""
    rerank_requested: bool = False
    dense_search_latency_ms: float = 0.0
    sparse_search_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        self.diagnostics = dict(self.diagnostics)
        for position, candidate in enumerate(self.candidates, 1):
            candidate.rank = position

    @property
    def latency(self) -> LatencyBreakdown:
        return LatencyBreakdown(
            query_embedding_ms=self.embed_latency_ms,
            dense_search_ms=self.dense_search_latency_ms,
            sparse_search_ms=self.sparse_search_latency_ms,
            fusion_ms=self.fusion_latency_ms,
            rerank_ms=self.rerank_latency_ms,
            total_ms=self.total_latency_ms,
        )

    def as_dicts(self) -> list[dict[str, Any]]:
        return [candidate.to_dict() for candidate in self.candidates]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "rerank_requested": self.rerank_requested,
            "candidates": self.as_dicts(),
            "latency": self.latency.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    action: str
    version: int | None
    chunk_count: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def require_text(value: Any, field_name: str = "text", max_characters: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    result = value.strip()
    if max_characters is not None and len(result) > max_characters:
        raise ValueError(f"{field_name} exceeds the configured {max_characters} character limit")
    return result


def require_finite_score(value: Any, field_name: str = "score") -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{field_name} must be finite")
    return score


__all__ = [
    "Chunk", "Document", "DocumentSource", "EmbeddingResult", "IngestionResult",
    "LatencyBreakdown", "MetadataFilter", "ParsedDocument", "RetrievalCandidate",
    "RetrievalResult", "require_finite_score", "require_text",
]
