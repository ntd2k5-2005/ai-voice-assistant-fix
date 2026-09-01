"""Stable HTTP request/response schemas for the RAG service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetadataFilterRequest(StrictModel):
    field: str = Field(min_length=1, max_length=256)
    operator: Literal["=", "!=", ">", ">=", "<", "<=", "in", "contains"]
    value: Any


class RetrievalRequest(StrictModel):
    query: str = Field(min_length=1)
    mode: Literal["dense", "sparse", "hybrid"] | None = None
    rerank: bool = True
    top_k: int | None = Field(default=None, ge=1)
    candidate_k: int | None = Field(default=None, ge=1)
    filters: list[MetadataFilterRequest] = Field(default_factory=list, max_length=100)
    debug: bool = False

    @model_validator(mode="after")
    def validate_candidate_window(self) -> "RetrievalRequest":
        if (
            self.top_k is not None
            and self.candidate_k is not None
            and self.candidate_k < self.top_k
        ):
            raise ValueError("candidate_k must be >= top_k")
        return self


class ScoreBreakdown(StrictModel):
    dense: float | None = Field(default=None, ge=0.0, le=1.0)
    sparse: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion: float | None = Field(default=None, ge=0.0, le=1.0)
    rerank: float | None = Field(default=None, ge=0.0, le=1.0)


class RawScoreBreakdown(StrictModel):
    dense: float | None = None
    sparse: float | None = None
    fusion: float | None = None
    rerank: float | None = None
    retrieval: float | None = None


class RetrievalContext(StrictModel):
    rank: int = Field(ge=1)
    chunk_id: str
    document_id: str
    document_version: int | None = None
    text: str
    source: str | None = None
    page: int | None = None
    section: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    sheet: str | None = None
    row: int | None = None
    slide: int | None = None
    score: float = Field(ge=0.0, le=1.0)
    score_source: Literal["dense", "sparse", "fusion", "rerank", "retrieval"]
    scores: ScoreBreakdown | None = None
    raw_scores: RawScoreBreakdown | None = None
    metadata: dict[str, Any] | None = None


class LatencyResponse(StrictModel):
    total_ms: float = Field(ge=0.0)
    query_embedding_ms: float | None = Field(default=None, ge=0.0)
    dense_search_ms: float | None = Field(default=None, ge=0.0)
    sparse_search_ms: float | None = Field(default=None, ge=0.0)
    fusion_ms: float | None = Field(default=None, ge=0.0)
    rerank_ms: float | None = Field(default=None, ge=0.0)


class RetrievalResponse(StrictModel):
    query: str
    mode: Literal["dense", "sparse", "hybrid"]
    rerank_requested: bool
    rerank_applied: bool
    contexts: list[RetrievalContext]
    latency: LatencyResponse
    diagnostics: dict[str, Any] | None = None


class TextIngestionRequest(StrictModel):
    document_id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    operation_id: str | None = Field(default=None, min_length=1, max_length=256)
    debug: bool = False


class IngestionResponse(StrictModel):
    document_id: str
    action: str
    version: int | None = None
    chunk_count: int = Field(ge=0)
    active: bool | None = None
    operation_id: str | None = None
    cleanup_pending: bool | None = None
    diagnostics: dict[str, Any] | None = None


class DeleteResponse(StrictModel):
    document_id: str
    action: str
    version: int | None = None
    deleted: bool
    operation_id: str | None = None


class KnowledgeDocument(StrictModel):
    document_id: str
    filename: str | None = None
    version: int | None = None
    status: str | None = None
    content_hash: str | None = None
    created_at: float | str | None = None
    updated_at: float | str | None = None
    last_error: str | None = None
    cleanup_status: str | None = None
    cleanup_error: str | None = None
    chunk_count: int | None = Field(default=None, ge=0)


class KnowledgeListResponse(StrictModel):
    documents: list[KnowledgeDocument]


class KnowledgeDetailResponse(StrictModel):
    document: KnowledgeDocument
    chunks: list[dict[str, Any]] | None = None
    chunks_truncated: bool = False
    inspected_chunk_count: int | None = None
    active_chunk_count: int | None = None


class RecoveryResponse(StrictModel):
    recovered_operations: list[str]


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(StrictModel):
    status: Literal["ready"] = "ready"
    config_fingerprint: str | None = None
    active_document_count: int | None = None


class ErrorResponse(StrictModel):
    code: str
    message: str
    details: Any | None = None


__all__ = [name for name in tuple(globals()) if name.endswith(("Request", "Response")) or name in {
    "KnowledgeDocument", "MetadataFilterRequest", "RawScoreBreakdown",
    "RetrievalContext", "ScoreBreakdown",
}]
