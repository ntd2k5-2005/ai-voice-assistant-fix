from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LooseModel(BaseModel):
    # Upstream services may add fields over time; adapters only require
    # the verified fields they consume.
    model_config = ConfigDict(extra="allow")


class ServiceStatus(LooseModel):
    service: str
    reachable: bool
    ready: bool | None = None
    latency_ms: float | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class STTResult(LooseModel):
    text: str
    model: str | None = None
    sample_rate: int | None = None
    audio_duration_s: float | None = None
    inference_ms: float | None = None
    rtf: float | None = None
    client_latency_ms: float


class RAGContext(LooseModel):
    rank: int
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
    score: float
    score_source: str
    metadata: dict[str, Any] | None = None


class RAGLatency(LooseModel):
    total_ms: float
    query_embedding_ms: float | None = None
    dense_search_ms: float | None = None
    sparse_search_ms: float | None = None
    fusion_ms: float | None = None
    rerank_ms: float | None = None


class RAGResult(LooseModel):
    query: str
    mode: Literal["dense", "sparse", "hybrid"]
    rerank_requested: bool
    rerank_applied: bool
    contexts: list[RAGContext]
    latency: RAGLatency
    client_latency_ms: float


class ChatMessage(LooseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class LLMResult(LooseModel):
    text: str
    model: str | None = None
    timestamp: str | None = None
    request_id: str | None = None
    client_latency_ms: float


class TTSResult(LooseModel):
    audio_path: str
    sample_rate: int
    channels: int
    dtype: str
    duration_s: float
    byte_count: int
    ttfb_ms: float | None = None
    total_ms: float


class SourceView(LooseModel):
    rank: int
    document_id: str
    chunk_id: str
    score: float
    score_source: str
    source: str | None = None
    section: str | None = None
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    excerpt: str


class PipelineTimings(LooseModel):
    stt_ms: float
    rag_ms: float
    llm_ms: float
    tts_ms: float
    total_ms: float
    rag_internal_ms: float | None = None
    stt_inference_ms: float | None = None
    tts_ttfb_ms: float | None = None


class PipelineResult(LooseModel):
    transcript: str
    answer: str
    audio_path: str | None = None
    sources: list[SourceView] = Field(default_factory=list)
    timings: PipelineTimings
    rag_mode: str
    rerank_applied: bool
    llm_model: str | None = None
    service_details: dict[str, Any] = Field(default_factory=dict)
