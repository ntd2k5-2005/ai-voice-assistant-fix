"""Conversion between dependency-free RAG contracts and HTTP schemas."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contracts import IngestionResult, MetadataFilter, RetrievalCandidate, RetrievalResult
from ..observability import bounded_json
from .schemas import (
    DeleteResponse,
    IngestionResponse,
    KnowledgeDetailResponse,
    KnowledgeDocument,
    KnowledgeListResponse,
    LatencyResponse,
    RawScoreBreakdown,
    RetrievalContext,
    RetrievalResponse,
    ScoreBreakdown,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def filters_from_request(values) -> list[MetadataFilter] | None:
    if not values:
        return None
    return [MetadataFilter(item.field, item.operator, item.value) for item in values]


def _heading_path(candidate: RetrievalCandidate) -> list[str]:
    raw = candidate.metadata.get("heading_path")
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item).strip()]
    if candidate.section:
        return [part.strip() for part in str(candidate.section).split(">") if part.strip()]
    return []


def _canonical_score(candidate: RetrievalCandidate) -> tuple[float, str]:
    pairs = (
        (candidate.final_score, candidate.final_score_source),
        (candidate.rerank_score, "rerank"),
        (candidate.fusion_score, "fusion"),
        (candidate.dense_score, "dense"),
        (candidate.sparse_score, "sparse"),
    )
    for score, source in pairs:
        if score is not None:
            value = min(1.0, max(0.0, float(score)))
            normalized_source = source if source in {"dense", "sparse", "fusion", "rerank"} else "retrieval"
            return value, normalized_source
    return min(1.0, max(0.0, float(candidate.retrieval_score))), "retrieval"


def retrieval_to_response(
    result: RetrievalResult,
    *,
    debug: bool,
    observability: Any | None = None,
) -> RetrievalResponse:
    contexts: list[RetrievalContext] = []
    for position, candidate in enumerate(result.candidates, 1):
        score, source = _canonical_score(candidate)
        contexts.append(
            RetrievalContext(
                rank=int(candidate.rank or position),
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                document_version=candidate.document_version,
                text=candidate.text,
                source=candidate.source or None,
                page=candidate.page,
                section=candidate.section,
                heading_path=_heading_path(candidate),
                sheet=candidate.sheet,
                row=candidate.row,
                slide=candidate.slide,
                score=score,
                score_source=source,
                scores=(
                    ScoreBreakdown(
                        dense=candidate.dense_score,
                        sparse=candidate.sparse_score,
                        fusion=candidate.fusion_score,
                        rerank=candidate.rerank_score,
                    )
                    if debug
                    else None
                ),
                raw_scores=(
                    RawScoreBreakdown(
                        dense=candidate.dense_raw_score,
                        sparse=candidate.sparse_raw_score,
                        fusion=candidate.fusion_raw_score,
                        rerank=candidate.rerank_raw_score,
                        retrieval=candidate.retrieval_raw_score,
                    )
                    if debug
                    else None
                ),
                metadata=(
                    bounded_json(
                        candidate.metadata,
                        max_items=int(getattr(observability, "max_nested_items", 100)),
                        max_string_characters=int(
                            getattr(observability, "max_text_characters_per_item", 4_000)
                        ),
                    )[0]
                    if debug
                    else None
                ),
            )
        )
    latency = result.latency
    return RetrievalResponse(
        query=result.query,
        mode=result.mode,
        rerank_requested=result.rerank_requested,
        rerank_applied=bool(result.diagnostics.get("rerank_applied")),
        contexts=contexts,
        latency=LatencyResponse(
            total_ms=max(0.0, float(latency.total_ms)),
            query_embedding_ms=(
                None
                if latency.query_embedding_ms is None
                else max(0.0, float(latency.query_embedding_ms))
            ),
            dense_search_ms=max(0.0, float(latency.dense_search_ms)),
            sparse_search_ms=max(0.0, float(latency.sparse_search_ms)),
            fusion_ms=max(0.0, float(latency.fusion_ms)),
            rerank_ms=max(0.0, float(latency.rerank_ms)),
        ),
        diagnostics=_json_safe(result.diagnostics) if debug else None,
    )


def ingestion_to_response(result: IngestionResult, *, debug: bool) -> IngestionResponse:
    diagnostics = dict(result.diagnostics or {})
    lifecycle = diagnostics.get("trace", {}).get("lifecycle", {}) if isinstance(diagnostics.get("trace"), Mapping) else {}
    return IngestionResponse(
        document_id=result.document_id,
        action=result.action,
        version=result.version,
        chunk_count=result.chunk_count,
        active=(
            bool(lifecycle.get("active"))
            if isinstance(lifecycle, Mapping) and "active" in lifecycle
            else (result.action not in {"delete", "failed"})
        ),
        operation_id=(
            str(diagnostics.get("operation_id")) if diagnostics.get("operation_id") else None
        ),
        cleanup_pending=(
            bool(diagnostics.get("cleanup_pending"))
            if "cleanup_pending" in diagnostics
            else None
        ),
        diagnostics=_json_safe(diagnostics) if debug else None,
    )


def delete_to_response(document_id: str, value: Mapping[str, Any]) -> DeleteResponse:
    return DeleteResponse(
        document_id=document_id,
        action=str(value.get("action") or "delete"),
        version=(int(value["version"]) if value.get("version") is not None else None),
        deleted=str(value.get("action") or "delete") == "delete",
        operation_id=(str(value["operation_id"]) if value.get("operation_id") else None),
    )


def _document_from_row(row: Mapping[str, Any]) -> KnowledgeDocument:
    content_hash = row.get("hash", row.get("content_hash"))
    return KnowledgeDocument(
        document_id=str(row.get("document_id") or ""),
        filename=(str(row["filename"]) if row.get("filename") else None),
        version=(int(row["version"]) if row.get("version") is not None else None),
        status=(str(row["status"]) if row.get("status") else None),
        content_hash=(str(content_hash) if content_hash else None),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        last_error=(str(row["last_error"]) if row.get("last_error") else None),
        cleanup_status=(str(row["cleanup_status"]) if row.get("cleanup_status") else None),
        cleanup_error=(str(row["cleanup_error"]) if row.get("cleanup_error") else None),
        chunk_count=(int(row["chunk_count"]) if row.get("chunk_count") is not None else None),
    )


def knowledge_list_to_response(value: Any) -> KnowledgeListResponse:
    if value is None:
        rows: list[Mapping[str, Any]] = []
    elif isinstance(value, Mapping):
        rows = [value]
    else:
        rows = list(value)
    return KnowledgeListResponse(documents=[_document_from_row(row) for row in rows])


def knowledge_detail_to_response(value: Any, *, include_chunks: bool) -> KnowledgeDetailResponse | None:
    if value is None:
        return None
    if not include_chunks:
        if not isinstance(value, Mapping):
            raise TypeError("single-document inspection must return a mapping")
        return KnowledgeDetailResponse(document=_document_from_row(value))
    if not isinstance(value, Mapping):
        raise TypeError("chunk inspection must return a mapping")
    documents = list(value.get("documents") or [])
    if not documents:
        return None
    return KnowledgeDetailResponse(
        document=_document_from_row(documents[0]),
        chunks=[_json_safe(row) for row in list(value.get("active_chunks") or [])],
        chunks_truncated=bool(value.get("chunks_truncated", False)),
        inspected_chunk_count=(
            int(value["inspected_chunk_count"])
            if value.get("inspected_chunk_count") is not None
            else None
        ),
        active_chunk_count=(
            int(value["active_chunk_count"])
            if value.get("active_chunk_count") is not None
            else None
        ),
    )


__all__ = [
    "delete_to_response", "filters_from_request", "ingestion_to_response",
    "knowledge_detail_to_response", "knowledge_list_to_response",
    "retrieval_to_response",
]
