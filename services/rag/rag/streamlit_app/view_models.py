"""Pure presentation models for the Streamlit inspection console.

These helpers do not import Streamlit and never execute parsing, embedding,
search, fusion, reranking, or storage operations. They only shape diagnostics
returned by the public :mod:`rag` facade into deterministic display data.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from rag.config import RagConfig


UNAVAILABLE = "UNAVAILABLE"


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _mapping(value: Any) -> dict[str, Any]:
    plain = _plain(value)
    if isinstance(plain, Mapping):
        return dict(plain)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, Mapping) else {}
    if hasattr(value, "__dict__"):
        return {str(key): _plain(item) for key, item in vars(value).items()}
    return {}


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [_mapping(item) for item in value]


_MISSING = object()


def _dig(mapping: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _display(value: Any) -> Any:
    if value is _MISSING:
        return UNAVAILABLE
    if value is None:
        return "auto / not explicitly configured"
    if isinstance(value, tuple):
        return list(value)
    return value


def _config_row(
    label: str,
    configured: Mapping[str, Any],
    resolved: Mapping[str, Any],
    configured_path: str,
    resolved_path: str | None = None,
    *,
    runtime_derived: bool = False,
) -> dict[str, Any]:
    configured_value = _dig(configured, configured_path)
    resolved_value = _dig(resolved, resolved_path or configured_path)
    if resolved_value is _MISSING and not runtime_derived:
        resolved_value = configured_value
    status = (
        "unavailable"
        if resolved_value is _MISSING
        else "same"
        if resolved_value == configured_value
        else "resolved"
    )
    return {
        "setting": label,
        "configured": _display(configured_value),
        "resolved": _display(resolved_value),
        "status": status,
    }


def build_config_view(
    config: RagConfig,
    runtime_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build configured-versus-resolved rows for every UI config section."""

    snapshot = _mapping(runtime_snapshot or {})
    configured = _mapping(snapshot.get("configured") or config.snapshot())
    resolved = _mapping(snapshot.get("resolved") or {})

    def rows(*specs):
        return [
            _config_row(label, configured, resolved, cpath, rpath, runtime_derived=dynamic)
            for label, cpath, rpath, dynamic in specs
        ]

    return {
        "Parser / OCR": rows(
            ("Enabled formats", "parser.allowed_extensions", None, False),
            ("Maximum file bytes", "parser.max_file_bytes", None, False),
            ("Maximum pages", "parser.max_pages", None, False),
            ("Maximum parsed units", "parser.max_units", None, False),
            ("OCR strategy", "parser.ocr.strategy", None, False),
            ("OCR configured backend", "parser.ocr.backend", None, False),
            ("OCR ordered fallback", "parser.ocr.backend", "parser.ocr_fallback", True),
            ("OCR languages", "parser.ocr.languages", None, False),
        ),
        "Chunking": rows(
            ("Configured maximum tokens", "chunking.max_tokens", None, False),
            ("Effective maximum tokens", "chunking.max_tokens", "chunking.effective_max_tokens", True),
            ("Minimum chunk tokens", "chunking.min_chunk_tokens", None, False),
            ("Overlap tokens", "chunking.overlap_tokens", None, False),
            ("Reserved tokens", "chunking.reserved_tokens", None, False),
            ("Preserve tables", "chunking.preserve_tables", None, False),
            ("Counting tokenizer", "chunking.tokenizer_name", "chunking.tokenizer", True),
        ),
        "Embedding": rows(
            ("Model", "embedding.model_name", "embedding.model_name", False),
            ("Family / profile", "embedding.profile_family", "embedding.family", True),
            ("Dimension", "embedding.dimension", "embedding.dimension", True),
            ("Device", "embedding.device", "embedding.device", True),
            ("Dtype", "embedding.dtype", "embedding.dtype", True),
            ("Maximum sequence length", "embedding.max_length", "embedding.max_sequence_length", True),
            ("Batch size", "embedding.batch_size", "embedding.batch_size", True),
            ("Normalization", "embedding.normalize", "embedding.normalized", True),
            ("Revision", "embedding.revision", "embedding.revision", True),
            ("Trust remote code", "embedding.trust_remote_code", None, False),
            ("Cache only", "embedding.local_files_only", None, False),
        ),
        "Sparse": rows(
            ("Analyzer", "sparse.analyzer", "sparse.analyzer", False),
            ("Query operator", "sparse.query_operator", None, False),
        ),
        "Retrieval / fusion": rows(
            ("Default mode", "retrieval.default_mode", None, False),
            ("Default top_k", "retrieval.top_k", None, False),
            ("Default candidate_k", "retrieval.candidate_k", None, False),
            ("Fusion strategy", "fusion.strategy", None, False),
            ("Dense weight", "fusion.dense_weight", None, False),
            ("Sparse weight", "fusion.sparse_weight", None, False),
            ("RRF k", "fusion.rrf_k", None, False),
            ("Parallel hybrid", "retrieval.parallel_hybrid", None, False),
        ),
        "Reranker": rows(
            ("Model", "reranker.model_name", "reranker.model_name", False),
            ("Family / profile", "reranker.profile_family", "reranker.family", True),
            ("Adapter", "reranker.adapter", "reranker.adapter", True),
            ("Score extraction", "reranker.score_extraction", "reranker.score_extraction", True),
            ("Runtime available", "reranker.model_name", "reranker.available", True),
            ("Device", "reranker.device", "reranker.device", True),
            ("Dtype", "reranker.dtype", "reranker.dtype", True),
            ("Maximum length", "reranker.max_length", "reranker.max_length", True),
            ("Batch size", "reranker.batch_size", "reranker.batch_size", True),
            ("Failure policy", "reranker.failure_policy", None, False),
            ("Revision", "reranker.revision", "reranker.revision", True),
            ("Code revision", "reranker.code_revision", "reranker.code_revision", True),
            ("Trust remote code", "reranker.trust_remote_code", None, False),
            ("Cache only", "reranker.local_files_only", None, False),
        ),
        "Storage / lifecycle": rows(
            ("Dense collection", "dense_store.collection_name", None, False),
            ("Dense path", "paths.qdrant", None, False),
            ("Sparse database", "paths.sparse", None, False),
            ("Registry database", "paths.registry", None, False),
            ("Recover on startup", "lifecycle.recover_on_startup", None, False),
            ("Cleanup old versions", "lifecycle.cleanup_old_versions", None, False),
            ("Visibility retry count", "lifecycle.visibility_retry_count", None, False),
            ("Active documents", "paths.registry", "storage.active_document_count", True),
            ("Active versions", "paths.registry", "storage.active_version_count", True),
        ),
        "Inspection trace bounds": rows(
            ("Maximum traced items", "observability.max_trace_items", None, False),
            (
                "Maximum text characters per item",
                "observability.max_text_characters_per_item",
                None,
                False,
            ),
            (
                "Vector preview dimensions",
                "observability.vector_preview_dimensions",
                None,
                False,
            ),
            (
                "Maximum inspected chunks",
                "observability.max_inspection_chunks",
                None,
                False,
            ),
        ),
    }


def chunk_statistics(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize trace-provided token counts without retokenizing in the UI."""

    counts: list[int] = []
    for chunk in chunks:
        raw = chunk.get("token_count")
        if isinstance(raw, bool):
            continue
        try:
            count = int(raw)
        except (TypeError, ValueError):
            continue
        if count >= 0:
            counts.append(count)
    if not counts:
        return {
            "chunk_count": len(chunks),
            "token_counts_available": False,
            "min_tokens": None,
            "max_tokens": None,
            "mean_tokens": None,
            "median_tokens": None,
        }
    return {
        "chunk_count": len(chunks),
        "token_counts_available": len(counts) == len(chunks),
        "min_tokens": min(counts),
        "max_tokens": max(counts),
        "mean_tokens": float(mean(counts)),
        "median_tokens": float(median(counts)),
    }


def _trace(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    value = diagnostics.get("trace", diagnostics.get("ingestion_trace", {}))
    return _mapping(value)


def build_ingestion_view(result: Any) -> list[dict[str, Any]]:
    """Convert one or more public ``IngestionResult`` objects into stage views."""

    results = result if isinstance(result, (list, tuple)) else [result]
    documents: list[dict[str, Any]] = []
    for raw in results:
        summary = _mapping(raw)
        diagnostics = _mapping(summary.get("diagnostics") or {})
        trace = _trace(diagnostics)
        parsed = _list_of_mappings(trace.get("parsed_units"))
        normalized = _list_of_mappings(trace.get("normalized_units"))
        chunks = _list_of_mappings(trace.get("chunks"))
        sources = _list_of_mappings(trace.get("sources"))
        source = _mapping(trace.get("source"))
        if not source and len(sources) == 1:
            source = sources[0]
        lifecycle = _mapping(trace.get("lifecycle"))
        for key in ("document_id", "version", "operation_id"):
            if key not in lifecycle:
                fallback = summary.get(key, diagnostics.get(key))
                if fallback is not None:
                    lifecycle[key] = fallback
        lifecycle.setdefault("action", summary.get("action"))
        lifecycle.setdefault("active", summary.get("action") != "unchanged" or summary.get("version") is not None)
        documents.append(
            {
                "summary": summary,
                "trace_available": bool(trace),
                "source": source,
                "sources": sources or ([source] if source else []),
                "parsed_units": parsed,
                "normalized_units": normalized,
                "chunks": chunks,
                "chunk_statistics": chunk_statistics(chunks),
                "embedding": _mapping(trace.get("embedding")),
                "sparse": _mapping(trace.get("sparse")),
                "lifecycle": lifecycle,
                "diagnostics": diagnostics,
            }
        )
    return documents


def candidate_row(value: Any) -> dict[str, Any]:
    """Flatten a candidate while preserving ``None`` as not-applicable."""

    candidate = _mapping(value)
    metadata = _mapping(candidate.get("metadata"))
    row = {
        "rank": candidate.get("rank"),
        "chunk_id": candidate.get("chunk_id", candidate.get("_id", candidate.get("id"))),
        "document_id": candidate.get("document_id", metadata.get("document_id")),
        "document_version": candidate.get(
            "document_version", metadata.get("version", metadata.get("document_version"))
        ),
        "source": candidate.get("source") or metadata.get("source") or metadata.get("source_path"),
        "page": candidate.get("page", metadata.get("page", metadata.get("page_number"))),
        "section": candidate.get("section") or metadata.get("section") or metadata.get("heading"),
        "sheet": candidate.get("sheet") or metadata.get("sheet") or metadata.get("sheet_name"),
        "row": candidate.get("row", metadata.get("row", metadata.get("row_number"))),
        "slide": candidate.get("slide", metadata.get("slide", metadata.get("slide_number"))),
        "text": candidate.get("text", ""),
        "dense_rank": candidate.get("dense_rank"),
        "sparse_rank": candidate.get("sparse_rank"),
        "fusion_rank": candidate.get("fusion_rank"),
        "pre_rerank_rank": candidate.get("pre_rerank_rank"),
        "post_rerank_rank": candidate.get("post_rerank_rank"),
        "dense_score": candidate.get("dense_score"),
        "sparse_score": candidate.get("sparse_score"),
        "fusion_score": candidate.get("fusion_score"),
        "rerank_score": candidate.get("rerank_score"),
        "retrieval_score": candidate.get("retrieval_score"),
        "final_score": candidate.get("final_score"),
        "final_score_source": candidate.get("final_score_source"),
        "dense_raw_score": candidate.get("dense_raw_score"),
        "sparse_raw_score": candidate.get("sparse_raw_score"),
        "fusion_raw_score": candidate.get("fusion_raw_score"),
        "rerank_raw_score": candidate.get("rerank_raw_score"),
        "retrieval_raw_score": candidate.get("retrieval_raw_score"),
        "retrieval_source": candidate.get("retrieval_source"),
        "metadata": metadata,
    }
    return row


def _candidate_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [candidate_row(item) for item in value]


def build_retrieval_view(result: Any, *, original_query: str | None = None) -> dict[str, Any]:
    """Build a truthful view of every retrieval stage from public diagnostics."""

    result_map = _mapping(result)
    diagnostics = _mapping(result_map.get("diagnostics") or {})
    trace = _mapping(diagnostics.get("trace", diagnostics.get("retrieval_trace", {})))
    query = _mapping(trace.get("query"))
    mode = str(result_map.get("mode") or query.get("mode") or "")
    if original_query is not None:
        # The public engine receives the validated/normalized query; only the
        # caller can preserve the exact widget input without pretending it was
        # a runtime-normalized value.
        query["original"] = original_query
    else:
        query.setdefault("original", result_map.get("query", ""))
    query.setdefault("normalized", result_map.get("query", ""))
    query.setdefault("mode", mode)
    query.setdefault("embedding_executed", mode in {"dense", "hybrid"})
    query.setdefault("sparse_executed", mode in {"sparse", "hybrid"})
    query.setdefault("rerank_requested", bool(result_map.get("rerank_requested")))

    embedding = _mapping(trace.get("query_embedding"))
    if mode == "sparse":
        embedding.setdefault("executed", False)
        embedding.setdefault("reason", "Sparse retrieval mode does not execute query embedding.")
    elif embedding:
        embedding.setdefault("executed", True)

    dense_present = "dense_candidates" in trace
    sparse_present = "sparse_candidates" in trace
    dense = _candidate_rows(trace.get("dense_candidates"))
    sparse = _candidate_rows(trace.get("sparse_candidates"))

    fusion = _mapping(trace.get("fusion"))
    if mode != "hybrid":
        fusion = {
            **fusion,
            "executed": False,
            "reason": "Fusion is only executed for hybrid retrieval.",
            "candidates": [],
        }
    else:
        fusion.setdefault("executed", True)
        fusion.setdefault("strategy", diagnostics.get("fusion_strategy"))
        fusion["candidates"] = _candidate_rows(fusion.get("candidates"))

    rerank = _mapping(trace.get("rerank"))
    rerank_requested = bool(result_map.get("rerank_requested"))
    if not rerank_requested:
        rerank = {
            **rerank,
            "executed": False,
            "reason": "Reranking was not requested.",
            "before": [],
            "after": [],
        }
    else:
        rerank.setdefault("executed", bool(diagnostics.get("rerank_applied")))
        rerank.setdefault("applied", bool(diagnostics.get("rerank_applied")))
        rerank.setdefault("fallback", bool(diagnostics.get("rerank_fallback")))
        rerank.setdefault("error", diagnostics.get("rerank_error"))
        rerank["before"] = _candidate_rows(rerank.get("before"))
        rerank["after"] = _candidate_rows(rerank.get("after"))

    latency = _mapping(result_map.get("latency"))
    if not latency and hasattr(result, "latency"):
        latency = _mapping(getattr(result, "latency"))
    trace_latency = _mapping(trace.get("latency"))
    if trace_latency:
        latency = {**latency, **trace_latency}

    final = _candidate_rows(result_map.get("candidates"))
    warnings = diagnostics.get("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
    elif not isinstance(warnings, Sequence):
        warnings = []

    return {
        "trace_available": bool(trace),
        "query": query,
        "query_embedding": embedding,
        "dense": {
            "executed": mode in {"dense", "hybrid"},
            "available": dense_present,
            "candidates": dense,
        },
        "sparse": {
            "executed": mode in {"sparse", "hybrid"},
            "available": sparse_present,
            "candidates": sparse,
        },
        "fusion": fusion,
        "rerank": rerank,
        "final": final,
        "latency": latency,
        "diagnostics": diagnostics,
        "warnings": list(warnings),
    }


def build_inspection_view(payload: Any) -> dict[str, Any]:
    """Normalize public knowledge inspection output without touching stores."""

    root = _plain(payload)
    documents: list[dict[str, Any]] = []
    root_chunks: list[dict[str, Any]] = []
    if root is None:
        pass
    elif isinstance(root, list):
        documents = [_mapping(item) for item in root]
    elif isinstance(root, Mapping):
        if isinstance(root.get("documents"), list):
            documents = [_mapping(item) for item in root["documents"]]
            root_chunks = _list_of_mappings(root.get("active_chunks"))
        elif isinstance(root.get("document"), Mapping):
            documents = [_mapping(root["document"])]
            root_chunks = _list_of_mappings(root.get("active_chunks"))
        else:
            documents = [_mapping(root)]

    summary_rows: list[dict[str, Any]] = []
    chunks_by_document: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        document_id = str(document.get("document_id") or "")
        chunks = _list_of_mappings(
            document.get("active_chunks", document.get("chunks", []))
        )
        if chunks:
            chunks_by_document[document_id] = chunks
        summary_rows.append(
            {
                "document_id": document_id,
                "source": document.get("source") or document.get("filename"),
                "version": document.get("version"),
                "hash": document.get("hash") or document.get("content_hash"),
                "state": document.get("status") or document.get("state"),
                "chunk_count": document.get("chunk_count", len(chunks) if chunks else None),
                "created_at": document.get("created_at"),
                "updated_at": document.get("updated_at"),
                "operation_id": document.get("operation_id") or document.get("pending_operation_id"),
            }
        )
    for chunk in root_chunks:
        document_id = str(
            chunk.get("document_id") or _mapping(chunk.get("metadata")).get("document_id") or ""
        )
        chunks_by_document.setdefault(document_id, []).append(chunk)
    return {
        "documents": documents,
        "summary_rows": summary_rows,
        "chunks_by_document": chunks_by_document,
        "empty": not documents,
    }


def finite_vector_summary(values: Sequence[Any]) -> dict[str, Any]:
    """Small pure helper used by tests and trace validation."""

    numbers = [float(value) for value in values]
    return {
        "dimension": len(numbers),
        "finite": all(math.isfinite(value) for value in numbers),
        "norm": math.sqrt(sum(value * value for value in numbers)),
    }


__all__ = [
    "UNAVAILABLE",
    "build_config_view",
    "build_ingestion_view",
    "build_inspection_view",
    "build_retrieval_view",
    "candidate_row",
    "chunk_statistics",
    "finite_vector_summary",
]
