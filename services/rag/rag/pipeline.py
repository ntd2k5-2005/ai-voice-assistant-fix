"""Canonical orchestration for knowledge ingestion and retrieval.

The pipeline owns no global model state and never imports the bundled
``ragflow-main`` reference tree. Ingestion is document-atomic across the
dense and sparse representations through :class:`StorageLifecycleCoordinator`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time
from enum import Enum
import hashlib
import inspect
import json
import logging
import math
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any
import unicodedata
import warnings

import numpy as np

from .config import RagConfig
from .contracts import DocumentSource, IngestionResult, ParsedDocument, RetrievalResult
from .errors import EmptyDocumentError, LifecycleError
from .ingestion.parsers import ParserRegistry
from .lifecycle import DocumentRegistry, StorageLifecycleCoordinator
from .observability import bounded_json
from .processing.clean_text import normalize_document
from .processing.chunking import UltimateChunker
from .retrieval import DenseRetriever, RetrievalEngine, SQLiteBM25Retriever


logger = logging.getLogger(__name__)


_VOLATILE_METADATA_KEYS = frozenset(
    {
        "parsed_at",
        "processed_at",
        "indexed_at",
        "duration",
        "duration_ms",
        "elapsed_ms",
        "source_modified_at",
        "chunk_id",
        "chunk_index",
        "content_hash",
        "index_identity",
        "lifecycle_operation_id",
        "lifecycle_state",
        "status",
        "storage_identity",
        "token_count",
        "token_estimate",
        "token_limit",
        "version",
    }
)


def _canonical_value(value: Any, *, key: str | None = None) -> Any:
    """Return a deterministic JSON value or fail before a lifecycle begins."""

    if key in _VOLATILE_METADATA_KEYS:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("document metadata cannot contain non-finite numbers")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            normalized_key = str(raw_key)
            if normalized_key in _VOLATILE_METADATA_KEYS:
                continue
            result[normalized_key] = _canonical_value(
                value[raw_key], key=normalized_key
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    raise TypeError(
        "document metadata must be JSON-compatible; "
        f"received {type(value).__name__}"
    )


def _canonical_document_content(units: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "text": str(unit.get("text") or ""),
            "source": str(unit.get("source") or ""),
            "source_part": int(unit.get("source_part", index)),
            "metadata": _canonical_value(dict(unit.get("metadata") or {})),
        }
        for index, unit in enumerate(units)
    ]
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _embedding_identity(
    embedder: Any,
) -> tuple[str, int | None, bool | None, str | None, str | None]:
    identity = getattr(embedder, "identity", None)
    model_name = str(
        getattr(embedder, "model_name", "")
        or getattr(identity, "model_name", "")
        or ""
    )
    dimension = getattr(embedder, "embedding_dim", None)
    if dimension is None:
        dimension = getattr(identity, "dimension", None)
    normalized = getattr(identity, "normalized", None)
    if normalized is None:
        normalized = getattr(embedder, "normalize", None)
    revision = getattr(identity, "revision", None) or getattr(
        embedder, "resolved_revision", None
    )
    profile_hash = getattr(identity, "profile_hash", None) or getattr(
        embedder, "profile_hash", None
    )
    return (
        model_name,
        int(dimension) if dimension is not None else None,
        normalized,
        str(revision) if revision is not None else None,
        str(profile_hash) if profile_hash is not None else None,
    )


def _model_tokenizer(embedder: Any):
    current = embedder
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        tokenizer = getattr(current, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        model = getattr(current, "model", None)
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        current = getattr(current, "engine", None)
    return None


def _model_max_length(embedder: Any) -> int | None:
    current = embedder
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, "max_length", None)
        if value is not None:
            try:
                resolved = int(value)
            except (TypeError, ValueError):
                resolved = 0
            if resolved > 1:
                return resolved
        current = getattr(current, "engine", None)
    return None


def _accepts_keyword(callable_value: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callable_value).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _json_safe(value: Any) -> Any:
    """Return JSON-compatible diagnostics without exposing model tensors."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_safe(scalar())
        except (TypeError, ValueError):
            pass
    return str(value)


def _resolved_config(component: Any) -> dict[str, Any]:
    current = component
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, "resolved_config", None)
        if callable(value):
            value = value()
        if isinstance(value, Mapping) and value:
            return _json_safe(value)
        current = getattr(current, "engine", None)
    return {}


class RetrievalPipeline:
    """Long-lived parser/chunker/model/store orchestration.

    ``ingest`` returns canonical chunks for compatibility. Typed operation
    summaries are always available in ``last_ingestion_results`` and are used
    by :class:`rag.service.KnowledgeService`.
    """

    def __init__(
        self,
        embedder,
        store,
        chunker=None,
        reranker=None,
        *,
        config: RagConfig | None = None,
        sparse_store=None,
        registry: DocumentRegistry | None = None,
        parser_registry: ParserRegistry | None = None,
        lifecycle: StorageLifecycleCoordinator | None = None,
    ) -> None:
        self.config = config or RagConfig()
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.chunker = chunker or UltimateChunker(
            config=self.config.chunking,
            model_max_tokens=_model_max_length(embedder),
            tokenizer=_model_tokenizer(embedder),
        )
        self.parser_registry = parser_registry or ParserRegistry(self.config.parser)
        self.sparse_retriever = sparse_store
        self.registry = registry
        self.lifecycle = lifecycle
        self._lock = RLock()
        self._closed = False
        self.retrieval_engine: RetrievalEngine | None = None
        self.last_ingestion_results: list[IngestionResult] = []
        self.last_ingestion_diagnostics: dict[str, Any] = {
            "cleanup_pending": [],
            "cleanup_errors": {},
            "operations": [],
        }

        model_name, dimension, normalized, revision, profile_hash = _embedding_identity(
            embedder
        )
        ensure_identity = getattr(store, "ensure_embedding_identity", None)
        if ensure_identity is not None and dimension is not None:
            parameters = inspect.signature(ensure_identity).parameters
            strict_identity = {}
            if "revision" in parameters:
                strict_identity["revision"] = revision
            if "profile_hash" in parameters:
                strict_identity["profile_hash"] = profile_hash
            ensure_identity(
                model_name,
                dimension,
                normalized,
                **strict_identity,
            )

        if self.lifecycle is not None:
            self.registry = self.lifecycle.registry
            self.store = self.lifecycle.dense_store
            self.sparse_retriever = self.lifecycle.sparse_store
        self._refresh_runtime()

    def _refresh_runtime(self) -> None:
        if self.registry is not None and self.sparse_retriever is not None:
            lifecycle_methods = (
                "stage_chunks",
                "activate_version",
                "verify_chunks",
                "get_manifest",
                "delete_document",
            )
            lifecycle_capable = all(
                all(hasattr(store, name) for name in lifecycle_methods)
                for store in (self.store, self.sparse_retriever)
            )
            if self.lifecycle is None and lifecycle_capable:
                self.lifecycle = StorageLifecycleCoordinator(
                    self.registry,
                    self.store,
                    self.sparse_retriever,
                    cleanup_old_versions=self.config.lifecycle.cleanup_old_versions,
                )
        visibility_registry = None
        if self.registry is not None:
            dense_search = getattr(self.store, "search", None)
            sparse_search = getattr(self.sparse_retriever, "search", None)
            dense_supports = callable(dense_search) and _accepts_keyword(
                dense_search, "visibility_snapshot"
            )
            sparse_supports = self.sparse_retriever is None or (
                callable(sparse_search)
                and _accepts_keyword(sparse_search, "visibility_snapshot")
            )
            # A shared epoch is useful only when every executing branch can
            # enforce it. Persistent production stores do; the optional
            # in-memory sparse experiment deliberately does not claim lifecycle
            # consistency and therefore runs without a shared registry token.
            if dense_supports and sparse_supports:
                visibility_registry = self.registry
        trace = self.config.observability
        if self.sparse_retriever is not None:
            self.retrieval_engine = RetrievalEngine(
                dense=DenseRetriever(self.embedder, self.store),
                sparse=self.sparse_retriever,
                reranker=self.reranker,
                dense_weight=self.config.fusion.dense_weight,
                sparse_weight=self.config.fusion.sparse_weight,
                rerank_timeout_ms=self.config.reranker.timeout_ms,
                fusion_strategy=self.config.fusion.strategy,
                rrf_k=self.config.fusion.rrf_k,
                rerank_failure=self.config.reranker.failure_policy,
                parallel_branches=self.config.retrieval.parallel_hybrid,
                max_query_chars=self.config.retrieval.max_query_characters,
                allow_sparse_rerank=self.config.retrieval.allow_sparse_rerank,
                visibility_registry=visibility_registry,
                visibility_retries=self.config.lifecycle.visibility_retry_count,
                trace_max_items=trace.max_trace_items,
                trace_max_nested_items=trace.max_nested_items,
                trace_text_characters=trace.max_text_characters_per_item,
                trace_vector_dimensions=trace.vector_preview_dimensions,
            )
        else:
            self.retrieval_engine = RetrievalEngine(
                dense=DenseRetriever(self.embedder, self.store),
                reranker=self.reranker,
                rerank_timeout_ms=self.config.reranker.timeout_ms,
                rerank_failure=self.config.reranker.failure_policy,
                max_query_chars=self.config.retrieval.max_query_characters,
                visibility_registry=visibility_registry,
                visibility_retries=self.config.lifecycle.visibility_retry_count,
                trace_max_items=trace.max_trace_items,
                trace_max_nested_items=trace.max_nested_items,
                trace_text_characters=trace.max_text_characters_per_item,
                trace_vector_dimensions=trace.vector_preview_dimensions,
            )

    def build_sparse_index(self, chunks=None):
        """Build an in-memory sparse branch for read-only experiments."""

        from .retrieval import SparseRetriever

        if chunks is None:
            raise ValueError("chunks are required for an in-memory sparse index")
        self.sparse_retriever = SparseRetriever(chunks)
        self.lifecycle = None
        self._refresh_runtime()
        return self.sparse_retriever

    def attach_sparse_store(self, sparse_store):
        self.sparse_retriever = sparse_store
        self.lifecycle = None
        self._refresh_runtime()
        return self

    def build_persistent_sparse_index(self, path):
        self.sparse_retriever = SQLiteBM25Retriever(
            path,
            analyzer=self.config.sparse.analyzer,
            query_operator=self.config.sparse.query_operator,
            timeout_seconds=self.config.lifecycle.lock_timeout_seconds,
        )
        self.lifecycle = None
        self._refresh_runtime()
        return self.sparse_retriever

    def attach_registry(self, registry):
        self.registry = registry
        self.lifecycle = None
        self._refresh_runtime()
        return self

    def _require_lifecycle(self) -> StorageLifecycleCoordinator:
        coordinator = self.lifecycle
        if coordinator is None:
            raise LifecycleError(
                "durable ingestion requires a DocumentRegistry and lifecycle-capable "
                "dense and sparse stores"
            )
        required = (
            "stage_chunks",
            "activate_version",
            "verify_chunks",
            "get_manifest",
            "delete_document",
        )
        for label, store in (("dense", self.store), ("sparse", self.sparse_retriever)):
            missing = [name for name in required if not hasattr(store, name)]
            if missing:
                raise LifecycleError(
                    f"{label} store does not implement lifecycle methods: "
                    + ", ".join(missing)
                )
        return coordinator

    @staticmethod
    def _parsed_mapping(item: ParsedDocument, part_index: int) -> dict[str, Any]:
        metadata = dict(item.metadata)
        metadata.setdefault("source_part", part_index)
        metadata.setdefault("unit_index", item.unit_index)
        page = metadata.get("page", metadata.get("page_number"))
        heading = metadata.get("heading") or metadata.get("section") or item.title
        return {
            "document_id": item.document_id,
            "source": item.source,
            "text": item.content,
            "page": page,
            "heading": heading,
            "method": metadata.get("parser_method"),
            "metadata": metadata,
            "source_part": part_index,
            "file_name": item.file_name,
            "file_type": item.file_type,
            "unit_type": item.unit_type,
            "title": item.title,
        }

    def _source_trace(
        self,
        source: DocumentSource,
        parsed: Sequence[ParsedDocument],
        elapsed_ms: float,
    ) -> dict[str, Any]:
        limit = self.config.observability.max_text_characters_per_item
        file_name = source.file_name
        file_type = None
        size_bytes = 0
        digest = hashlib.sha256()
        identity_kind = "content_sha256"
        raw_preview = None
        source_label = ""
        if source.source_type == "bytes":
            payload = bytes(source.value)
            size_bytes = len(payload)
            digest.update(payload)
            source_label = file_name or "<bytes>"
            if source.mime_type and source.mime_type.startswith("text/"):
                raw_preview = payload.decode("utf-8", errors="replace")[:limit]
        elif source.source_type == "text":
            text = str(source.value)
            payload = text.encode("utf-8")
            size_bytes = len(payload)
            digest.update(payload)
            raw_preview = text[:limit]
            source_label = file_name or "<text>"
        else:
            path = Path(source.value)
            source_label = str(path)
            file_name = file_name or path.name
            file_type = path.suffix.casefold() or None
            try:
                size_bytes = int(path.stat().st_size)
                with path.open("rb") as handle:
                    while block := handle.read(1024 * 1024):
                        digest.update(block)
            except OSError:
                # Parsing already owns source-read errors. Diagnostics remain
                # non-disruptive if a custom parser consumed a virtual path.
                identity_kind = "source_label_sha256"
                digest.update(str(path.resolve(strict=False)).encode("utf-8"))
        if file_type is None:
            candidate = file_name or (parsed[0].file_name if parsed else None)
            file_type = (
                Path(candidate).suffix.casefold() if candidate else None
            ) or (parsed[0].file_type if parsed else None)
        parser_methods = sorted(
            {
                str(item.metadata.get("parser_method") or "unknown")
                for item in parsed
            }
        )
        return {
            "source": source_label,
            "source_type": source.source_type,
            "file_name": file_name or (parsed[0].file_name if parsed else None),
            "file_type": file_type,
            "mime_type": source.mime_type,
            "size_bytes": size_bytes,
            "source_identity": f"sha256:{digest.hexdigest()}",
            "source_identity_kind": identity_kind,
            "parser": parser_methods[0] if len(parser_methods) == 1 else parser_methods,
            "parse_elapsed_ms": float(elapsed_ms),
            "parsed_unit_count": len(parsed),
            "raw_preview": raw_preview,
        }

    def _unit_trace(self, item: Mapping[str, Any], index: int) -> dict[str, Any]:
        metadata = dict(item.get("metadata") or {})
        text = str(item.get("text") or "")
        limit = self.config.observability.max_text_characters_per_item
        provenance_keys = (
            "page", "page_number", "heading", "heading_path", "heading_level",
            "section", "parent_section", "section_depth", "sheet", "sheet_name",
            "row", "row_number", "row_start", "row_end", "slide", "slide_number",
            "table", "table_index", "unit_index", "unit_type", "source_part",
            "parser_source_part", "parser_method", "ocr_used", "url",
        )
        provenance = {
            key: _json_safe(metadata[key])
            for key in provenance_keys
            if key in metadata and metadata[key] is not None
        }
        return {
            "index": int(index),
            "document_id": str(item.get("document_id") or metadata.get("document_id") or ""),
            "source": str(item.get("source") or metadata.get("source") or ""),
            "file_name": item.get("file_name"),
            "file_type": item.get("file_type"),
            "unit_type": item.get("unit_type", metadata.get("unit_type", "text")),
            "parser": item.get("method") or metadata.get("parser_method"),
            "character_count": len(text),
            "text": text[:limit],
            "text_truncated": len(text) > limit,
            "provenance": provenance,
        }

    def _chunk_trace(self, chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        limit = self.config.observability.max_text_characters_per_item
        rows = []
        for index, chunk in enumerate(
            chunks[: self.config.observability.max_trace_items]
        ):
            metadata = dict(chunk.get("metadata") or {})
            text = str(chunk.get("text") or "")
            provenance_keys = (
                "source_part", "source_parts", "component_ids", "logical_start",
                "logical_end", "page", "page_number", "heading", "heading_path",
                "heading_level", "section", "parent_section", "section_depth", "sheet",
                "sheet_name", "row", "row_number", "row_start", "row_end", "slide",
                "slide_number", "table", "table_index", "chunk_kind", "ocr_used",
                "below_min_tokens", "min_chunk_tokens",
            )
            provenance = {
                key: _json_safe(metadata[key])
                for key in provenance_keys
                if key in metadata and metadata[key] is not None
            }
            rows.append(
                {
                    "index": index,
                    "chunk_id": str(chunk.get("chunk_id") or chunk.get("_id") or ""),
                    "document_id": str(chunk.get("document_id") or metadata.get("document_id") or ""),
                    "document_version": int(
                        chunk.get("version", metadata.get("version", 0)) or 0
                    ),
                    "token_count": chunk.get(
                        "token_count", metadata.get("token_count", metadata.get("token_estimate"))
                    ),
                    "character_count": len(text),
                    "text": text[:limit],
                    "text_truncated": len(text) > limit,
                    "source": str(chunk.get("source") or metadata.get("source") or ""),
                    "provenance": provenance,
                }
            )
        return rows

    def ingest_sources(
        self,
        sources: Iterable[DocumentSource],
        parser=None,
        *,
        force: bool = False,
        operation_id: str | None = None,
        include_trace: bool = False,
    ) -> list[dict[str, Any]]:
        parser = parser or self.parser_registry
        documents: list[dict[str, Any]] = []
        next_part: dict[str, int] = defaultdict(int)
        trace_seed: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"sources": []}
        )
        for source in sources:
            if not isinstance(source, DocumentSource):
                raise TypeError("ingest_sources expects DocumentSource instances")
            parse_started = perf_counter()
            parsed = parser.parse(source)
            parse_elapsed_ms = (perf_counter() - parse_started) * 1000
            logger.info(
                "source parsed",
                extra={
                    "rag_event": "source_parsed",
                    "source_type": source.source_type,
                    "parsed_units": len(parsed),
                    "parser_methods": sorted(
                        {
                            str(item.metadata.get("parser_method") or "unknown")
                            for item in parsed
                        }
                    ),
                },
            )
            parsed_document_ids: set[str] = set()
            for part_index, item in enumerate(parsed):
                mapping = self._parsed_mapping(item, part_index)
                document_id = str(mapping["document_id"])
                parsed_document_ids.add(document_id)
                global_part = next_part[document_id]
                next_part[document_id] += 1
                mapping["source_part"] = global_part
                mapping["metadata"] = dict(mapping["metadata"])
                mapping["metadata"]["source_part"] = global_part
                documents.append(mapping)
            if include_trace:
                source_trace = self._source_trace(source, parsed, parse_elapsed_ms)
                for parsed_document_id in parsed_document_ids:
                    trace_seed[parsed_document_id]["sources"].append(source_trace)
        with self._lock:
            self._ensure_open()
            return self._ingest(
                documents,
                force=force,
                operation_id=operation_id,
                include_trace=include_trace,
                trace_seed=trace_seed,
            )

    def ingest(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        force: bool = False,
        operation_id: str | None = None,
        include_trace: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            return self._ingest(
                documents,
                force=force,
                operation_id=operation_id,
                include_trace=include_trace,
            )

    def _prepare_documents(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        include_trace: bool,
        trace_seed: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        part_numbers: dict[str, int] = defaultdict(int)
        traces: dict[str, dict[str, Any]] = {
            str(document_id): {
                "sources": list(value.get("sources") or []),
                "parsed_units": [],
                "normalized_units": [],
                "parsed_units_total": 0,
                "normalized_units_total": 0,
            }
            for document_id, value in (trace_seed or {}).items()
        }
        trace_limit = self.config.observability.max_trace_items
        for raw in documents:
            if not isinstance(raw, Mapping):
                raise TypeError("each document unit must be a mapping")
            item = dict(raw)
            item["metadata"] = dict(item.get("metadata") or {})
            top_document_id = item.get("document_id")
            metadata_document_id = item["metadata"].get("document_id")
            if (
                top_document_id not in (None, "")
                and metadata_document_id not in (None, "")
                and str(top_document_id) != str(metadata_document_id)
            ):
                raise ValueError("top-level and metadata document_id values conflict")
            document_id = str(
                item.get("document_id")
                or item["metadata"].get("document_id")
                or item.get("source")
                or item.get("url")
                or ""
            ).strip()
            if not document_id:
                raise ValueError("each document unit requires document_id or source")
            item["document_id"] = document_id
            item["metadata"]["document_id"] = document_id
            supplied_part = item.get("source_part", item["metadata"].get("source_part"))
            if supplied_part is not None:
                item["metadata"].setdefault("parser_source_part", supplied_part)
            item["source_part"] = part_numbers[document_id]
            item["metadata"]["source_part"] = item["source_part"]
            part_numbers[document_id] += 1
            if include_trace:
                trace = traces.setdefault(
                    document_id,
                    {
                        "sources": [],
                        "parsed_units": [],
                        "normalized_units": [],
                        "parsed_units_total": 0,
                        "normalized_units_total": 0,
                    },
                )
                trace["parsed_units_total"] += 1
                if len(trace["parsed_units"]) < trace_limit:
                    trace["parsed_units"].append(
                        self._unit_trace(item, trace["parsed_units_total"] - 1)
                    )
            normalized = normalize_document(item, self.config.normalization)
            if normalized["text"].strip():
                grouped[document_id].append(normalized)
                if include_trace:
                    trace["normalized_units_total"] += 1
                    if len(trace["normalized_units"]) < trace_limit:
                        trace["normalized_units"].append(
                            self._unit_trace(
                                normalized, trace["normalized_units_total"] - 1
                            )
                        )
        return grouped, traces

    def _embed_chunks(
        self, chunks: list[dict[str, Any]], *, include_trace: bool
    ) -> dict[str, Any]:
        if not hasattr(self.embedder, "embed_documents"):
            raise TypeError("ingestion embedder must implement embed_documents()")
        started = perf_counter()
        raw_vectors = self.embedder.embed_documents([chunk["text"] for chunk in chunks])
        elapsed_ms = (perf_counter() - started) * 1000
        vectors = np.asarray(raw_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise RuntimeError(
                "embed_documents() must preserve input order and return one vector per chunk"
            )
        expected = getattr(self.store, "dim", None)
        if expected is not None and vectors.shape[1] != int(expected):
            raise RuntimeError(
                f"embedding dimension {vectors.shape[1]} does not match store dimension {expected}"
            )
        if not np.isfinite(vectors).all():
            raise RuntimeError("embed_documents() returned non-finite values")
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk["vector"] = vector.tolist()
        if not include_trace:
            return {}
        resolved = _resolved_config(self.embedder)
        norms = np.linalg.norm(vectors, axis=1)
        preview_dimensions = min(
            vectors.shape[1], self.config.observability.vector_preview_dimensions
        )
        preview_count = min(
            vectors.shape[0], self.config.observability.max_trace_items
        )
        return {
            "model": resolved.get("model_name")
            or getattr(self.embedder, "model_name", None),
            "family": resolved.get("family"),
            "dimension": int(vectors.shape[1]),
            "chunk_count": int(vectors.shape[0]),
            "batch_size": resolved.get("batch_size"),
            "normalized": resolved.get("normalize"),
            "elapsed_ms": float(elapsed_ms),
            "vector_norm": {
                "min": float(norms.min()),
                "max": float(norms.max()),
                "mean": float(norms.mean()),
            },
            "finite": bool(np.isfinite(vectors).all()),
            "vector_previews": [
                [float(value) for value in row[:preview_dimensions]]
                for row in vectors[:preview_count]
            ],
            "preview_dimensions": preview_dimensions,
            "vectors_omitted": (
                preview_dimensions < vectors.shape[1]
                or preview_count < vectors.shape[0]
            ),
            "previews_truncated": vectors.shape[0] > preview_count,
        }

    def _abort_pre_storage(
        self, operation: Mapping[str, Any], error: BaseException
    ) -> None:
        if operation.get("unchanged") or self.registry is None:
            return
        self.registry.abort_ingest(
            str(operation["document_id"]),
            f"pre-storage ingestion failed: {type(error).__name__}: {error}",
            operation_id=str(operation["operation_id"]),
        )

    def _ingest(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        force: bool,
        operation_id: str | None,
        include_trace: bool,
        trace_seed: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        coordinator = self._require_lifecycle()
        grouped, traces = self._prepare_documents(
            documents,
            include_trace=include_trace,
            trace_seed=trace_seed,
        )
        if not grouped:
            self.last_ingestion_results = []
            return []
        if operation_id is not None and len(grouped) != 1:
            raise ValueError("one operation_id can address exactly one document")

        all_chunks: list[dict[str, Any]] = []
        summaries: list[IngestionResult] = []
        operations: list[dict[str, Any]] = []
        cleanup_pending: list[str] = []
        cleanup_errors: dict[str, str] = {}

        # Each document has its own compare-and-set transaction. A multi-
        # document call is deliberately not advertised as a global transaction.
        for document_id, units in grouped.items():
            canonical_content = _canonical_document_content(units)
            first = units[0]
            filename = str(
                first.get("file_name") or first.get("source") or document_id
            )
            operation = coordinator.begin_ingest(
                document_id,
                filename,
                canonical_content,
                force=force,
                operation_id=operation_id,
            )
            operations.append(dict(operation))
            trace_context = traces.get(document_id, {})
            sources = list(trace_context.get("sources") or [])
            if sources:
                source_trace = dict(sources[0])
                if len(sources) > 1:
                    source_limit = self.config.observability.max_trace_items
                    source_trace.update(
                        {
                            "source_count": len(sources),
                            "size_bytes": sum(
                                int(item.get("size_bytes") or 0) for item in sources
                            ),
                            "parsed_unit_count": sum(
                                int(item.get("parsed_unit_count") or 0)
                                for item in sources
                            ),
                            "parse_elapsed_ms": sum(
                                float(item.get("parse_elapsed_ms") or 0.0)
                                for item in sources
                            ),
                            "items": sources[:source_limit],
                            "sources_truncated": len(sources) > source_limit,
                        }
                    )
            else:
                raw_size = sum(
                    len(str(unit.get("text") or "").encode("utf-8"))
                    for unit in units
                )
                source_trace = {
                    "source": str(first.get("source") or document_id),
                    "source_type": "parsed_mapping",
                    "file_name": first.get("file_name") or filename,
                    "file_type": first.get("file_type")
                    or Path(filename).suffix.casefold()
                    or None,
                    "mime_type": None,
                    "size_bytes": raw_size,
                    "source_identity": "sha256:"
                    + hashlib.sha256(canonical_content.encode("utf-8")).hexdigest(),
                    "source_identity_kind": "canonical_units_sha256",
                    "parser": first.get("method")
                    or dict(first.get("metadata") or {}).get("parser_method")
                    or "direct",
                    "parse_elapsed_ms": None,
                    "parsed_unit_count": len(units),
                    "raw_preview": None,
                }
            logger.info(
                "document ingestion started",
                extra={
                    "rag_event": "ingest_start",
                    "document_id": document_id,
                    "action": operation.get("action"),
                    "target_version": operation.get("pending_version", operation.get("version")),
                    "parsed_units": len(units),
                },
            )
            if operation.get("unchanged"):
                count = getattr(self.store, "count_document", lambda *_a, **_k: 0)(
                    document_id, version=int(operation["version"])
                )
                unchanged_diagnostics: dict[str, Any] = {
                    "operation_id": operation["operation_id"],
                    "unchanged": True,
                }
                if include_trace:
                    active_chunks = []
                    list_chunks = getattr(self.store, "list_chunks", None)
                    if callable(list_chunks):
                        active_chunks = list_chunks(
                            document_id,
                            version=int(operation["version"]),
                            state="active",
                            limit=self.config.observability.max_inspection_chunks,
                        )
                    unchanged_diagnostics["trace"] = {
                        "source": source_trace,
                        "parsed_units": list(trace_context.get("parsed_units") or []),
                        "normalized_units": list(
                            trace_context.get("normalized_units") or []
                        ),
                        "chunks": self._chunk_trace(active_chunks),
                        "embedding": {
                            "status": "skipped",
                            "reason": "Canonical content hash matches the active version.",
                            "chunk_count": int(count),
                            "elapsed_ms": 0.0,
                            "vector_previews": [],
                        },
                        "sparse": {
                            "status": "unchanged",
                            "reason": "The existing verified sparse index remains active.",
                            "chunk_count": int(count),
                            "analyzer": self.config.sparse.analyzer,
                            "elapsed_ms": 0.0,
                        },
                        "lifecycle": {
                            "document_id": document_id,
                            "version": int(operation["version"]),
                            "operation_id": str(operation["operation_id"]),
                            "previous_version": int(operation["version"]),
                            "transitions": [
                                {"stage": "content_hash_match", "status": "unchanged"}
                            ],
                            "verification_status": "existing_active_version",
                            "active": True,
                        },
                        "truncation": {
                            "max_items": self.config.observability.max_trace_items,
                            "sources_truncated": bool(
                                source_trace.get("sources_truncated", False)
                            ),
                            "parsed_units_truncated": int(
                                trace_context.get("parsed_units_total") or 0
                            ) > len(trace_context.get("parsed_units") or []),
                            "normalized_units_truncated": int(
                                trace_context.get("normalized_units_total") or 0
                            ) > len(trace_context.get("normalized_units") or []),
                            "chunks_truncated": int(count) > len(active_chunks),
                        },
                    }
                summaries.append(
                    IngestionResult(
                        document_id=document_id,
                        action="unchanged",
                        version=int(operation["version"]),
                        chunk_count=int(count),
                        diagnostics=unchanged_diagnostics,
                    )
                )
                logger.info(
                    "document unchanged",
                    extra={
                        "rag_event": "ingest_unchanged",
                        "document_id": document_id,
                        "version": int(operation["version"]),
                        "chunks": int(count),
                    },
                )
                continue

            version = int(operation["pending_version"])
            chunks: list[dict[str, Any]] = []
            embedding_trace: dict[str, Any] = {}
            try:
                versioned_units: list[dict[str, Any]] = []
                for unit in units:
                    versioned = dict(unit)
                    versioned["metadata"] = dict(unit.get("metadata") or {})
                    versioned["version"] = version
                    versioned["metadata"]["version"] = version
                    versioned_units.append(versioned)

                chunk_documents = getattr(self.chunker, "chunk_documents", None)
                if callable(chunk_documents):
                    # Batch-aware chunking can preserve heading/subsection
                    # context across ordered parser units and suppress bare
                    # heading-only searchable chunks.
                    chunks = list(chunk_documents(versioned_units))
                else:
                    for versioned in versioned_units:
                        chunks.extend(self.chunker.chunk_document(versioned))
                    finalize_chunks = getattr(self.chunker, "finalize_document_chunks", None)
                    if callable(finalize_chunks):
                        chunks = list(finalize_chunks(chunks))
                if not chunks:
                    raise EmptyDocumentError(
                        f"document {document_id!r} produced no indexable chunks"
                    )
                embedding_trace = self._embed_chunks(
                    chunks, include_trace=include_trace
                )
                logger.info(
                    "document chunks embedded",
                    extra={
                        "rag_event": "chunks_embedded",
                        "document_id": document_id,
                        "version": version,
                        "chunks": len(chunks),
                    },
                )
            except BaseException as exc:
                self._abort_pre_storage(operation, exc)
                raise

            lifecycle_started = perf_counter()
            result = coordinator.stage_and_activate(
                operation,
                chunks,
                content=canonical_content,
                filename=filename,
            )
            lifecycle_elapsed_ms = (perf_counter() - lifecycle_started) * 1000
            activated = list(result["chunks"])
            all_chunks.extend(activated)
            diagnostics = {
                key: value
                for key, value in result.items()
                if key not in {"chunks", "document_id", "version", "action"}
            }
            if include_trace:
                previous_version = int(operation.get("version") or 0)
                lifecycle_timings = dict(result.get("timings_ms") or {})
                transitions = [
                    {"stage": "registry_begin", "status": "pending"},
                    {"stage": "dense_write", "status": "staged"},
                    {"stage": "dense_verify", "status": "verified"},
                    {"stage": "sparse_write", "status": "staged"},
                    {"stage": "sparse_verify", "status": "verified"},
                    {"stage": "physical_readiness", "status": "active_registry_gated"},
                    {"stage": "registry_cas", "status": "committed"},
                    {
                        "stage": "old_version_cleanup",
                        "status": (
                            "not_applicable"
                            if previous_version <= 0
                            else "pending"
                            if result.get("cleanup_pending")
                            else "retained_stale"
                            if result.get("old_versions_retained")
                            else "complete"
                        ),
                    },
                ]
                diagnostics["trace"] = {
                    "source": source_trace,
                    "parsed_units": list(trace_context.get("parsed_units") or []),
                    "normalized_units": list(
                        trace_context.get("normalized_units") or []
                    ),
                    "chunks": self._chunk_trace(activated),
                    "embedding": embedding_trace,
                    "sparse": {
                        "chunk_count": len(activated),
                        "analyzer": self.config.sparse.analyzer,
                        "query_operator": self.config.sparse.query_operator,
                        "status": "verified_active",
                        "elapsed_ms": float(
                            lifecycle_timings.get("sparse_write", 0.0)
                            + lifecycle_timings.get("sparse_verify", 0.0)
                        ),
                        "write_elapsed_ms": lifecycle_timings.get("sparse_write"),
                        "verify_elapsed_ms": lifecycle_timings.get("sparse_verify"),
                        "timing_scope": "sparse_stage_and_verification",
                    },
                    "lifecycle": {
                        "document_id": document_id,
                        "version": int(result["version"]),
                        "operation_id": str(result["operation_id"]),
                        "previous_version": previous_version,
                        "transitions": transitions,
                        "verification_status": "dense_sparse_parity_verified",
                        "active": True,
                        "cleanup_pending": bool(result.get("cleanup_pending")),
                        "cleanup_error": result.get("cleanup_error"),
                        "elapsed_ms": float(
                            lifecycle_timings.get("total", lifecycle_elapsed_ms)
                        ),
                        "timings_ms": _json_safe(lifecycle_timings),
                    },
                    "truncation": {
                        "max_items": self.config.observability.max_trace_items,
                        "max_text_characters_per_item": (
                            self.config.observability.max_text_characters_per_item
                        ),
                        "sources_truncated": bool(
                            source_trace.get("sources_truncated", False)
                        ),
                        "parsed_units_truncated": int(
                            trace_context.get("parsed_units_total") or 0
                        ) > len(trace_context.get("parsed_units") or []),
                        "normalized_units_truncated": int(
                            trace_context.get("normalized_units_total") or 0
                        ) > len(trace_context.get("normalized_units") or []),
                        "chunks_truncated": len(activated)
                        > self.config.observability.max_trace_items,
                    },
                }
            summaries.append(
                IngestionResult(
                    document_id=document_id,
                    action=str(result["action"]),
                    version=int(result["version"]),
                    chunk_count=len(activated),
                    diagnostics=diagnostics,
                )
            )
            if result.get("cleanup_pending"):
                cleanup_pending.append(document_id)
            if result.get("cleanup_error"):
                cleanup_errors[document_id] = str(result["cleanup_error"])
            logger.info(
                "document ingestion completed",
                extra={
                    "rag_event": "ingest_complete",
                    "document_id": document_id,
                    "action": result.get("action"),
                    "version": result.get("version"),
                    "chunks": len(activated),
                    "cleanup_pending": bool(result.get("cleanup_pending")),
                    "old_versions_retained": bool(result.get("old_versions_retained")),
                },
            )

        self.last_ingestion_results = summaries
        self.last_ingestion_diagnostics = {
            "cleanup_pending": cleanup_pending,
            "cleanup_errors": cleanup_errors,
            "operations": operations,
        }
        return all_chunks

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        value = unicodedata.normalize("NFC", query.strip())
        if not value:
            raise ValueError("query must be a non-empty string")
        if any(unicodedata.category(char) == "Cc" and not char.isspace() for char in value):
            raise ValueError("query contains unsupported control characters")
        return value

    def retrieve_result(
        self,
        query: str,
        top_k: int | None = None,
        query_filter=None,
        mode: str | None = None,
        candidate_k: int | None = None,
        *,
        rerank: bool = False,
        filters=None,
        include_trace: bool = False,
    ) -> RetrievalResult:
        with self._lock:
            self._ensure_open()
            if filters is not None and query_filter is not None:
                raise ValueError("pass either filters or query_filter, not both")
            resolved_filters = filters if filters is not None else query_filter
            normalized_mode = str(
                self.config.retrieval.default_mode if mode is None else mode
            ).casefold()
            if normalized_mode == "hybrid_rerank":
                warnings.warn(
                    "mode='hybrid_rerank' is deprecated; use mode='hybrid', rerank=True",
                    DeprecationWarning,
                    stacklevel=2,
                )
                normalized_mode, rerank = "hybrid", True
            query_preparation_started = perf_counter()
            normalized_query = self._normalize_query(query)
            query_preparation_ms = (
                perf_counter() - query_preparation_started
            ) * 1000
            engine = self.retrieval_engine
            if engine is None:
                raise RuntimeError("retrieval engine is not initialized")
            result = engine.retrieve(
                normalized_query,
                top_k=self.config.retrieval.top_k if top_k is None else int(top_k),
                candidate_k=(
                    self.config.retrieval.candidate_k
                    if candidate_k is None
                    else int(candidate_k)
                ),
                filters=resolved_filters,
                rerank=bool(rerank),
                mode=normalized_mode,
                include_trace=bool(include_trace),
            )
            if include_trace:
                trace = dict(result.diagnostics.get("trace") or {})
                query_trace = dict(trace.get("query") or {})
                query_trace["original"] = query
                query_trace["normalized"] = normalized_query
                query_trace["normalization_changed"] = query != normalized_query
                query_trace["preparation_elapsed_ms"] = float(query_preparation_ms)
                trace["query"] = query_trace
                latency_trace = dict(trace.get("latency") or {})
                latency_trace["query_preparation_ms"] = float(query_preparation_ms)
                trace["latency"] = latency_trace
                result.diagnostics["trace"] = trace
            logger.info(
                "retrieval completed",
                extra={
                    "rag_event": "retrieval_complete",
                    "mode": result.mode,
                    "candidates": len(result.candidates),
                    "rerank_requested": result.rerank_requested,
                    "rerank_applied": bool(result.diagnostics.get("rerank_applied")),
                    "fallback": bool(result.diagnostics.get("rerank_fallback")),
                    "total_latency_ms": result.total_latency_ms,
                },
            )
            return result

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        query_filter=None,
        mode: str | None = None,
        candidate_k: int | None = None,
        *,
        rerank: bool = False,
        filters=None,
        include_trace: bool = False,
    ) -> list[dict[str, Any]]:
        return self.retrieve_result(
            query,
            top_k,
            query_filter,
            mode,
            candidate_k,
            rerank=rerank,
            filters=filters,
            include_trace=include_trace,
        ).as_dicts()

    def delete_document(
        self, document_id: str, *, operation_id: str | None = None
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            value = str(document_id).strip()
            if not value:
                raise ValueError("document_id must be a non-empty string")
            result = self._require_lifecycle().delete(value, operation_id=operation_id)
            logger.info(
                "document deletion completed",
                extra={
                    "rag_event": "delete_complete",
                    "document_id": value,
                    "unchanged": bool(result.get("unchanged")),
                },
            )
            return result

    def recover(self) -> list[str]:
        with self._lock:
            self._ensure_open()
            recovered = self._require_lifecycle().recover()
            logger.info(
                "lifecycle recovery completed",
                extra={
                    "rag_event": "recovery_complete",
                    "recovered_documents": len(recovered),
                },
            )
            return recovered

    def inspect(
        self,
        document_id: str | None = None,
        *,
        include_chunks: bool = False,
        include_chunk_counts: bool = False,
    ):
        if self.registry is None:
            raise LifecycleError("document registry is not attached")
        registry_value = self.registry.inspect(document_id)

        def with_chunk_counts(value):
            if value is None:
                return None
            is_single = isinstance(value, Mapping)
            rows = [dict(value)] if is_single else [dict(row) for row in value]
            counter = getattr(self.store, "count_document", None)
            for row in rows:
                visible_version = self.registry.visible_version(
                    str(row["document_id"])
                )
                if visible_version is None:
                    row["chunk_count"] = 0
                elif callable(counter):
                    row["chunk_count"] = int(
                        counter(
                            str(row["document_id"]),
                            version=int(visible_version),
                        )
                    )
                else:
                    row["chunk_count"] = None
            return rows[0] if is_single else rows

        if not include_chunks:
            return (
                with_chunk_counts(registry_value)
                if include_chunk_counts
                else registry_value
            )
        if registry_value is None:
            documents: list[dict[str, Any]] = []
        elif isinstance(registry_value, Mapping):
            documents = [dict(registry_value)]
        else:
            documents = [dict(row) for row in registry_value]
        list_chunks = getattr(self.store, "list_chunks", None)
        if not callable(list_chunks):
            raise LifecycleError("dense store does not expose read-only chunk inspection")
        limit = self.config.observability.max_inspection_chunks
        active_chunks: list[dict[str, Any]] = []
        total_active_chunks = 0
        counter = getattr(self.store, "count_document", None)
        for row in documents:
            remaining = limit - len(active_chunks)
            visible_version = self.registry.visible_version(str(row["document_id"]))
            if visible_version is None:
                row["chunk_count"] = 0
                continue
            document_chunk_count = (
                int(
                    counter(
                        str(row["document_id"]), version=int(visible_version)
                    )
                )
                if callable(counter)
                else None
            )
            if document_chunk_count is not None:
                row["chunk_count"] = document_chunk_count
                total_active_chunks += document_chunk_count
            if remaining <= 0:
                continue
            values = list_chunks(
                str(row["document_id"]),
                version=int(visible_version),
                state="active",
                limit=remaining,
            )
            for value in values:
                safe_value = dict(value)
                text = str(safe_value.get("text") or "")
                safe_value["text"] = text[
                    : self.config.observability.max_text_characters_per_item
                ]
                safe_metadata, counters = bounded_json(
                    safe_value.get("metadata") or {},
                    max_items=self.config.observability.max_nested_items,
                    max_string_characters=(
                        self.config.observability.max_text_characters_per_item
                    ),
                )
                safe_value["metadata"] = safe_metadata
                if len(text) > self.config.observability.max_text_characters_per_item:
                    counters["strings_truncated"] += 1
                    safe_value["text_truncated"] = True
                nonzero = {
                    key: count for key, count in counters.items() if count
                }
                if nonzero:
                    safe_value["diagnostic_bounds"] = nonzero
                active_chunks.append(safe_value)
            row["inspected_chunk_count"] = len(values)
        if not callable(counter):
            total_active_chunks = len(active_chunks)
        return {
            "documents": _json_safe(documents),
            "active_chunks": active_chunks,
            "active_chunk_count": total_active_chunks,
            "inspected_chunk_count": len(active_chunks),
            "chunk_limit": limit,
            "chunk_limit_reached": len(active_chunks) >= limit,
            "chunks_truncated": total_active_chunks > len(active_chunks),
        }

    def runtime_snapshot(self) -> dict[str, Any]:
        """Expose configured intent and resolved runtime facts, read-only."""

        configured = self.config.snapshot()
        payload = json.dumps(
            configured,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        embedding = _resolved_config(self.embedder)
        reranker = _resolved_config(self.reranker)
        registry_value = [] if self.registry is None else self.registry.inspect()
        if registry_value is None:
            registry_rows = []
        elif isinstance(registry_value, Mapping):
            registry_rows = [dict(registry_value)]
        else:
            registry_rows = [dict(row) for row in registry_value]
        active_rows = [
            row
            for row in registry_rows
            if row.get("hash") and row.get("status") in {"active", "indexing"}
        ]
        counter = getattr(self.chunker, "counter", None)
        tokenizer = getattr(counter, "tokenizer", None)
        tokenizer_name = (
            getattr(tokenizer, "name_or_path", None)
            or self.config.chunking.tokenizer_name
            or type(counter).__name__
        )
        dense_identity = getattr(self.store, "index_identity", {})
        if callable(dense_identity):
            dense_identity = dense_identity()
        sparse_identity = getattr(self.sparse_retriever, "index_identity", {})
        if callable(sparse_identity):
            sparse_identity = sparse_identity()
        visibility_epoch = None
        if self.registry is not None:
            snapshotter = getattr(self.registry, "visibility_snapshot", None)
            if callable(snapshotter):
                visibility_epoch = getattr(snapshotter(), "epoch", None)
        resolved = {
            "parser": {
                "registry": type(self.parser_registry).__name__,
                "allowed_extensions": list(self.config.parser.allowed_extensions),
                "sniff_mime": self.config.parser.sniff_mime,
                "ocr_strategy": self.config.parser.ocr.strategy,
                "ocr_backend": self.config.parser.ocr.backend,
                "ocr_fallback": (
                    ["tesseract", "rapidocr"]
                    if self.config.parser.ocr.backend == "auto"
                    else [self.config.parser.ocr.backend]
                ),
            },
            "chunking": {
                "implementation": type(self.chunker).__name__,
                "effective_max_tokens": int(
                    getattr(self.chunker, "effective_max_tokens", 0)
                ),
                "max_tokens": int(getattr(self.chunker, "max_tokens", 0)),
                "min_chunk_tokens": int(
                    getattr(self.chunker, "min_chunk_tokens", 0)
                ),
                "overlap_tokens": int(getattr(self.chunker, "overlap_tokens", 0)),
                "reserved_tokens": int(getattr(self.chunker, "reserved_tokens", 0)),
                "tokenizer": str(tokenizer_name),
                "token_counter": type(counter).__name__,
            },
            "embedding": {
                **embedding,
                "max_sequence_length": embedding.get("max_length"),
                "normalized": embedding.get("normalize"),
            },
            "sparse": {
                "analyzer": self.config.sparse.analyzer,
                "query_operator": self.config.sparse.query_operator,
                "available": self.sparse_retriever is not None,
                "index_identity": _json_safe(sparse_identity),
            },
            "retrieval": {
                "default_mode": self.config.retrieval.default_mode,
                "top_k": self.config.retrieval.top_k,
                "candidate_k": self.config.retrieval.candidate_k,
                "fusion_strategy": self.config.fusion.strategy,
                "parallel_hybrid": self.config.retrieval.parallel_hybrid,
                "visibility_retry_count": self.config.lifecycle.visibility_retry_count,
            },
            "reranker": {
                **reranker,
                "available": self.reranker is not None,
                "max_length": reranker.get("max_length"),
            },
            "storage": {
                "dense": _json_safe(dense_identity),
                "sparse": _json_safe(sparse_identity),
                "active_document_count": len(active_rows),
                "active_version_count": len(active_rows),
            },
            "lifecycle": {
                "coordinator": type(self.lifecycle).__name__
                if self.lifecycle is not None
                else None,
                "recover_on_startup": self.config.lifecycle.recover_on_startup,
                "cleanup_old_versions": self.config.lifecycle.cleanup_old_versions,
                "visibility_epoch": visibility_epoch,
            },
            "observability": self.config.snapshot()["observability"],
        }
        return {
            "configured": configured,
            "resolved": _json_safe(resolved),
            "config_fingerprint": hashlib.sha256(payload).hexdigest(),
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("pipeline is closed")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            seen: set[int] = set()
            errors: list[BaseException] = []
            for component in (
                self.reranker,
                self.embedder,
                self.sparse_retriever,
                self.store,
                self.registry,
            ):
                if component is None or id(component) in seen:
                    continue
                seen.add(id(component))
                closer = getattr(component, "close", None)
                if closer is None:
                    client = getattr(component, "client", None)
                    closer = getattr(client, "close", None)
                if closer is not None:
                    try:
                        closer()
                    except BaseException as exc:  # finish closing other resources
                        errors.append(exc)
            if errors:
                raise RuntimeError(
                    "one or more pipeline resources failed to close: "
                    + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
                ) from errors[0]

    def __enter__(self) -> "RetrievalPipeline":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def create_canonical_pipeline(config: RagConfig | None = None) -> RetrievalPipeline:
    """Instantiate the sole production model/store/lifecycle composition."""

    from .Embeddings.capabilities import EmbeddingProfileOverrides
    from .Embeddings.embeddings import EmbeddingEngine
    from .Embeddings.factory import adapter_for
    from .retrieval.reranker_capabilities import RerankerProfileOverrides
    from .retrieval.reranking import Reranker
    from .storage.qdrant import QdrantStore

    config = config or RagConfig()
    embedding = config.embedding
    embedding_overrides = EmbeddingProfileOverrides(
        family=embedding.profile_family,
        query_template=embedding.query_template,
        document_template=embedding.document_template,
        query_instruction=embedding.query_instruction,
        document_instruction=embedding.document_instruction,
        normalize=embedding.normalize,
        requires_remote_code=embedding.profile_requires_remote_code,
    )
    engine = EmbeddingEngine(
        model_name=embedding.model_name,
        device=embedding.device,
        batch_size=embedding.batch_size,
        normalize=embedding.normalize,
        safety_margin=embedding.batch_safety_margin,
        local_files_only=embedding.local_files_only,
        trust_remote_code=embedding.trust_remote_code,
        revision=embedding.revision,
        dtype=embedding.dtype,
        max_length=embedding.max_length,
        fallback_max_length=embedding.fallback_max_length,
        dimension=embedding.dimension,
        max_oom_retries=embedding.oom_retries,
        profile_overrides=embedding_overrides,
    )
    embedder = adapter_for(engine, model_name=embedding.model_name)

    store = QdrantStore(
        collection_name=config.dense_store.collection_name,
        dim=embedder.embedding_dim,
        persist_dir=str(config.paths.qdrant),
        embedding_model=embedding.model_name,
        normalization=engine.normalize,
        embedding_revision=getattr(engine, "resolved_revision", embedding.revision),
        embedding_profile_hash=getattr(engine, "profile_hash", None),
        distance=config.dense_store.distance,
    )
    registry = DocumentRegistry(
        config.paths.registry,
        timeout_seconds=config.lifecycle.lock_timeout_seconds,
    )
    sparse_store = SQLiteBM25Retriever(
        config.paths.sparse,
        analyzer=config.sparse.analyzer,
        query_operator=config.sparse.query_operator,
        timeout_seconds=config.lifecycle.lock_timeout_seconds,
    )
    lifecycle = StorageLifecycleCoordinator(
        registry,
        store,
        sparse_store,
        cleanup_old_versions=config.lifecycle.cleanup_old_versions,
    )

    reranker = None
    reranker_config = config.reranker
    if reranker_config.model_name:
        architecture = {
            "auto": None,
            "sequence_classification": "sequence_classification",
            "causal_lm": "causal_yes_no",
        }[reranker_config.adapter]
        reranker_overrides = RerankerProfileOverrides(
            family=reranker_config.profile_family,
            architecture=architecture,
            score_extraction=reranker_config.score_extraction,
            positive_label_index=reranker_config.positive_label_index,
            higher_is_better=reranker_config.higher_is_better,
            pair_template=reranker_config.pair_template,
            instruction=reranker_config.instruction,
            requires_remote_code=reranker_config.profile_requires_remote_code,
        )
        reranker = Reranker(
            model_name=reranker_config.model_name,
            device=reranker_config.device,
            max_length=reranker_config.max_length,
            trust_remote_code=reranker_config.trust_remote_code,
            batch_size=reranker_config.batch_size,
            dtype=reranker_config.dtype,
            local_files_only=reranker_config.local_files_only,
            revision=reranker_config.revision,
            code_revision=reranker_config.code_revision,
            timeout_ms=reranker_config.timeout_ms,
            fallback_max_length=reranker_config.fallback_max_length,
            max_oom_retries=reranker_config.oom_retries,
            profile_overrides=reranker_overrides,
            yes_token_id=reranker_config.yes_token_id,
            no_token_id=reranker_config.no_token_id,
        )

    tokenizer = _model_tokenizer(embedder)
    chunker = UltimateChunker(
        config=config.chunking,
        model_max_tokens=engine.max_length,
        tokenizer=tokenizer,
    )
    pipeline = RetrievalPipeline(
        embedder=embedder,
        store=store,
        chunker=chunker,
        reranker=reranker,
        config=config,
        sparse_store=sparse_store,
        registry=registry,
        parser_registry=ParserRegistry(config.parser),
        lifecycle=lifecycle,
    )
    if config.lifecycle.recover_on_startup:
        pipeline.recover()
    return pipeline


__all__ = ["RetrievalPipeline", "create_canonical_pipeline"]
