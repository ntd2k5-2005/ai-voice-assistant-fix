"""Small public facade over the canonical pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import RagConfig
from .contracts import DocumentSource, IngestionResult, RetrievalResult
from .errors import SourceTooLargeError
from .pipeline import RetrievalPipeline, create_canonical_pipeline


def _path_is_file(value: str) -> bool:
    try:
        return Path(value).is_file()
    except (OSError, ValueError):
        return False


def _is_file_like(value: Any) -> bool:
    return not isinstance(value, (DocumentSource, str, bytes, Path)) and (
        callable(getattr(value, "read", None))
        or callable(getattr(value, "getvalue", None))
    )


class KnowledgeService:
    """Add, update, reindex, inspect, recover, and delete knowledge."""

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        self.pipeline = pipeline

    def _coerce_source(
        self,
        value: DocumentSource | str | bytes | Path,
        *,
        source_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        document_id: str | None = None,
        file_name: str | None = None,
        mime_type: str | None = None,
    ) -> DocumentSource:
        merged = dict(metadata or {})
        if document_id is not None:
            merged["document_id"] = str(document_id)
        if isinstance(value, DocumentSource):
            if source_type is not None and source_type != value.source_type:
                raise ValueError("source_type conflicts with the supplied DocumentSource")
            return DocumentSource(
                value=value.value,
                source_type=value.source_type,
                metadata={**dict(value.metadata), **merged},
                mime_type=mime_type or value.mime_type,
                file_name=file_name or value.file_name,
            )
        if _is_file_like(value):
            if source_type not in (None, "bytes"):
                raise ValueError("file-like sources require source_type='bytes'")
            limit = int(self.pipeline.config.parser.max_file_bytes)
            reader = getattr(value, "read", None)
            position = None
            tell = getattr(value, "tell", None)
            seek = getattr(value, "seek", None)
            if callable(tell):
                try:
                    position = tell()
                except (OSError, ValueError):
                    position = None
            try:
                if callable(seek):
                    try:
                        seek(0)
                    except (OSError, ValueError):
                        pass
                if callable(reader):
                    try:
                        payload = reader(limit + 1)
                    except TypeError:
                        payload = None
                else:
                    payload = None
                if payload is None:
                    getter = getattr(value, "getvalue", None)
                    if not callable(getter):
                        raise TypeError("file-like source read() must accept a size bound")
                    declared_size = getattr(value, "size", None)
                    buffer_getter = getattr(value, "getbuffer", None)
                    if declared_size is None and callable(buffer_getter):
                        buffer = buffer_getter()
                        try:
                            declared_size = len(buffer)
                        finally:
                            release = getattr(buffer, "release", None)
                            if callable(release):
                                release()
                    if (
                        isinstance(declared_size, bool)
                        or not isinstance(declared_size, int)
                        or declared_size < 0
                    ):
                        raise TypeError(
                            "getvalue-only source must expose an integer .size or getbuffer() "
                            "so the read can be bounded"
                        )
                    if declared_size > limit:
                        raise SourceTooLargeError(
                            f"uploaded source exceeds configured limit of {limit} bytes"
                        )
                    payload = getter()
            finally:
                if position is not None and callable(seek):
                    try:
                        seek(position)
                    except (OSError, ValueError):
                        pass
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            elif isinstance(payload, (bytearray, memoryview)):
                payload = bytes(payload)
            if not isinstance(payload, bytes):
                raise TypeError("file-like source must yield bytes or text")
            if len(payload) > limit:
                raise SourceTooLargeError(
                    f"uploaded source exceeds configured limit of {limit} bytes"
                )
            inferred_name = getattr(value, "name", None)
            file_name = file_name or (
                Path(str(inferred_name)).name if inferred_name else None
            )
            if not file_name:
                raise ValueError("file-like source requires file_name or a .name attribute")
            mime_type = mime_type or getattr(value, "type", None)
            value = payload
            resolved_type = "bytes"
        elif isinstance(value, bytes):
            resolved_type = source_type or "bytes"
        elif isinstance(value, Path):
            resolved_type = source_type or "file"
        elif isinstance(value, str):
            resolved_type = source_type or ("file" if _path_is_file(value) else "text")
        else:
            raise TypeError(
                "source must be DocumentSource, str, bytes, Path, or a bounded file-like object"
            )
        return DocumentSource(
            value=value,
            source_type=resolved_type,
            metadata=merged,
            mime_type=mime_type,
            file_name=file_name,
        )

    @staticmethod
    def _return_results(
        results: list[IngestionResult],
    ) -> IngestionResult | list[IngestionResult]:
        if len(results) == 1:
            return results[0]
        return results

    def ingest(
        self,
        source: DocumentSource | str | bytes | Path | Iterable[DocumentSource],
        *,
        source_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        document_id: str | None = None,
        file_name: str | None = None,
        mime_type: str | None = None,
        operation_id: str | None = None,
        include_trace: bool = False,
    ) -> IngestionResult | list[IngestionResult]:
        if isinstance(source, (DocumentSource, str, bytes, Path)) or _is_file_like(source):
            sources = [
                self._coerce_source(
                    source,
                    source_type=source_type,
                    metadata=metadata,
                    document_id=document_id,
                    file_name=file_name,
                    mime_type=mime_type,
                )
            ]
        else:
            if any(
                value is not None
                for value in (source_type, metadata, document_id, file_name, mime_type)
            ):
                raise ValueError(
                    "per-source options must be encoded in each DocumentSource "
                    "when ingesting an iterable"
                )
            sources = list(source)
            sources = [self._coerce_source(item) for item in sources]
        self.pipeline.ingest_sources(
            sources,
            operation_id=operation_id,
            include_trace=include_trace,
        )
        return self._return_results(list(self.pipeline.last_ingestion_results))

    def update(self, *args, **kwargs):
        """Content-aware alias; the lifecycle reports ``update`` or ``unchanged``."""

        return self.ingest(*args, **kwargs)

    def reindex(
        self,
        source: DocumentSource | str | bytes | Path | Iterable[DocumentSource],
        *,
        source_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        document_id: str | None = None,
        file_name: str | None = None,
        mime_type: str | None = None,
        operation_id: str | None = None,
        include_trace: bool = False,
    ) -> IngestionResult | list[IngestionResult]:
        if isinstance(source, (DocumentSource, str, bytes, Path)) or _is_file_like(source):
            sources = [
                self._coerce_source(
                    source,
                    source_type=source_type,
                    metadata=metadata,
                    document_id=document_id,
                    file_name=file_name,
                    mime_type=mime_type,
                )
            ]
        else:
            if any(
                value is not None
                for value in (source_type, metadata, document_id, file_name, mime_type)
            ):
                raise ValueError(
                    "per-source options must be encoded in each DocumentSource "
                    "when reindexing an iterable"
                )
            sources = list(source)
            sources = [self._coerce_source(item) for item in sources]
        self.pipeline.ingest_sources(
            sources,
            force=True,
            operation_id=operation_id,
            include_trace=include_trace,
        )
        return self._return_results(list(self.pipeline.last_ingestion_results))

    def delete(
        self, document_id: str, *, operation_id: str | None = None
    ) -> Mapping[str, Any]:
        return self.pipeline.delete_document(document_id, operation_id=operation_id)

    def recover(self) -> list[str]:
        return self.pipeline.recover()

    def inspect(
        self,
        document_id: str | None = None,
        *,
        include_chunks: bool = False,
        include_chunk_counts: bool = False,
    ):
        return self.pipeline.inspect(
            document_id,
            include_chunks=include_chunks,
            include_chunk_counts=include_chunk_counts,
        )


class RagModule:
    """Canonical application entry point for the two supported flows."""

    def __init__(
        self,
        config: RagConfig | None = None,
        *,
        pipeline: RetrievalPipeline | None = None,
    ) -> None:
        if pipeline is not None and config is not None and pipeline.config != config:
            raise ValueError("injected pipeline config does not match the supplied config")
        self.pipeline = pipeline or create_canonical_pipeline(config)
        self.config = self.pipeline.config
        self.knowledge = KnowledgeService(self.pipeline)

    @classmethod
    def from_components(
        cls,
        *,
        embedder,
        dense_store,
        sparse_store,
        registry,
        chunker=None,
        reranker=None,
        parser_registry=None,
        config: RagConfig | None = None,
        lifecycle=None,
        recover: bool = True,
    ) -> "RagModule":
        pipeline = RetrievalPipeline(
            embedder,
            dense_store,
            chunker,
            reranker,
            config=config,
            sparse_store=sparse_store,
            registry=registry,
            parser_registry=parser_registry,
            lifecycle=lifecycle,
        )
        if recover:
            pipeline.recover()
        return cls(pipeline=pipeline)

    def retrieve(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        candidate_k: int | None = None,
        filters=None,
        rerank: bool = False,
        include_trace: bool = False,
    ) -> RetrievalResult:
        return self.pipeline.retrieve_result(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            mode=mode,
            filters=filters,
            rerank=rerank,
            include_trace=include_trace,
        )

    def runtime_snapshot(self) -> Mapping[str, Any]:
        return self.pipeline.runtime_snapshot()

    def close(self) -> None:
        self.pipeline.close()

    def __enter__(self) -> "RagModule":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = ["KnowledgeService", "RagModule"]
