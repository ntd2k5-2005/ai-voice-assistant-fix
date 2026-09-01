"""Flow 1 controls and stage-by-stage inspection."""

from __future__ import annotations

import inspect
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from rag import DocumentSource, RagModule
from rag.streamlit_app.utils import (
    config_fingerprint,
    get_rag_module,
    json_value,
    operational_error_message,
)
from rag.streamlit_app.view_models import build_ingestion_view


logger = logging.getLogger(__name__)


_SAMPLE_TEXT = """# RAG service overview

The service stores dense vectors in Qdrant and lexical entries in SQLite FTS5.
An indexed document becomes visible only after both stores are verified and the
document registry atomically commits the new version.
"""


def source_from_form(
    source_kind: str,
    document_id: str,
    uploaded_file,
    raw_text: str,
) -> DocumentSource:
    """Create the public source contract without parsing anything in the UI."""

    identity = document_id.strip()
    if source_kind == "File":
        if uploaded_file is None:
            raise ValueError("Hãy chọn một file trước khi tạo tri thức.")
        if not identity:
            identity = Path(uploaded_file.name).stem.strip()
    elif not identity:
        raise ValueError("Document ID không được để trống khi nhập text trực tiếp.")
    metadata = {"document_id": identity, "ui_source": "streamlit"}
    if source_kind == "File":
        return DocumentSource(
            value=uploaded_file.getvalue(),
            source_type="bytes",
            metadata=metadata,
            mime_type=uploaded_file.type or None,
            file_name=uploaded_file.name,
        )
    if not raw_text.strip():
        raise ValueError("Text content must not be empty.")
    return DocumentSource(
        value=raw_text,
        source_type="text",
        metadata=metadata,
        file_name=f"{identity}.md",
        mime_type="text/markdown",
    )


def execute_knowledge_operation(knowledge, operation: str, source: DocumentSource):
    """Execute a public knowledge action with an opt-in trace when supported."""

    include_trace = "include_trace" in inspect.signature(knowledge.ingest).parameters
    kwargs = {"include_trace": True} if include_trace else {}
    if operation == "Add / ingest":
        return knowledge.ingest(source, **kwargs)
    if operation == "Update":
        return knowledge.update(source, **kwargs)
    if operation == "Force reindex":
        return knowledge.reindex(source, **kwargs)
    raise ValueError(f"unknown knowledge operation: {operation!r}")


def _stage_unavailable(label: str) -> None:
    st.warning(
        f"{label} trace is UNAVAILABLE from the public operation result. "
        "The UI will not reconstruct this stage with a duplicate pipeline."
    )


def _render_unit(unit: Mapping[str, Any], index: int, *, normalized: bool) -> None:
    metadata = unit.get("metadata") if isinstance(unit.get("metadata"), Mapping) else {}
    provenance = unit.get("provenance") if isinstance(unit.get("provenance"), Mapping) else {}
    unit_type = unit.get("unit_type") or metadata.get("unit_type") or "text"
    source_position = (
        unit.get("page")
        or metadata.get("page")
        or provenance.get("page")
        or unit.get("sheet")
        or metadata.get("sheet")
        or provenance.get("sheet")
        or unit.get("slide")
        or metadata.get("slide")
        or provenance.get("slide")
        or "n/a"
    )
    label = f"Unit {index + 1} · {unit_type} · position {source_position}"
    with st.expander(label, expanded=index == 0):
        text = unit.get("text", unit.get("content", ""))
        st.text(str(text))
        st.caption("Provenance and structure")
        st.json(
            json_value(
                {
                    key: value
                    for key, value in unit.items()
                    if key not in {"text", "content"}
                }
            )
        )
        if normalized:
            st.caption("This is the normalized structured unit consumed by chunking.")


def _render_units(units: list[dict[str, Any]], *, key: str, normalized: bool) -> None:
    if not units:
        _stage_unavailable("Normalized output" if normalized else "Parsed document")
        return
    page_size = int(
        st.selectbox(
            "Units per page",
            (5, 10, 25, 50),
            index=1,
            key=f"{key}_unit_page_size",
        )
    )
    page_count = max(1, math.ceil(len(units) / page_size))
    page = int(
        st.number_input(
            "Unit page",
            min_value=1,
            max_value=page_count,
            value=1,
            key=f"{key}_unit_page",
        )
    )
    start = (page - 1) * page_size
    st.caption(f"Showing units {start + 1}–{min(start + page_size, len(units))} of {len(units)}")
    for offset, unit in enumerate(units[start : start + page_size], start):
        _render_unit(unit, offset, normalized=normalized)


def _render_chunks(view: Mapping[str, Any], *, key: str, overlap_tokens: int) -> None:
    chunks = list(view["chunks"])
    stats = dict(view["chunk_statistics"])
    metrics = (
        ("Chunks", stats.get("chunk_count")),
        ("Minimum tokens", stats.get("min_tokens")),
        ("Maximum tokens", stats.get("max_tokens")),
        ("Mean tokens", stats.get("mean_tokens")),
        ("Median tokens", stats.get("median_tokens")),
        ("Configured overlap", overlap_tokens),
    )
    for column, (label, value) in zip(st.columns(6), metrics, strict=True):
        if isinstance(value, float):
            value = f"{value:.1f}"
        column.metric(label, "N/A" if value is None else value)
    if not chunks:
        _stage_unavailable("Chunk")
        return
    if not stats.get("token_counts_available"):
        st.warning("Some chunk token counts were not reported; statistics are partial.")
    token_rows = [
        {"chunk": chunk.get("index", index), "tokens": chunk.get("token_count")}
        for index, chunk in enumerate(chunks)
        if chunk.get("token_count") is not None
    ]
    if token_rows:
        st.bar_chart(token_rows, x="chunk", y="tokens")

    page_size = int(
        st.selectbox(
            "Chunks per page",
            (5, 10, 25, 50),
            index=1,
            key=f"{key}_chunk_page_size",
        )
    )
    page_count = max(1, math.ceil(len(chunks) / page_size))
    page = int(
        st.number_input(
            "Chunk page",
            min_value=1,
            max_value=page_count,
            value=1,
            key=f"{key}_chunk_page",
        )
    )
    start = (page - 1) * page_size
    for index, chunk in enumerate(chunks[start : start + page_size], start):
        chunk_id = chunk.get("chunk_id", chunk.get("_id", f"chunk-{index}"))
        tokens = chunk.get("token_count", "unknown")
        with st.expander(f"#{index} · {chunk_id} · {tokens} tokens", expanded=index == 0):
            st.write(chunk.get("text", ""))
            st.json(
                json_value(
                    {
                        key: value
                        for key, value in chunk.items()
                        if key not in {"text", "vector", "embedding"}
                    }
                )
            )


def _render_embedding(data: Mapping[str, Any]) -> None:
    if not data:
        _stage_unavailable("Embedding")
        return
    fields = (
        ("Model", data.get("model")),
        ("Dimension", data.get("dimension")),
        ("Embedded chunks", data.get("chunk_count")),
        ("Batch size", data.get("batch_size", data.get("resolved_batch_size"))),
        ("Elapsed", data.get("elapsed_ms")),
        ("Finite", data.get("finite")),
    )
    for column, (label, value) in zip(st.columns(6), fields, strict=True):
        if label == "Elapsed" and value is not None:
            value = f"{float(value):.2f} ms"
        column.metric(label, "N/A" if value is None else value)
    st.json(json_value({key: value for key, value in data.items() if key != "vector_previews"}))
    vector_norm = data.get("vector_norm", data.get("vector_norms"))
    if vector_norm is not None:
        st.caption("Vector norm statistics")
        st.json(json_value(vector_norm))
    previews = data.get("vector_previews")
    if isinstance(previews, list) and previews:
        with st.expander("Short vector previews"):
            st.json(json_value(previews))


def _render_sparse(data: Mapping[str, Any]) -> None:
    if not data:
        _stage_unavailable("Sparse indexing")
        return
    columns = st.columns(5)
    columns[0].metric("Indexed chunks", data.get("chunk_count", "N/A"))
    columns[1].metric("Analyzer", data.get("analyzer", "N/A"))
    columns[2].metric("Status", data.get("status", "N/A"))
    elapsed = data.get("elapsed_ms")
    columns[3].metric(
        "Reported elapsed",
        "N/A" if elapsed is None else f"{float(elapsed):.2f} ms",
    )
    columns[4].metric("Timing scope", data.get("timing_scope", "sparse stage"))
    st.json(json_value(data))


def _render_lifecycle(data: Mapping[str, Any]) -> None:
    if not data:
        _stage_unavailable("Lifecycle")
        return
    fields = (
        ("Document ID", data.get("document_id")),
        ("Version", data.get("version")),
        ("Previous version", data.get("previous_version")),
        ("Operation ID", data.get("operation_id")),
        ("Verification", data.get("verification_status", data.get("verified"))),
        ("Active", data.get("active")),
    )
    for column, (label, value) in zip(st.columns(6), fields, strict=True):
        column.metric(label, "N/A" if value is None else value)
    transitions = data.get("transitions")
    if isinstance(transitions, list) and transitions:
        st.caption("State transitions")
        st.dataframe(transitions, width="stretch", hide_index=True)
    st.json(json_value(data))


def _render_document_flow(view: Mapping[str, Any], *, config, index: int) -> None:
    summary = view["summary"]
    document_id = summary.get("document_id", f"document-{index}")
    st.subheader(f"Document: {document_id}")
    st.caption(
        f"Action={summary.get('action')} · version={summary.get('version')} · "
        f"chunks={summary.get('chunk_count')}"
    )
    stages = st.tabs(
        [
            "1. Source",
            "2. Parsed",
            "3. Normalized",
            "4. Chunks",
            "5. Embedding",
            "6. Sparse index",
            "7. Activation",
        ]
    )
    prefix = f"ingestion_{document_id}_{index}"
    with stages[0]:
        source = view["source"]
        sources = view.get("sources") or ([source] if source else [])
        if sources:
            for source_index, source_item in enumerate(sources):
                if len(sources) > 1:
                    st.caption(f"Uploaded source {source_index + 1} of {len(sources)}")
                st.json(json_value(source_item))
                raw_preview = source_item.get("raw_preview")
                if raw_preview:
                    st.text_area(
                        "Raw preview",
                        str(raw_preview),
                        disabled=True,
                        key=f"{prefix}_raw_{source_index}",
                    )
        else:
            _stage_unavailable("Source")
    with stages[1]:
        _render_units(view["parsed_units"], key=f"{prefix}_parsed", normalized=False)
    with stages[2]:
        _render_units(view["normalized_units"], key=f"{prefix}_normalized", normalized=True)
    with stages[3]:
        _render_chunks(
            view,
            key=prefix,
            overlap_tokens=config.chunking.overlap_tokens,
        )
    with stages[4]:
        _render_embedding(view["embedding"])
    with stages[5]:
        _render_sparse(view["sparse"])
    with stages[6]:
        _render_lifecycle(view["lifecycle"])
    with st.expander("Complete typed ingestion result and diagnostics"):
        st.json(json_value(summary))


def render_tab_ingestion(module: RagModule | None = None) -> None:
    st.header("Flow 1 — Tạo tri thức")
    st.write(
        "**1. Chọn dữ liệu → 2. Parser đọc cấu trúc → 3. Normalize → 4. Chunk → "
        "5. Embed + sparse index → 6. Verify → 7. Activate knowledge.**"
    )
    module = module or get_rag_module()
    result_key = f"rag_last_ingestion:{config_fingerprint(module.config)}"

    source_kind = st.radio(
        "Nguồn dữ liệu",
        ("File", "Text"),
        horizontal=True,
        key="source_kind",
        help="File là chế độ demo chính. Text dùng để smoke-test nhanh.",
    )

    uploaded_file = None
    raw_text = ""
    if source_kind == "File":
        extensions = [value.removeprefix(".") for value in module.config.parser.allowed_extensions]
        uploaded_file = st.file_uploader(
            "📤 Upload tài liệu",
            type=extensions,
            help="File sẽ đi qua parser/chunker/embedding/index thật của RagModule.",
        )
        if uploaded_file is not None:
            st.success(
                f"Đã chọn: {uploaded_file.name} · {uploaded_file.type or 'unknown MIME'} · "
                f"{uploaded_file.size:,} bytes"
            )
    else:
        raw_text = st.text_area(
            "Nội dung text",
            value="",
            height=180,
            placeholder=_SAMPLE_TEXT,
        )
        if raw_text:
            st.caption(f"UTF-8 size: {len(raw_text.encode('utf-8')):,} bytes")

    id_col, operation_col = st.columns([2, 1])
    with id_col:
        document_id = st.text_input(
            "Document ID (để trống sẽ lấy tên file)",
            value="",
            key="ingest_document_id",
        )
    with operation_col:
        operation = st.selectbox(
            "Thao tác",
            ("Add / ingest", "Update", "Force reindex"),
            key="ingestion_operation",
        )

    if st.button("🚀 Tạo / cập nhật tri thức", type="primary", key="run_ingestion", width="stretch"):
        try:
            source = source_from_form(source_kind, document_id, uploaded_file, raw_text)
            with st.spinner("Executing the public knowledge lifecycle..."):
                result = execute_knowledge_operation(module.knowledge, operation, source)
            st.session_state[result_key] = result
            # Knowledge management is cached only for display. Invalidate its
            # snapshot so the inspection tab reloads the newly committed
            # version through KnowledgeService during this same rerun.
            st.session_state.pop(
                f"rag_inspection:{config_fingerprint(module.config)}",
                None,
            )
            # Refresh runtime_snapshot as well, so global active counts and the
            # lifecycle result cannot disagree after a successful commit.
            st.rerun()
        except Exception as exc:
            logger.exception("knowledge flow failed")
            st.error("Knowledge ingestion did not complete.")
            st.code(operational_error_message(exc), language=None)

    result = st.session_state.get(result_key)
    if result is None:
        st.info("Upload một tài liệu và bấm **Tạo / cập nhật tri thức** để chạy pipeline thật.")
        return

    views = build_ingestion_view(result)
    if not views:
        st.warning("The public operation returned no document result.")
        return
    total_chunks = sum(int(view["summary"].get("chunk_count") or 0) for view in views)
    st.success(
        f"✅ Knowledge đã được xử lý: {len(views)} document · {total_chunks} chunk."
    )
    summary_cols = st.columns(3)
    summary_cols[0].metric("Documents", len(views))
    summary_cols[1].metric("Chunks", total_chunks)
    summary_cols[2].metric("Embedding", module.config.embedding.model_name.split("/")[-1])
    st.caption(
        "Kết quả phía dưới là trace của chính lần ingest vừa chạy, không phải dữ liệu mô phỏng."
    )
    with st.expander("🔬 Xem chi tiết từng bước của Flow 1", expanded=True):
        for index, view in enumerate(views):
            _render_document_flow(view, config=module.config, index=index)


__all__ = ["execute_knowledge_operation", "render_tab_ingestion", "source_from_form"]
