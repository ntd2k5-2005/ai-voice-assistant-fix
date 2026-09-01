"""Knowledge inspection and lifecycle actions through ``KnowledgeService``."""

from __future__ import annotations

import inspect
import logging
from typing import Any

import streamlit as st

from rag import RagModule
from rag.streamlit_app.components.tab_ingestion import (
    execute_knowledge_operation,
    source_from_form,
)
from rag.streamlit_app.utils import (
    config_fingerprint,
    get_rag_module,
    json_value,
    operational_error_message,
)
from rag.streamlit_app.view_models import build_inspection_view


logger = logging.getLogger(__name__)


def inspect_knowledge(knowledge, document_id: str | None = None) -> Any:
    """Request active chunks when the public facade advertises that option."""

    parameters = inspect.signature(knowledge.inspect).parameters
    if "include_chunks" in parameters:
        return knowledge.inspect(document_id, include_chunks=True)
    return knowledge.inspect(document_id)


def _refresh(knowledge, state_key: str) -> None:
    st.session_state[state_key] = inspect_knowledge(knowledge)


def _render_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        st.info("No active chunks were returned for this document.")
        return
    st.caption(f"Active chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        chunk_id = chunk.get("chunk_id", chunk.get("_id", f"chunk-{index}"))
        page = chunk.get("page", metadata.get("page", "n/a"))
        section = chunk.get("section") or metadata.get("section") or metadata.get("heading") or "n/a"
        with st.expander(f"#{index} · {chunk_id} · page {page} · {section}", expanded=index == 0):
            st.write(chunk.get("text", ""))
            st.json(json_value({key: value for key, value in chunk.items() if key != "text"}))


def render_tab_inspection(module: RagModule | None = None) -> None:
    st.header("Knowledge inspection and lifecycle management")
    st.caption(
        "Inspect, update, reindex, delete, and recover through KnowledgeService only. "
        "The UI never opens or mutates Qdrant, FTS5, or the registry directly."
    )
    module = module or get_rag_module()
    knowledge = module.knowledge
    fingerprint = config_fingerprint(module.config)
    inspection_key = f"rag_inspection:{fingerprint}"
    ingestion_key = f"rag_last_ingestion:{fingerprint}"
    notice_key = f"rag_knowledge_notice:{fingerprint}"
    notice = st.session_state.pop(notice_key, None)
    if notice:
        st.success(str(notice))

    refresh_col, recover_col = st.columns(2)
    if refresh_col.button("Refresh active knowledge", key="refresh_inspection", width="stretch"):
        try:
            _refresh(knowledge, inspection_key)
        except Exception as exc:
            logger.exception("knowledge inspection failed")
            st.error(operational_error_message(exc))
    if recover_col.button("Recover interrupted operations", key="recover_knowledge", width="stretch"):
        try:
            recovered = knowledge.recover()
            _refresh(knowledge, inspection_key)
            st.session_state[notice_key] = (
                f"Recovery completed; affected documents/operations: {len(recovered)}."
            )
            st.rerun()
        except Exception as exc:
            logger.exception("knowledge recovery failed")
            st.error(operational_error_message(exc))

    payload = st.session_state.get(inspection_key)
    if payload is None:
        try:
            payload = inspect_knowledge(knowledge)
            st.session_state[inspection_key] = payload
        except Exception as exc:
            logger.exception("initial knowledge inspection failed")
            st.error(operational_error_message(exc))
            return

    view = build_inspection_view(payload)
    if view["empty"]:
        st.info("No knowledge documents are registered.")
        return

    st.subheader("Registered documents")
    st.dataframe(view["summary_rows"], width="stretch", hide_index=True)
    document_ids = [
        row["document_id"] for row in view["summary_rows"] if row["document_id"]
    ]
    if not document_ids:
        st.warning("Inspection returned records without a document_id.")
        with st.expander("Raw public inspection output"):
            st.json(json_value(payload))
        return

    selected = st.selectbox("Select a document", document_ids, key="managed_document_id")
    detail_payload = payload
    if selected not in view["chunks_by_document"]:
        try:
            detail_payload = inspect_knowledge(knowledge, selected)
            detail = build_inspection_view(detail_payload)
            detail_chunks = detail["chunks_by_document"].get(selected, [])
        except Exception as exc:
            logger.exception("document detail inspection failed")
            st.error(operational_error_message(exc))
            detail_chunks = []
    else:
        detail_chunks = view["chunks_by_document"][selected]

    detail_tab, update_tab, delete_tab = st.tabs(
        ["Active document and chunks", "Update / reindex", "Delete"]
    )
    with detail_tab:
        selected_rows = [row for row in view["summary_rows"] if row["document_id"] == selected]
        if selected_rows:
            st.json(json_value(selected_rows[0]))
        _render_chunks(detail_chunks)
        with st.expander("Raw typed inspection response"):
            st.json(json_value(detail_payload))

    with update_tab:
        st.caption("The selected document ID is preserved; content is sent to KnowledgeService.update/reindex.")
        source_kind = st.radio(
            "Update source",
            ("Text", "File"),
            horizontal=True,
            key="inspection_update_source_kind",
        )
        update_file = None
        update_text = ""
        if source_kind == "File":
            extensions = [value.removeprefix(".") for value in module.config.parser.allowed_extensions]
            update_file = st.file_uploader(
                "Replacement file",
                type=extensions,
                key="inspection_update_file",
            )
        else:
            update_text = st.text_area(
                "Replacement text",
                key="inspection_update_text",
                height=180,
            )
        update_col, reindex_col = st.columns(2)
        if update_col.button("Content-aware update", key="inspection_update", width="stretch"):
            try:
                source = source_from_form(source_kind, selected, update_file, update_text)
                result = execute_knowledge_operation(knowledge, "Update", source)
                st.session_state[ingestion_key] = result
                _refresh(knowledge, inspection_key)
                st.session_state[notice_key] = (
                    "Update operation completed through KnowledgeService."
                )
                st.rerun()
            except Exception as exc:
                logger.exception("knowledge update failed")
                st.error(operational_error_message(exc))
        if reindex_col.button("Force reindex", key="inspection_reindex", width="stretch"):
            try:
                source = source_from_form(source_kind, selected, update_file, update_text)
                result = execute_knowledge_operation(knowledge, "Force reindex", source)
                st.session_state[ingestion_key] = result
                _refresh(knowledge, inspection_key)
                st.session_state[notice_key] = (
                    "Reindex operation completed through KnowledgeService."
                )
                st.rerun()
            except Exception as exc:
                logger.exception("knowledge reindex failed")
                st.error(operational_error_message(exc))

    with delete_tab:
        st.warning(
            "Delete tombstones the selected document and removes its dense and sparse versions "
            "through the lifecycle coordinator."
        )
        confirmed = st.checkbox(
            f"I confirm deletion of {selected!r}.",
            key="confirm_delete",
        )
        if st.button(
            "Delete selected document",
            disabled=not confirmed,
            key="delete_document",
            type="primary",
        ):
            try:
                result = knowledge.delete(selected)
                _refresh(knowledge, inspection_key)
                st.session_state[notice_key] = (
                    f"Lifecycle delete completed for {result.get('document_id', selected)!r}."
                )
                st.rerun()
            except Exception as exc:
                logger.exception("knowledge delete failed")
                st.error(operational_error_message(exc))


__all__ = ["inspect_knowledge", "render_tab_inspection"]
