"""Practical Streamlit demo for the two public RAG flows."""

from __future__ import annotations

import logging

import streamlit as st

from rag.streamlit_app.components.config_panel import render_config_panel
from rag.streamlit_app.components.tab_ingestion import render_tab_ingestion
from rag.streamlit_app.components.tab_inspection import render_tab_inspection
from rag.streamlit_app.components.tab_retrieval import render_tab_retrieval
from rag.streamlit_app.utils import (
    clear_rag_module,
    config_fingerprint,
    get_rag_module,
    operational_error_message,
    public_runtime_snapshot,
    resolve_config,
)


logger = logging.getLogger(__name__)


def _render_sidebar(module, runtime_snapshot, overrides) -> None:
    with st.sidebar:
        st.title("RAG Module")
        st.caption("Demo thật của 2 luồng: tạo tri thức và truy xuất tri thức.")
        st.write("**Embedding**")
        st.code(module.config.embedding.model_name, language=None)
        st.write("**Reranker**")
        st.code(module.config.reranker.model_name or "disabled", language=None)
        st.caption(
            "Config fingerprint: "
            + runtime_snapshot.get("config_fingerprint", config_fingerprint(module.config))[:16]
            + "…"
        )
        with st.expander("Cấu hình hệ thống (nâng cao)", expanded=False):
            render_config_panel(
                module.config,
                runtime_snapshot,
                overrides=overrides,
                compact=True,
            )
        if st.button("Reload RAG runtime", width="stretch"):
            clear_rag_module(module)
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="RAG Module Demo",
        page_icon="🧠",
        layout="wide",
    )
    st.title("🧠 RAG Module Demo")
    st.caption(
        "Upload tài liệu → kiểm tra pipeline tạo tri thức → truy vấn → xem chính xác các context được lấy ra."
    )

    try:
        effective_config, overrides = resolve_config()
        module = get_rag_module(effective_config)
        runtime_snapshot = public_runtime_snapshot(module)
    except Exception as exc:
        logger.exception("unable to initialize the configured RAG runtime")
        st.error("Không thể khởi tạo RAG runtime với cấu hình hiện tại.")
        st.code(operational_error_message(exc), language=None)
        st.info(
            "Kiểm tra model cache/network, storage path và các biến môi trường RAG_* nếu có."
        )
        return

    _render_sidebar(module, runtime_snapshot, overrides)

    ingestion, retrieval, inspection = st.tabs(
        [
            "1️⃣ Tạo tri thức",
            "2️⃣ Truy xuất tri thức",
            "3️⃣ Quản lý tri thức",
        ]
    )
    with ingestion:
        render_tab_ingestion(module)
    with retrieval:
        render_tab_retrieval(module)
    with inspection:
        render_tab_inspection(module)


if __name__ == "__main__":
    main()


__all__ = ["main"]
