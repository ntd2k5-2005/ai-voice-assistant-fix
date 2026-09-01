"""Flow 2 controls and stage-by-stage typed retrieval visualization."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Mapping

import streamlit as st

from rag import MetadataFilter, RagModule
from rag.streamlit_app.utils import (
    config_fingerprint,
    get_rag_module,
    json_value,
    operational_error_message,
)
from rag.streamlit_app.view_models import build_retrieval_view


logger = logging.getLogger(__name__)


def parse_filters(raw: str) -> list[MetadataFilter] | None:
    """Parse an explicit JSON filter list into public filter contracts."""

    if not raw.strip():
        return None
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("Metadata filters must be a JSON list.")
    filters: list[MetadataFilter] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Metadata filter #{index + 1} must be an object.")
        try:
            filters.append(
                MetadataFilter(
                    field=item["field"],
                    operator=item["operator"],
                    value=item.get("value"),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Metadata filter #{index + 1} requires field, operator, and value."
            ) from exc
    return filters


def retrieve_with_trace(module: RagModule, query: str, **kwargs):
    """Call the public retrieval facade and request its bounded trace if exposed."""

    parameters = inspect.signature(module.retrieve).parameters
    if "include_trace" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs["include_trace"] = True
    return module.retrieve(query, **kwargs)


def _score(value: Any) -> float | None:
    return None if value is None else round(float(value), 6)


def _candidate_table(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": candidate.get("rank"),
            "chunk_id": candidate.get("chunk_id"),
            "document_id": candidate.get("document_id"),
            "source": candidate.get("source"),
            "page": candidate.get("page"),
            "section": candidate.get("section"),
            "text preview": str(candidate.get("text") or "")[:240],
            "dense": _score(candidate.get("dense_score")),
            "sparse": _score(candidate.get("sparse_score")),
            "fusion": _score(candidate.get("fusion_score")),
            "rerank": _score(candidate.get("rerank_score")),
            "final": _score(candidate.get("final_score")),
        "final source": candidate.get("final_score_source"),
        }
        for candidate in candidates
    ]


def _render_candidate_stage(
    title: str,
    stage: Mapping[str, Any],
    *,
    not_executed_reason: str,
) -> None:
    st.subheader(title)
    if not stage.get("executed"):
        st.info(f"{title}: NOT EXECUTED. {not_executed_reason}")
        return
    if not stage.get("available"):
        st.warning(
            f"{title} was executed, but its candidate trace is UNAVAILABLE. "
            "Final candidates are not relabeled as branch candidates."
        )
        return
    candidates = list(stage.get("candidates") or [])
    if not candidates:
        st.info(f"{title} executed and returned zero candidates.")
        return
    st.caption(f"Candidates returned by this branch: {len(candidates)}")
    st.dataframe(_candidate_table(candidates), width="stretch", hide_index=True)


def _render_query_embedding(data: Mapping[str, Any], mode: str) -> None:
    st.subheader("Query embedding")
    if mode == "sparse" or data.get("executed") is False:
        st.info(
            "Query embedding: NOT EXECUTED. "
            + str(data.get("reason") or "Sparse retrieval mode does not use dense embeddings.")
        )
        return
    if not data:
        st.warning("Query embedding executed, but its public trace is UNAVAILABLE.")
        return
    metrics = (
        ("Model", data.get("model")),
        ("Dimension", data.get("dimension")),
        ("Latency", data.get("elapsed_ms")),
        ("Vector norm", data.get("vector_norm", data.get("norm"))),
        ("Finite", data.get("finite")),
    )
    for column, (label, value) in zip(st.columns(5), metrics, strict=True):
        if label == "Latency" and value is not None:
            value = f"{float(value):.2f} ms"
        column.metric(label, "N/A" if value is None else value)
    preview = data.get("vector_preview")
    if preview is not None:
        with st.expander("Small query-vector preview"):
            st.json(json_value(preview))


def _render_fusion(data: Mapping[str, Any]) -> None:
    st.subheader("Fusion")
    if not data.get("executed"):
        st.info(f"Fusion: NOT EXECUTED. {data.get('reason', '')}")
        return
    st.write("Strategy:", data.get("strategy", "UNAVAILABLE"))
    parameters = data.get("parameters")
    if parameters:
        st.json(json_value(parameters))
    candidates = list(data.get("candidates") or [])
    if candidates:
        rows = [
            {
                "fusion rank": item.get("fusion_rank") or item.get("rank"),
                "chunk_id": item.get("chunk_id"),
                "dense rank": item.get("dense_rank"),
                "dense score": _score(item.get("dense_score")),
                "sparse rank": item.get("sparse_rank"),
                "sparse score": _score(item.get("sparse_score")),
                "fusion score": _score(item.get("fusion_score")),
                "source": item.get("source"),
            }
            for item in candidates
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("Fusion executed and returned zero candidates, or the candidate trace is unavailable.")


def _render_rerank(data: Mapping[str, Any], model_name: str | None) -> None:
    st.subheader("Reranking")
    if not data.get("executed") and not data.get("applied") and not data.get("fallback"):
        st.info(f"Reranker: NOT EXECUTED. {data.get('reason', '')}")
        return
    st.write("Configured model:", data.get("model") or model_name or "UNAVAILABLE")
    metrics = st.columns(4)
    metrics[0].metric("Applied", data.get("applied", False))
    metrics[1].metric("Fallback", data.get("fallback", False))
    elapsed = data.get("elapsed_ms")
    metrics[2].metric("Latency", "N/A" if elapsed is None else f"{float(elapsed):.2f} ms")
    metrics[3].metric("Error", data.get("error") or "none")

    before = list(data.get("before") or [])
    after = list(data.get("after") or [])
    if not before and not after:
        st.warning("Reranking status is available, but before/after candidate order is UNAVAILABLE.")
        return
    before_by_id = {item.get("chunk_id"): item for item in before}
    after_by_id = {item.get("chunk_id"): item for item in after}
    ordered_ids = [item.get("chunk_id") for item in after] + [
        item.get("chunk_id") for item in before if item.get("chunk_id") not in after_by_id
    ]
    rows = []
    for chunk_id in ordered_ids:
        pre = before_by_id.get(chunk_id, {})
        post = after_by_id.get(chunk_id, {})
        rows.append(
            {
                "pre-rerank rank": pre.get("pre_rerank_rank") or pre.get("rank"),
                "post-rerank rank": post.get("post_rerank_rank") or post.get("rank"),
                "chunk_id": chunk_id,
                "retrieval/fusion score": _score(
                    pre.get("fusion_score", pre.get("retrieval_score"))
                ),
                "rerank score": _score(post.get("rerank_score")),
                "source": post.get("source") or pre.get("source"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_final(candidates: list[dict[str, Any]]) -> None:
    st.subheader("Final Top-K contexts")
    st.caption(
        "Tất cả score hiển thị ở đây nằm trong [0, 1] và chỉ biểu diễn mức độ liên quan; "
        "không phải xác suất đúng. Raw backend/model score chỉ nằm trong phần kỹ thuật."
    )
    if not candidates:
        st.info("Public API returned zero final contexts.")
        return
    st.dataframe(_candidate_table(candidates), width="stretch", hide_index=True)
    for candidate in candidates:
        label = (
            f"#{candidate.get('rank') or '?'} · "
            f"score={_score(candidate.get('final_score'))} · "
            f"{candidate.get('section') or candidate.get('source') or 'context'}"
        )
        with st.expander(label, expanded=candidate.get("rank") == 1):
            st.markdown("**Context trả về cho downstream LLM/API**")
            st.write(candidate.get("text", ""))

            score_columns = st.columns(5)
            score_columns[0].metric("Final", "N/A" if candidate.get("final_score") is None else f"{float(candidate['final_score']):.4f}")
            score_columns[1].metric("Dense", "N/A" if candidate.get("dense_score") is None else f"{float(candidate['dense_score']):.4f}")
            score_columns[2].metric("Sparse", "N/A" if candidate.get("sparse_score") is None else f"{float(candidate['sparse_score']):.4f}")
            score_columns[3].metric("Fusion", "N/A" if candidate.get("fusion_score") is None else f"{float(candidate['fusion_score']):.4f}")
            score_columns[4].metric("Rerank", "N/A" if candidate.get("rerank_score") is None else f"{float(candidate['rerank_score']):.4f}")
            st.caption(f"Final score source: {candidate.get('final_score_source') or 'unknown'}")

            provenance = {
                "document_id": candidate.get("document_id"),
                "document_version": candidate.get("document_version"),
                "source": candidate.get("source"),
                "page": candidate.get("page"),
                "section": candidate.get("section"),
                "sheet": candidate.get("sheet"),
                "row": candidate.get("row"),
                "slide": candidate.get("slide"),
                "chunk_id": candidate.get("chunk_id"),
            }
            st.markdown("**Provenance**")
            st.json(json_value({key: value for key, value in provenance.items() if value is not None}))

            with st.expander("🔬 Metadata / raw scores kỹ thuật", expanded=False):
                st.json(
                    json_value(
                        {
                            "raw_scores": {
                                "dense": candidate.get("dense_raw_score"),
                                "sparse": candidate.get("sparse_raw_score"),
                                "fusion": candidate.get("fusion_raw_score"),
                                "rerank": candidate.get("rerank_raw_score"),
                                "retrieval": candidate.get("retrieval_raw_score"),
                            },
                            "metadata": candidate.get("metadata") or {},
                        }
                    )
                )


def _render_latency(view: Mapping[str, Any]) -> None:
    st.subheader("Latency and diagnostics")
    latency = dict(view.get("latency") or {})
    fields = (
        ("Query preparation", latency.get("query_preparation_ms")),
        ("Embedding", latency.get("query_embedding_ms", latency.get("query_embedding"))),
        ("Dense search", latency.get("dense_search_ms", latency.get("dense_search"))),
        ("Sparse search", latency.get("sparse_search_ms", latency.get("sparse_search"))),
        ("Fusion", latency.get("fusion_ms", latency.get("fusion"))),
        ("Reranking", latency.get("rerank_ms", latency.get("rerank"))),
        ("Total", latency.get("total_ms", latency.get("total"))),
    )
    for column, (label, value) in zip(st.columns(7), fields, strict=True):
        column.metric(label, "N/A" if value is None else f"{float(value):.2f} ms")
    for warning in view.get("warnings") or []:
        st.warning(str(warning))
    st.json(json_value(view.get("diagnostics") or {}))


def render_tab_retrieval(module: RagModule | None = None) -> None:
    st.header("Flow 2 — Truy xuất tri thức")
    st.write(
        "Nhập câu hỏi để xem **retrieval thật**: query → dense/sparse/hybrid → fusion → "
        "optional rerank → Top-K context. Module này chưa sinh câu trả lời LLM."
    )
    module = module or get_rag_module()
    config = module.config
    result_key = f"rag_last_retrieval:{config_fingerprint(config)}"

    query = st.text_input(
        "🔎 Câu hỏi",
        value="",
        placeholder="Nhập câu hỏi liên quan đến tài liệu đã ingest...",
        key="retrieval_query",
    )
    mode_col, rerank_col = st.columns(2)
    configured_mode = getattr(config.retrieval, "default_mode", "hybrid")
    with mode_col:
        mode = st.selectbox(
            "Retrieval mode",
            ("dense", "sparse", "hybrid"),
            index=("dense", "sparse", "hybrid").index(configured_mode),
            key="retrieval_mode",
        )
    with rerank_col:
        rerank = st.checkbox(
            "Dùng reranker",
            value=config.reranker.model_name is not None,
            key="retrieval_rerank",
            help=config.reranker.model_name or "No reranker configured",
        )

    top_k = int(config.retrieval.top_k)
    candidate_k = int(config.retrieval.candidate_k)
    filter_json = ""
    with st.expander("⚙️ Tùy chọn retrieval nâng cao", expanded=False):
        limit_col, candidate_col = st.columns(2)
        with limit_col:
            top_k = int(
                st.number_input(
                    "top_k",
                    min_value=1,
                    max_value=100,
                    value=config.retrieval.top_k,
                    key="retrieval_top_k",
                )
            )
        with candidate_col:
            candidate_k = int(
                st.number_input(
                    "candidate_k",
                    min_value=1,
                    max_value=500,
                    value=config.retrieval.candidate_k,
                    key="retrieval_candidate_k",
                )
            )
        st.caption(
            f"Fusion: {config.fusion.strategy} · dense_weight={config.fusion.dense_weight} · "
            f"sparse_weight={config.fusion.sparse_weight} · rrf_k={config.fusion.rrf_k}"
        )
        filter_json = st.text_area(
            "Metadata filters (JSON, optional)",
            value="",
            key="retrieval_filters",
            placeholder='[{"field": "language", "operator": "=", "value": "vi"}]',
        )

    if st.button("🔎 Chạy retrieval", type="primary", key="run_retrieval", width="stretch"):
        try:
            if mode is None:
                raise ValueError("Choose a retrieval mode explicitly.")
            if candidate_k < top_k:
                raise ValueError("candidate_k must be greater than or equal to top_k.")
            filters = parse_filters(filter_json)
            result = retrieve_with_trace(
                module,
                query,
                mode=mode,
                top_k=top_k,
                candidate_k=candidate_k,
                filters=filters,
                rerank=rerank,
            )
            st.session_state[result_key] = (query, result)
        except Exception as exc:
            logger.exception("retrieval flow failed")
            st.error("Retrieval did not complete.")
            st.code(operational_error_message(exc), language=None)

    state = st.session_state.get(result_key)
    if state is None:
        st.info("Hãy nhập câu hỏi và bấm **Chạy retrieval**. Kết quả sẽ lấy từ knowledge đã index thật.")
        return
    original_query, result = state
    view = build_retrieval_view(result, original_query=original_query)

    st.success(
        f"✅ Retrieval hoàn tất · mode={view['query'].get('mode')} · "
        f"Top-K={len(view['final'])}."
    )
    _render_final(view["final"])

    with st.expander("🔬 Xem pipeline retrieval chi tiết", expanded=False):
        st.subheader("Query preparation")
        st.json(json_value(view["query"]))
        _render_query_embedding(view["query_embedding"], str(view["query"].get("mode")))
        branch_columns = st.columns(2)
        with branch_columns[0]:
            _render_candidate_stage(
                "Dense candidates",
                view["dense"],
                not_executed_reason="The selected mode does not use dense retrieval.",
            )
        with branch_columns[1]:
            _render_candidate_stage(
                "Sparse candidates",
                view["sparse"],
                not_executed_reason="The selected mode does not use sparse retrieval.",
            )
        _render_fusion(view["fusion"])
        _render_rerank(view["rerank"], config.reranker.model_name)
        _render_latency(view)
        with st.expander("Exact public RetrievalResult"):
            st.json(json_value(result))


__all__ = ["parse_filters", "render_tab_retrieval", "retrieve_with_trace"]
