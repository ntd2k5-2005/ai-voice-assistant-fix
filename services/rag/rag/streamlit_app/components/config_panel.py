"""Configured-versus-resolved runtime panel."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import streamlit as st

from rag.config import RagConfig
from rag.streamlit_app.utils import ConfigOverride, json_value
from rag.streamlit_app.view_models import build_config_view


def _cell(value: Any) -> Any:
    value = json_value(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return "null"
    return str(value)


def render_config_panel(
    config: RagConfig,
    runtime_snapshot: Mapping[str, Any],
    *,
    overrides: Sequence[ConfigOverride] = (),
    compact: bool = False,
) -> None:
    """Render all authoritative config sections and runtime resolutions."""

    st.subheader("Effective runtime configuration")
    st.caption(
        "Configured values come from RagConfig. Resolved values are reported by "
        "the running public facade; UNAVAILABLE explicitly means no runtime value was reported."
    )
    if overrides:
        st.info("Explicit environment overrides are active.")
        st.dataframe(
            [
                {
                    "environment": item.environment,
                    "setting": item.setting,
                    "value": _cell(item.value),
                }
                for item in overrides
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("Overrides: none. The UI is using RagConfig values unchanged.")

    sections = build_config_view(config, runtime_snapshot)
    for section, rows in sections.items():
        with st.expander(section, expanded=not compact and section in {"Chunking", "Embedding"}):
            st.dataframe(
                [
                    {
                        **row,
                        "configured": _cell(row["configured"]),
                        "resolved": _cell(row["resolved"]),
                    }
                    for row in rows
                ],
                width="stretch",
                hide_index=True,
            )


__all__ = ["render_config_panel"]
