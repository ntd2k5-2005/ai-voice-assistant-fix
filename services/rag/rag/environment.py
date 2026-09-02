"""Environment adapter for the canonical immutable RAG configuration."""

from __future__ import annotations

from dataclasses import replace
import os
from typing import Mapping

from .config import RagConfig


def _boolean(name: str, raw: str) -> bool:
    value = raw.strip().casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def config_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    base: RagConfig | None = None,
) -> RagConfig:
    """Apply only explicit model/runtime overrides used by all service adapters."""

    env = os.environ if environ is None else environ
    config = base or RagConfig()

    embedding_updates = {}
    for environment, field_name in (
        ("RAG_EMBEDDING_MODEL", "model_name"),
        ("RAG_EMBEDDING_REVISION", "revision"),
        ("RAG_EMBEDDING_DEVICE", "device"),
        ("RAG_EMBEDDING_DTYPE", "dtype"),
    ):
        raw = env.get(environment, "").strip()
        if raw:
            embedding_updates[field_name] = raw
    raw = env.get("RAG_EMBEDDING_LOCAL_FILES_ONLY", "").strip()
    if raw:
        embedding_updates["local_files_only"] = _boolean(
            "RAG_EMBEDDING_LOCAL_FILES_ONLY", raw
        )
    if embedding_updates:
        config = replace(config, embedding=replace(config.embedding, **embedding_updates))

    reranker_updates = {}
    if "RAG_RERANKER_MODEL" in env:
        raw = env["RAG_RERANKER_MODEL"].strip()
        reranker_updates["model_name"] = (
            None if raw.casefold() in {"", "none", "disabled", "off"} else raw
        )
    for environment, field_name in (
        ("RAG_RERANKER_REVISION", "revision"),
        ("RAG_RERANKER_CODE_REVISION", "code_revision"),
        ("RAG_RERANKER_DEVICE", "device"),
        ("RAG_RERANKER_DTYPE", "dtype"),
    ):
        raw = env.get(environment, "").strip()
        if raw:
            reranker_updates[field_name] = raw
    raw = env.get("RAG_RERANKER_LOCAL_FILES_ONLY", "").strip()
    if raw:
        reranker_updates["local_files_only"] = _boolean(
            "RAG_RERANKER_LOCAL_FILES_ONLY", raw
        )
    if reranker_updates:
        config = replace(config, reranker=replace(config.reranker, **reranker_updates))

    return config


__all__ = ["config_from_environment"]
