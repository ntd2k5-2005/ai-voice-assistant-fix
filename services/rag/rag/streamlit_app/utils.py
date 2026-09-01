"""Shared helpers for the Streamlit inspection console.

The UI deliberately owns no RAG defaults and no retrieval implementation. It
starts with :class:`rag.config.RagConfig` and applies only explicit environment
overrides by replacing fields on that immutable configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from rag import RagModule
from rag.config import PathsConfig, RagConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigOverride:
    """One explicit UI environment override shown to the operator."""

    environment: str
    setting: str
    value: Any


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false, yes/no, 1/0, or on/off")


def resolve_config(
    environ: Mapping[str, str] | None = None,
    *,
    base: RagConfig | None = None,
) -> tuple[RagConfig, tuple[ConfigOverride, ...]]:
    """Resolve central defaults plus explicit environment overrides.

    No fallback value is re-declared here. An absent environment variable
    means exactly "use the value from ``RagConfig``".
    """

    env = os.environ if environ is None else environ
    config = base or RagConfig()
    overrides: list[ConfigOverride] = []

    runtime_dir = env.get("RAG_STREAMLIT_DATA_DIR", "").strip()
    if runtime_dir:
        root = Path(runtime_dir).expanduser().resolve(strict=False)
        paths = PathsConfig(
            qdrant=root / "qdrant",
            sparse=root / "sparse.db",
            registry=root / "registry.db",
            benchmark_output=root / "benchmarks",
        )
        config = replace(config, paths=paths)
        overrides.append(ConfigOverride("RAG_STREAMLIT_DATA_DIR", "paths", str(root)))

    embedding_updates: dict[str, Any] = {}
    for environment, field_name in (
        ("RAG_EMBEDDING_MODEL", "model_name"),
        ("RAG_EMBEDDING_REVISION", "revision"),
        ("RAG_EMBEDDING_DEVICE", "device"),
        ("RAG_EMBEDDING_DTYPE", "dtype"),
    ):
        raw = env.get(environment, "").strip()
        if raw:
            embedding_updates[field_name] = raw
            overrides.append(ConfigOverride(environment, f"embedding.{field_name}", raw))
    raw_local = env.get("RAG_EMBEDDING_LOCAL_FILES_ONLY", "").strip()
    if raw_local:
        value = _parse_bool("RAG_EMBEDDING_LOCAL_FILES_ONLY", raw_local)
        embedding_updates["local_files_only"] = value
        overrides.append(
            ConfigOverride(
                "RAG_EMBEDDING_LOCAL_FILES_ONLY",
                "embedding.local_files_only",
                value,
            )
        )
    if embedding_updates:
        config = replace(config, embedding=replace(config.embedding, **embedding_updates))

    reranker_updates: dict[str, Any] = {}
    if "RAG_RERANKER_MODEL" in env:
        raw_model = env["RAG_RERANKER_MODEL"].strip()
        model_value = (
            None
            if raw_model.casefold() in {"", "none", "disabled", "off"}
            else raw_model
        )
        reranker_updates["model_name"] = model_value
        overrides.append(
            ConfigOverride("RAG_RERANKER_MODEL", "reranker.model_name", model_value)
        )
    for environment, field_name in (
        ("RAG_RERANKER_REVISION", "revision"),
        ("RAG_RERANKER_DEVICE", "device"),
        ("RAG_RERANKER_DTYPE", "dtype"),
    ):
        raw = env.get(environment, "").strip()
        if raw:
            reranker_updates[field_name] = raw
            overrides.append(ConfigOverride(environment, f"reranker.{field_name}", raw))
    raw_local = env.get("RAG_RERANKER_LOCAL_FILES_ONLY", "").strip()
    if raw_local:
        value = _parse_bool("RAG_RERANKER_LOCAL_FILES_ONLY", raw_local)
        reranker_updates["local_files_only"] = value
        overrides.append(
            ConfigOverride(
                "RAG_RERANKER_LOCAL_FILES_ONLY",
                "reranker.local_files_only",
                value,
            )
        )
    if reranker_updates:
        config = replace(config, reranker=replace(config.reranker, **reranker_updates))

    return config, tuple(overrides)


def build_config(environ: Mapping[str, str] | None = None) -> RagConfig:
    """Return the effective config (kept as a small compatibility helper)."""

    return resolve_config(environ)[0]


def config_fingerprint(config: RagConfig) -> str:
    """Stable identity used to bind cached resources to the complete config."""

    payload = json.dumps(
        config.snapshot(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@st.cache_resource(show_spinner="Loading the configured RAG runtime...")
def _load_rag_module(config_key: str, config: RagConfig) -> RagModule:
    # Both the full immutable config and its human-auditable fingerprint are
    # cache inputs. A changed effective config can never reuse a stale model,
    # store, or index combination.
    if config_key != config_fingerprint(config):
        raise ValueError("RAG cache key does not match the effective configuration")
    return RagModule(config)


def get_rag_module(config: RagConfig | None = None) -> RagModule:
    """Return the resource cached exactly by its effective central config."""

    effective = config or build_config()
    return _load_rag_module(config_fingerprint(effective), effective)


def clear_rag_module(module: RagModule | None = None) -> None:
    """Close the active public facade and clear Streamlit's resource cache."""

    if module is not None:
        try:
            module.close()
        except Exception:
            logger.exception("failed to close cached RAG module")
    _load_rag_module.clear()


def public_runtime_snapshot(module: RagModule) -> Mapping[str, Any]:
    """Read runtime diagnostics only through the public facade."""

    provider = getattr(module, "runtime_snapshot", None)
    if not callable(provider):
        return {"configured": module.config.snapshot(), "resolved": {}}
    value = provider()
    if not isinstance(value, Mapping):
        raise TypeError("RagModule.runtime_snapshot() must return a mapping")
    reported_config = value.get("configured")
    if reported_config is not None and json_value(reported_config) != module.config.snapshot():
        raise RuntimeError(
            "runtime diagnostics report a different configured value than RagModule.config"
        )
    return value


def json_value(value: Any) -> Any:
    """Convert public result objects into JSON-safe presentation values."""

    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_value(value.item())
        except (TypeError, ValueError):
            pass
    return value


def operational_error_message(exc: BaseException) -> str:
    """Safe, useful error text for expected public-operation failures."""

    message = str(exc).strip() or "No additional details were provided."
    return f"{type(exc).__name__}: {message}"


__all__ = [
    "ConfigOverride",
    "build_config",
    "clear_rag_module",
    "config_fingerprint",
    "get_rag_module",
    "json_value",
    "operational_error_message",
    "public_runtime_snapshot",
    "resolve_config",
]
