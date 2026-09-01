"""Small environment-facing settings for the HTTP adapter.

RAG behavior remains owned by :mod:`rag.config`.  This module only resolves
transport/runtime concerns that are useful when the package is run as a
service.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class ApiSettings:
    """HTTP-server settings; deliberately separate from :class:`RagConfig`."""

    host: str = "127.0.0.1"
    port: int = 8000
    docs_enabled: bool = True
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            host=os.getenv("RAG_API_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=_env_int("RAG_API_PORT", 8000),
            docs_enabled=_env_bool("RAG_API_DOCS", True),
            log_level=(os.getenv("RAG_API_LOG_LEVEL", "info").strip() or "info").casefold(),
        )


__all__ = ["ApiSettings"]
