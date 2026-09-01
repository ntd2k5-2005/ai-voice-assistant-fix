"""FastAPI dependencies with no import-time model/storage initialization."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status


def get_rag_module(request: Request) -> Any:
    module = getattr(request.app.state, "rag_module", None)
    if module is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "rag_not_ready", "message": "RAG runtime is not ready"},
        )
    return module


__all__ = ["get_rag_module"]
