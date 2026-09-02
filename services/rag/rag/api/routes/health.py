from __future__ import annotations

import os

from fastapi import APIRouter, Depends

from ..dependencies import get_rag_module
from ..schemas import HealthResponse, ReadyResponse


router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def live() -> HealthResponse:
    return HealthResponse()


@router.get("/health/ready", response_model=ReadyResponse, response_model_exclude_none=True)
def ready(module=Depends(get_rag_module)) -> ReadyResponse:
    snapshot = module.runtime_snapshot()
    resolved = dict(snapshot.get("resolved") or {})
    storage = dict(resolved.get("storage") or {})
    models = {
        "embedding": resolved.get("embedding"),
        "sparse": resolved.get("sparse"),
        "reranker": resolved.get("reranker"),
    }
    return ReadyResponse(
        config_fingerprint=(
            str(snapshot.get("config_fingerprint"))
            if snapshot.get("config_fingerprint")
            else None
        ),
        active_document_count=(
            int(storage["active_document_count"])
            if storage.get("active_document_count") is not None
            else None
        ),
        runtime_profile=os.getenv("RUNTIME_PROFILE"),
        model_manifest_id=os.getenv("MODEL_MANIFEST_ID"),
        models=models if any(value is not None for value in models.values()) else None,
    )
