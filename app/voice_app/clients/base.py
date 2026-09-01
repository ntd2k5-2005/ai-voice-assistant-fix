from __future__ import annotations

import json
from typing import Any

import httpx

from ..errors import ContractError, ServiceError


def clean_base_url(url: str) -> str:
    return url.rstrip("/")


def response_error(service: str, response: httpx.Response) -> ServiceError:
    details: Any = None
    message = response.text[:1500]
    try:
        details = response.json()
        if isinstance(details, dict):
            message = str(
                details.get("message")
                or details.get("detail")
                or details.get("error")
                or message
            )
    except Exception:
        pass
    return ServiceError(
        service,
        message or "Upstream request failed",
        status_code=response.status_code,
        details=details,
    )


def expect_json_object(service: str, response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception as exc:
        raise ContractError(
            f"{service}: expected JSON object, got invalid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ContractError(
            f"{service}: expected JSON object, got {type(data).__name__}"
        )
    return data
