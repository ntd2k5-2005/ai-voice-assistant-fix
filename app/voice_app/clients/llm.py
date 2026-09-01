from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from typing import Any

import httpx

from ..contracts import ChatMessage, LLMResult, ServiceStatus
from ..errors import ContractError, ServiceError
from .base import clean_base_url, expect_json_object, response_error


class LLMClient:
    service_name = "LLM"

    def __init__(self, base_url: str, timeout_s: float = 180.0) -> None:
        self.base_url = clean_base_url(base_url)
        self.client = httpx.Client(timeout=httpx.Timeout(timeout_s))

    def close(self) -> None:
        self.client.close()

    def health(self) -> ServiceStatus:
        started = time.perf_counter()
        try:
            r = self.client.get(f"{self.base_url}/health/ready")
            latency = (time.perf_counter() - started) * 1000
            if r.status_code != 200:
                return ServiceStatus(
                    service=self.service_name,
                    reachable=True,
                    ready=False,
                    latency_ms=latency,
                    message=f"HTTP {r.status_code}",
                )
            data = expect_json_object(self.service_name, r)
            return ServiceStatus(
                service=self.service_name,
                reachable=True,
                ready=data.get("status") == "healthy",
                latency_ms=latency,
                message="Qwen/Ollama service ready",
                details=data,
            )
        except Exception as exc:
            return ServiceStatus(
                service=self.service_name,
                reachable=False,
                ready=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                message=str(exc),
            )

    @staticmethod
    def _payload(
        messages: Iterable[ChatMessage | dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        normalized = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized.append(msg.model_dump())
            else:
                normalized.append(
                    ChatMessage.model_validate(msg).model_dump()
                )
        return {
            "messages": normalized,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }

    def chat(
        self,
        messages: Iterable[ChatMessage | dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> LLMResult:
        payload = self._payload(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        try:
            r = self.client.post(
                f"{self.base_url}/v1/chat",
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise ServiceError(
                self.service_name,
                "Chat request timed out",
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(
                self.service_name,
                f"HTTP error: {exc}",
            ) from exc

        client_ms = (time.perf_counter() - started) * 1000
        if r.status_code != 200:
            raise response_error(self.service_name, r)

        body = expect_json_object(self.service_name, r)
        message = body.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"),
            str,
        ):
            raise ContractError(
                "LLM: response missing message.content string"
            )

        return LLMResult(
            text=message["content"].strip(),
            model=body.get("model"),
            timestamp=body.get("timestamp"),
            request_id=r.headers.get("X-Request-ID"),
            client_latency_ms=client_ms,
        )

    def chat_stream(
        self,
        messages: Iterable[ChatMessage | dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[dict[str, Any]]:
        payload = self._payload(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        first_delta_at: float | None = None

        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/stream",
                json=payload,
                headers={"Accept": "application/x-ndjson"},
            ) as r:
                if r.status_code != 200:
                    r.read()
                    raise response_error(self.service_name, r)

                content_type = (
                    r.headers.get("content-type") or ""
                ).lower()
                if "application/x-ndjson" not in content_type:
                    raise ContractError(
                        "LLM: expected application/x-ndjson, got "
                        f"{content_type or '<missing>'}"
                    )

                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ContractError(
                            f"LLM: invalid NDJSON line: {line[:200]!r}"
                        ) from exc

                    if not isinstance(event, dict) or "type" not in event:
                        raise ContractError(
                            f"LLM: malformed stream event: {event!r}"
                        )
                    if event["type"] == "error":
                        raise ServiceError(
                            self.service_name,
                            str(event.get("message") or "stream error"),
                        )
                    if event["type"] == "delta" and event.get("content"):
                        if first_delta_at is None:
                            first_delta_at = time.perf_counter()
                        event["client_ttft_ms"] = (
                            first_delta_at - started
                        ) * 1000
                    if event["type"] == "done":
                        event["client_total_ms"] = (
                            time.perf_counter() - started
                        ) * 1000
                    yield event

        except ServiceError:
            raise
        except httpx.TimeoutException as exc:
            raise ServiceError(
                self.service_name,
                "Streaming chat timed out",
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(
                self.service_name,
                f"HTTP error: {exc}",
            ) from exc
