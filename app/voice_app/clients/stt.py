from __future__ import annotations

import mimetypes
import time
from pathlib import Path

import httpx

from ..contracts import STTResult, ServiceStatus
from ..errors import ContractError, ServiceError
from .base import clean_base_url, expect_json_object, response_error


class STTClient:
    """
    Adapter for current YuluWusu/Speech-to-Text-Long API.

    Demo V1 intentionally uses POST /transcribe (completed microphone/file).
    The upstream service also exposes WS /ws/transcribe with Silero VAD,
    but browser/WebSocket streaming is deliberately kept out of this first
    integration layer to keep end-to-end debugging deterministic.
    """

    service_name = "STT"

    def __init__(self, base_url: str, timeout_s: float = 180.0) -> None:
        self.base_url = clean_base_url(base_url)
        self.client = httpx.Client(timeout=httpx.Timeout(timeout_s))

    def close(self) -> None:
        self.client.close()

    def health(self) -> ServiceStatus:
        started = time.perf_counter()
        try:
            r = self.client.get(f"{self.base_url}/health")
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
                ready=data.get("status") == "ok",
                latency_ms=latency,
                message="PhoWhisper + Silero VAD service reachable",
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

    def transcribe(self, audio_path: str | Path) -> STTResult:
        path = Path(audio_path)
        if not path.is_file():
            raise ServiceError(self.service_name, f"Audio file not found: {path}")

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        started = time.perf_counter()

        try:
            with path.open("rb") as fh:
                r = self.client.post(
                    f"{self.base_url}/transcribe",
                    files={"file": (path.name, fh, content_type)},
                )
        except httpx.TimeoutException as exc:
            raise ServiceError(self.service_name, "Transcription timed out") from exc
        except httpx.HTTPError as exc:
            raise ServiceError(self.service_name, f"HTTP error: {exc}") from exc

        client_ms = (time.perf_counter() - started) * 1000
        if r.status_code != 200:
            raise response_error(self.service_name, r)

        body = expect_json_object(self.service_name, r)
        text = body.get("text")
        if not isinstance(text, str):
            raise ContractError("STT: response is missing string field 'text'")

        audio = body.get("audio") or {}
        inference = body.get("inference") or {}
        inference_s = inference.get("time_seconds")

        return STTResult(
            text=text.strip(),
            model=body.get("model"),
            sample_rate=audio.get("sample_rate"),
            audio_duration_s=audio.get("duration_seconds"),
            inference_ms=(
                float(inference_s) * 1000
                if isinstance(inference_s, (int, float))
                else None
            ),
            rtf=inference.get("rtf"),
            client_latency_ms=client_ms,
        )
