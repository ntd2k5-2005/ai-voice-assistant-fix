from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterator, Optional

import ollama

from llm.config import Settings
from llm.errors import ModelNotFoundError, OllamaUnavailableError

logger = logging.getLogger(__name__)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _message_dict(message: Any) -> dict[str, str]:
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        thinking = message.get("thinking")
    else:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        thinking = getattr(message, "thinking", None)

    content = (content or "").strip()
    thinking = (thinking or "").strip()
    if not content:
        raise RuntimeError(
            "Ollama returned empty assistant content "
            f"(thinking_len={len(thinking)}). "
            "Voice chat requires a non-empty final answer."
        )
    return {"role": role or "assistant", "content": content}


def _find_model(response: Any, model_name: str) -> Any | None:
    for model in _field(response, "models", []) or []:
        name = _field(model, "model") or _field(model, "name")
        if name == model_name:
            return model
    return None


def _active_processor(model: Any | None) -> str | None:
    if model is None:
        return None
    size = int(_field(model, "size", 0) or 0)
    size_vram = int(_field(model, "size_vram", 0) or 0)
    if size_vram <= 0:
        return "cpu"
    if size > 0 and size_vram >= size:
        return "gpu"
    return "cpu+gpu"


class LLMService:
    """Stateless Ollama service. Conversation history is supplied per request."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.model_name
        self.timeout = settings.ollama_timeout
        self.client = ollama.Client(
            host=settings.ollama_url,
            timeout=settings.ollama_timeout,
        )
        logger.info(
            "LLMService initialized model=%s ollama=%s timeout=%ss",
            self.model_name,
            settings.ollama_url,
            self.timeout,
        )

    def health_check(self) -> dict[str, Any]:
        try:
            info = self.client.show(self.model_name)
            listed = _find_model(self.client.list(), self.model_name)
            active = _find_model(self.client.ps(), self.model_name)
            details = _field(listed, "details") or _field(info, "details", {}) or {}
            return {
                "ollama_available": True,
                "model_loaded": True,
                "model_name": self.model_name,
                "model_digest": _field(listed, "digest") or _field(info, "digest"),
                "quantization_level": _field(details, "quantization_level"),
                "parameter_size": _field(details, "parameter_size"),
                "configured_placement": self.settings.ollama_placement,
                "configured_num_gpu": self.settings.ollama_num_gpu,
                "active_processor": _active_processor(active),
                "active_size_bytes": _field(active, "size"),
                "active_size_vram_bytes": _field(active, "size_vram"),
                "runtime_profile": os.getenv("RUNTIME_PROFILE"),
                "model_manifest_id": os.getenv("MODEL_MANIFEST_ID"),
                "error": None,
            }
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                return {
                    "ollama_available": True,
                    "model_loaded": False,
                    "model_name": self.model_name,
                    "error": f"Model {self.model_name} not found",
                }
            return {
                "ollama_available": False,
                "model_loaded": False,
                "model_name": self.model_name,
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "ollama_available": False,
                "model_loaded": False,
                "model_name": self.model_name,
                "error": str(exc),
            }

    def _options(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        options = {
            "temperature": (
                self.settings.default_temperature
                if temperature is None
                else temperature
            ),
            "top_p": self.settings.default_top_p if top_p is None else top_p,
            "num_predict": (
                self.settings.default_max_tokens
                if max_tokens is None
                else max_tokens
            ),
        }
        if self.settings.ollama_num_gpu is not None:
            options["num_gpu"] = self.settings.ollama_num_gpu
        if self.settings.ollama_num_ctx is not None:
            options["num_ctx"] = self.settings.ollama_num_ctx
        if self.settings.ollama_num_thread is not None:
            options["num_thread"] = self.settings.ollama_num_thread
        return options

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                think=False,
                options=self._options(temperature, top_p, max_tokens),
            )
            message = _message_dict(_field(response, "message"))
            logger.info(
                "[%s] chat completed in %.3fs",
                request_id or "-",
                time.perf_counter() - started,
            )
            return {"message": message}
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise ModelNotFoundError(self.model_name) from exc
            raise OllamaUnavailableError(self.settings.ollama_url, exc) from exc
        except (ModelNotFoundError, OllamaUnavailableError):
            raise
        except Exception as exc:
            raise OllamaUnavailableError(self.settings.ollama_url, exc) from exc

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield final-answer deltas from Ollama as soon as they are produced."""
        started = time.perf_counter()
        first_delta_at: float | None = None
        text_parts: list[str] = []
        final_eval_count: int | None = None
        final_eval_duration_ns: int | None = None

        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                think=False,
                stream=True,
                options=self._options(temperature, top_p, max_tokens),
            )
            for chunk in stream:
                message = _field(chunk, "message")
                content = str(_field(message, "content", "") or "")
                if content:
                    if first_delta_at is None:
                        first_delta_at = time.perf_counter()
                    text_parts.append(content)
                    yield {"type": "delta", "content": content}

                eval_count = _field(chunk, "eval_count")
                eval_duration = _field(chunk, "eval_duration")
                if isinstance(eval_count, int):
                    final_eval_count = eval_count
                if isinstance(eval_duration, int):
                    final_eval_duration_ns = eval_duration

            ended = time.perf_counter()
            full_text = "".join(text_parts).strip()
            if not full_text:
                raise RuntimeError("Ollama returned empty assistant content in stream")

            eval_seconds = (
                final_eval_duration_ns / 1_000_000_000
                if final_eval_duration_ns and final_eval_duration_ns > 0
                else None
            )
            yield {
                "type": "done",
                "message": {"role": "assistant", "content": full_text},
                "metrics": {
                    "ttft_ms": (
                        (first_delta_at - started) * 1000
                        if first_delta_at is not None
                        else None
                    ),
                    "total_latency_ms": (ended - started) * 1000,
                    "tokens_per_second": (
                        final_eval_count / eval_seconds
                        if final_eval_count is not None and eval_seconds
                        else None
                    ),
                    "output_tokens": final_eval_count,
                },
            }
            logger.info(
                "[%s] streaming chat completed in %.3fs",
                request_id or "-",
                ended - started,
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise ModelNotFoundError(self.model_name) from exc
            raise OllamaUnavailableError(self.settings.ollama_url, exc) from exc
        except (ModelNotFoundError, OllamaUnavailableError):
            raise
        except Exception as exc:
            raise OllamaUnavailableError(self.settings.ollama_url, exc) from exc

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        track_metrics: bool = False,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if track_metrics and self.settings.enable_metrics:
            return self._generate_with_metrics(
                prompt,
                self.settings.default_temperature
                if temperature is None
                else temperature,
                self.settings.default_top_p if top_p is None else top_p,
                self.settings.default_max_tokens
                if max_tokens is None
                else max_tokens,
            )
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                think=False,
                options=self._options(temperature, top_p, max_tokens),
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise ModelNotFoundError(self.model_name) from exc
            raise OllamaUnavailableError(self.settings.ollama_url, exc) from exc
        except Exception as exc:
            raise OllamaUnavailableError(self.settings.ollama_url, exc) from exc

        text = str(_field(response, "response", "") or "").strip()
        if not text:
            raise OllamaUnavailableError(
                self.settings.ollama_url,
                RuntimeError("Ollama returned empty generation content"),
            )
        return {"response": text, "metrics": None}

    def _generate_with_metrics(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        first_chunk_at: float | None = None
        text_parts: list[str] = []
        final_eval_count: int | None = None
        final_eval_duration_ns: int | None = None
        try:
            stream = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                think=False,
                stream=True,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                },
            )
            for chunk in stream:
                chunk_text = str(_field(chunk, "response", "") or "")
                if chunk_text and first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                if chunk_text:
                    text_parts.append(chunk_text)
                eval_count = _field(chunk, "eval_count")
                eval_duration = _field(chunk, "eval_duration")
                if isinstance(eval_count, int):
                    final_eval_count = eval_count
                if isinstance(eval_duration, int):
                    final_eval_duration_ns = eval_duration
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise ModelNotFoundError(self.model_name) from exc
            raise OllamaUnavailableError(self.settings.ollama_url, exc) from exc
        except Exception as exc:
            raise OllamaUnavailableError(self.settings.ollama_url, exc) from exc

        ended = time.perf_counter()
        text = "".join(text_parts).strip()
        if not text:
            raise OllamaUnavailableError(
                self.settings.ollama_url,
                RuntimeError("Ollama returned empty generation content"),
            )
        eval_seconds = (
            final_eval_duration_ns / 1_000_000_000
            if final_eval_duration_ns and final_eval_duration_ns > 0
            else None
        )
        return {
            "response": text,
            "metrics": {
                "ttft_ms": (
                    (first_chunk_at - started) * 1000
                    if first_chunk_at is not None
                    else None
                ),
                "total_latency_ms": (ended - started) * 1000,
                "tokens_per_second": (
                    final_eval_count / eval_seconds
                    if final_eval_count is not None and eval_seconds
                    else None
                ),
                "output_tokens": final_eval_count,
            },
        }
