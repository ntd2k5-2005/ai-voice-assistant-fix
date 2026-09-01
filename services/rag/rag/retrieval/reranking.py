"""Local multi-family reranking with deterministic batching and OOM recovery."""
from __future__ import annotations

import gc
import logging
import re
from typing import Any, Callable, Sequence

import numpy as np

from ..contracts import RetrievalCandidate
from ..errors import RerankerError
from ..model_identity import profile_fingerprint, resolved_model_revision

from .reranker_capabilities import (
    RerankerProfile,
    RerankerProfileOverrides,
    resolve_reranker_profile,
)

logger = logging.getLogger(__name__)


class RerankerOOMError(RerankerError, RuntimeError):
    """The reranker still exhausted memory with a single pair."""


def _torch_module():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - transformers requires torch here
        raise RuntimeError("torch is required for local reranking") from exc
    return torch


def _is_oom_error(exc: BaseException) -> bool:
    return (
        "outofmemory" in type(exc).__name__.casefold()
        or "out of memory" in str(exc).casefold()
        or "cuda oom" in str(exc).casefold()
    )


def _valid_max_length(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if 1 < result <= 1_000_000 else None


def _resolve_device(device: str | None) -> str:
    requested = (device or "auto").casefold()
    if requested == "auto":
        torch = _torch_module()
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not _torch_module().cuda.is_available():
        raise RuntimeError("CUDA was requested for reranking but is unavailable")
    return requested


class Reranker:
    """Long-lived local reranker selected through a model capability profile."""

    def __init__(
        self,
        model_name: str,
        device: str | None = "auto",
        max_length: int | None = None,
        allow_remote_code: bool = False,
        *,
        trust_remote_code: bool | None = None,
        batch_size: int | str | None = "auto",
        auto_batch: bool = True,
        dtype: str | None = "auto",
        local_files_only: bool = False,
        revision: str | None = None,
        code_revision: str | None = None,
        timeout_ms: int | None = None,
        fallback_max_length: int | None = 512,
        max_oom_retries: int = 8,
        profile: RerankerProfile | None = None,
        profile_overrides: RerankerProfileOverrides | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
        yes_token_id: int | None = None,
        no_token_id: int | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if timeout_ms is not None and timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if fallback_max_length is not None and _valid_max_length(fallback_max_length) is None:
            raise ValueError("fallback_max_length must be between 2 and 1,000,000")
        if isinstance(max_oom_retries, bool) or max_oom_retries < 0:
            raise ValueError("max_oom_retries cannot be negative")
        self.model_name = model_name.strip()
        self.profile = profile or resolve_reranker_profile(self.model_name, profile_overrides)
        self.device = _resolve_device(device)
        self.trust_remote_code = (
            bool(allow_remote_code) if trust_remote_code is None else bool(trust_remote_code)
        )
        if self.profile.requires_remote_code and not self.trust_remote_code:
            raise ValueError(
                f"reranker profile {self.profile.family!r} requires explicit trust_remote_code=True"
            )
        self.local_files_only = bool(local_files_only)
        self.revision = revision
        self.code_revision = code_revision
        self.dtype = dtype or "auto"
        self.timeout_ms = timeout_ms
        self.auto_batch = bool(auto_batch)
        self.max_oom_retries = int(max_oom_retries)
        self.initial_batch_size, self.batch_size_source = self._resolve_batch_size(batch_size)
        self.current_batch_size = self.initial_batch_size

        self._model_was_injected = tokenizer is not None and model is not None
        self.tokenizer, self.model = self._load(tokenizer, model)
        self.resolved_revision = resolved_model_revision(
            self.model,
            tokenizer=self.tokenizer,
            requested=self.revision,
        )
        self.resolved_code_revision = self._resolved_remote_code_revision()
        self.profile_hash = profile_fingerprint(self.profile)
        if self.profile.padding_side is not None:
            self.tokenizer.padding_side = self.profile.padding_side
        if self.profile.architecture == "causal_yes_no":
            if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(
                self.tokenizer, "eos_token_id", None
            ) is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        move = getattr(self.model, "to", None)
        if move is not None:
            self.model = move(self.device)
        self.compatibility_repairs: tuple[str, ...] = ()
        self._repair_profile_buffers()
        evaluate = getattr(self.model, "eval", None)
        if evaluate is not None:
            evaluate()
        self.resolved_dtype = self._resolve_loaded_dtype()
        self.max_length, self.max_length_source = self._resolve_max_length(
            max_length, fallback_max_length
        )
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        if self.profile.architecture == "causal_yes_no":
            self.yes_token_id = self._resolve_label_token("yes", yes_token_id)
            self.no_token_id = self._resolve_label_token("no", no_token_id)
        self._closed = False

    def _resolve_batch_size(self, value: int | str | None) -> tuple[int, str]:
        if value is None or (isinstance(value, str) and value.casefold() == "auto"):
            return (16 if self.device.startswith("cuda") else 8), "automatic"
        if isinstance(value, bool):
            raise ValueError("batch_size must be positive or 'auto'")
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("batch_size must be positive or 'auto'") from exc
        if resolved <= 0:
            raise ValueError("batch_size must be positive or 'auto'")
        return resolved, "explicit"

    def _load(self, tokenizer: Any | None, model: Any | None) -> tuple[Any, Any]:
        if tokenizer is not None and model is not None:
            return tokenizer, model
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be injected together")
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for local reranking") from exc

        kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        model_kwargs = dict(kwargs)
        # Explicitly pass ``auto`` as well: supported Transformers versions then
        # honor the checkpoint dtype instead of relying on version-specific
        # from_pretrained defaults.
        model_kwargs["dtype"] = self.dtype
        if self.code_revision is not None:
            model_kwargs["code_revision"] = self.code_revision
        model_class = (
            AutoModelForCausalLM
            if self.profile.architecture == "causal_yes_no"
            else AutoModelForSequenceClassification
        )
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
            loaded_model = model_class.from_pretrained(self.model_name, **model_kwargs)
            return loaded_tokenizer, loaded_model
        except Exception as exc:
            raise RuntimeError(
                f"failed to initialize reranker {self.model_name!r} "
                f"(architecture={self.profile.architecture!r}, device={self.device!r}, "
                f"trust_remote_code={self.trust_remote_code})"
            ) from exc

    def _resolved_remote_code_revision(self) -> str | None:
        """Return the concrete Hub commit embedded in a dynamic module path."""

        for value in (
            getattr(type(self.model), "__module__", ""),
            getattr(type(getattr(self.model, "config", None)), "__module__", ""),
        ):
            for part in str(value).split("."):
                if re.fullmatch(r"[0-9a-f]{40}", part):
                    return part
        return self.code_revision

    def _resolve_loaded_dtype(self) -> str:
        parameters = getattr(self.model, "parameters", None)
        if parameters is not None:
            try:
                value = str(next(parameters()).dtype)
            except (StopIteration, TypeError, AttributeError):
                pass
            else:
                return value.removeprefix("torch.")
        configured = getattr(getattr(self.model, "config", None), "dtype", None)
        if configured is None:
            configured = getattr(getattr(self.model, "config", None), "torch_dtype", None)
        if configured is not None:
            return str(configured).removeprefix("torch.")
        return self.dtype

    def _repair_profile_buffers(self) -> None:
        path = self.profile.position_ids_buffer_path
        if not path:
            return
        owner: Any = self.model
        parts = path.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part, None)
            if owner is None:
                if self._model_was_injected:
                    return
                raise RuntimeError(
                    f"reranker profile {self.profile.family!r} requires model buffer {path!r}"
                )
        register = getattr(owner, "register_buffer", None)
        if register is None:
            if self._model_was_injected:
                return
            raise RuntimeError(
                f"reranker profile {self.profile.family!r} cannot restore model buffer {path!r}"
            )
        limit = _valid_max_length(
            getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        )
        if limit is None:
            limit = _valid_max_length(self.profile.max_length)
        if limit is None:
            raise RuntimeError(
                f"reranker profile {self.profile.family!r} lacks a safe position-id limit"
            )
        torch = _torch_module()
        existing = getattr(owner, parts[-1], None)
        device = getattr(existing, "device", None)
        if device is None:
            parameters = getattr(self.model, "parameters", None)
            if parameters is not None:
                try:
                    device = next(parameters()).device
                except (StopIteration, TypeError, AttributeError):
                    device = None
        values = torch.arange(limit, dtype=torch.long, device=device)
        register(parts[-1], values, persistent=False)
        self.compatibility_repairs = (path,)

    def _resolve_max_length(
        self, explicit: int | None, fallback: int | None
    ) -> tuple[int, str]:
        if explicit is not None and _valid_max_length(explicit) is None:
            raise ValueError("max_length must be between 2 and 1,000,000")
        candidates = []
        for source, raw in (
            ("profile.max_length", self.profile.max_length),
            ("tokenizer.model_max_length", getattr(self.tokenizer, "model_max_length", None)),
            (
                "model.config.max_position_embeddings",
                getattr(getattr(self.model, "config", None), "max_position_embeddings", None),
            ),
        ):
            value = _valid_max_length(raw)
            if value is not None:
                candidates.append((source, value))
        if explicit is not None:
            capability = min((value for _, value in candidates), default=None)
            if capability is not None and explicit > capability:
                raise ValueError(
                    f"max_length={explicit} exceeds resolved reranker capability {capability}"
                )
            return int(explicit), "explicit"
        if not candidates and fallback is not None:
            return int(fallback), "configured fallback"
        if not candidates:
            raise ValueError(
                f"cannot safely resolve max sequence length for {self.model_name!r}; provide max_length"
            )
        source, value = min(candidates, key=lambda item: item[1])
        return value, source

    def _resolve_label_token(self, label: str, explicit: int | None) -> int:
        if explicit is not None:
            if explicit < 0:
                raise ValueError(f"{label}_token_id cannot be negative")
            return int(explicit)
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if convert is not None:
            token_id = convert(label)
            unknown = getattr(self.tokenizer, "unk_token_id", None)
            if token_id is not None and int(token_id) >= 0 and token_id != unknown:
                return int(token_id)
        encode = getattr(self.tokenizer, "encode", None)
        if encode is None:
            raise ValueError(f"tokenizer cannot resolve the {label!r} label token")
        ids = encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"{label!r} is not one tokenizer token; provide {label}_token_id explicitly"
            )
        return int(ids[0])

    def _move_inputs(self, inputs: Any) -> Any:
        move = getattr(inputs, "to", None)
        if move is not None:
            return move(self.device)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _model_logits(self, inputs: Any):
        torch = _torch_module()
        with torch.no_grad():
            output = self.model(**inputs)
        logits = getattr(output, "logits", None)
        if logits is None:
            raise RuntimeError("reranker model output does not contain logits")
        return logits

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if detach is not None:
            value = detach()
        floating = getattr(value, "float", None)
        if floating is not None:
            value = floating()
        cpu = getattr(value, "cpu", None)
        if cpu is not None:
            value = cpu()
        return np.asarray(value, dtype=np.float64)

    def _positive_index(self, width: int) -> int:
        if self.profile.positive_label_index is not None:
            index = self.profile.positive_label_index
        else:
            index = None
            labels = getattr(getattr(self.model, "config", None), "id2label", {}) or {}
            for raw_index, label in labels.items():
                normalized = str(label).casefold()
                words = set(re.findall(r"[a-z0-9]+", normalized))
                if words.intersection({"relevant", "positive", "true", "yes"}):
                    index = int(raw_index)
                    break
            if index is None and width == 2:
                index = 1
        if index is None or not 0 <= index < width:
            raise RuntimeError(
                "cannot resolve the positive reranker label; configure positive_label_index"
            )
        return int(index)

    def _extract_sequence_scores(self, logits: Any, expected: int) -> np.ndarray:
        values = self._to_numpy(logits)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] != expected:
            raise RuntimeError(
                f"unsupported reranker logits shape {values.shape}; expected first dimension {expected}"
            )
        extraction = self.profile.score_extraction
        if values.shape[1] == 1:
            scalar = values[:, 0]
            if extraction in {"auto", "sigmoid", "positive_probability"}:
                scores = 1.0 / (1.0 + np.exp(-np.clip(scalar, -80, 80)))
            elif extraction in {"scalar", "positive_logit"}:
                scores = scalar
            else:
                raise RuntimeError(
                    f"score extraction {extraction!r} is incompatible with scalar logits"
                )
        else:
            positive_index = self._positive_index(values.shape[1])
            if extraction in {"auto", "positive_probability", "sigmoid"}:
                shifted = values - values.max(axis=1, keepdims=True)
                probabilities = np.exp(shifted)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                scores = probabilities[:, positive_index]
            elif extraction in {"positive_logit", "scalar"}:
                scores = values[:, positive_index]
            else:
                raise RuntimeError(
                    f"score extraction {extraction!r} is incompatible with classifier logits"
                )
        return np.asarray(scores, dtype=np.float64)

    def _score_sequence_batch(self, query: str, documents: Sequence[str]) -> np.ndarray:
        pairs = [(query, document) for document in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        logits = self._model_logits(self._move_inputs(inputs))
        return self._extract_sequence_scores(logits, len(documents))

    def _score_causal_batch(self, query: str, documents: Sequence[str]) -> np.ndarray:
        prompts = [self.profile.format_pair(query, document) for document in documents]
        prefix_tokens = self.tokenizer.encode(self.profile.prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(self.profile.suffix, add_special_tokens=False)
        content_limit = self.max_length - len(prefix_tokens) - len(suffix_tokens)
        if content_limit <= 0:
            raise RuntimeError("reranker max_length is too small for its required prompt tokens")
        tokenized = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=content_limit,
        )
        raw_ids = tokenized["input_ids"]
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        assembled = [prefix_tokens + list(ids) + suffix_tokens for ids in raw_ids]
        inputs = self.tokenizer.pad(
            {"input_ids": assembled}, padding=True, return_tensors="pt"
        )
        moved = self._move_inputs(inputs)
        logits = self._model_logits(moved)
        if getattr(logits, "ndim", None) != 3:
            raise RuntimeError(f"causal reranker requires [batch, sequence, vocabulary] logits")
        torch = _torch_module()
        attention = moved.get("attention_mask")
        if attention is None:
            positions = torch.full(
                (logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device
            )
        else:
            indices = torch.arange(attention.shape[1], device=attention.device).unsqueeze(0)
            positions = (attention.long() * indices).max(dim=1).values
        rows = torch.arange(logits.shape[0], device=logits.device)
        final_logits = logits[rows, positions]
        label_logits = torch.stack(
            [final_logits[:, self.no_token_id], final_logits[:, self.yes_token_id]], dim=1
        )
        return self._to_numpy(torch.softmax(label_logits.float(), dim=1)[:, 1])

    def _score_batch(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if self.profile.architecture == "causal_yes_no":
            scores = self._score_causal_batch(query, documents)
        else:
            scores = self._score_sequence_batch(query, documents)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape != (len(documents),) or not np.isfinite(scores).all():
            raise RuntimeError("reranker returned invalid or non-finite scores")
        if not self.profile.higher_is_better:
            scores = -scores
        return scores

    def _cleanup_after_oom(self) -> None:
        torch = _torch_module()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _candidate_text(candidate: Any, index: int) -> str:
        text = (
            candidate.text
            if hasattr(candidate, "text")
            else candidate.get("text") if isinstance(candidate, dict) else None
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"reranking candidate {index} must contain non-empty text")
        return text.strip()

    def rerank(
        self,
        query: str,
        chunks: Sequence[Any],
        top_k: int = 10,
        batch_size: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[tuple[Any, float]]:
        if self._closed:
            raise RuntimeError("reranker is closed")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        candidates = list(chunks)
        if not candidates:
            return []
        documents = [self._candidate_text(candidate, index) for index, candidate in enumerate(candidates)]

        effective = int(batch_size or self.current_batch_size)
        offset = 0
        score_batches: list[np.ndarray] = []
        oom_retries = 0
        while offset < len(documents):
            size = min(effective, len(documents) - offset)
            try:
                scores = self._score_batch(query.strip(), documents[offset : offset + size])
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                self._cleanup_after_oom()
                if (
                    size == 1
                    or not self.auto_batch
                    or oom_retries >= self.max_oom_retries
                ):
                    raise RerankerOOMError(
                        f"reranker OOM at batch_size={size}; OOM retry policy exhausted"
                    ) from exc
                oom_retries += 1
                effective = max(1, size // 2)
                self.current_batch_size = min(self.current_batch_size, effective)
                logger.warning(
                    "reranker OOM at batch_size=%d; retrying the same candidate slice with %d",
                    size,
                    effective,
                )
                continue
            score_batches.append(scores)
            offset += size
            if progress_callback is not None:
                progress_callback(offset / len(documents))

        all_scores = np.concatenate(score_batches)
        ranked = list(zip(candidates, (float(value) for value in all_scores), strict=True))
        ranked.sort(
            key=lambda item: (
                -item[1],
                str(
                    getattr(
                        item[0],
                        "chunk_id",
                        item[0].get("chunk_id", item[0].get("_id", ""))
                        if isinstance(item[0], dict)
                        else "",
                    )
                ),
            )
        )
        return ranked[:top_k]

    def rerank_candidates(
        self,
        candidates: Sequence[Any],
        query: str,
        top_k: int = 10,
        batch_size: int | None = None,
    ) -> list[RetrievalCandidate]:
        result = []
        for item, score in self.rerank(
            query, candidates, top_k=top_k, batch_size=batch_size
        ):
            candidate = (
                RetrievalCandidate(**item.__dict__)
                if isinstance(item, RetrievalCandidate)
                else RetrievalCandidate.from_mapping(item, "rerank")
            )
            candidate.rerank_score = float(score)
            candidate.final_score = float(score)
            result.append(candidate)
        return result

    @property
    def resolved_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "revision": self.resolved_revision,
            "requested_revision": self.revision,
            "code_revision": self.resolved_code_revision,
            "requested_code_revision": self.code_revision,
            "profile_hash": self.profile_hash,
            "family": self.profile.family,
            "architecture": self.profile.architecture,
            "score_extraction": self.profile.score_extraction,
            "higher_is_better": self.profile.higher_is_better,
            "device": self.device,
            "dtype": self.resolved_dtype,
            "requested_dtype": self.dtype,
            "max_length": self.max_length,
            "max_length_source": self.max_length_source,
            "batch_size": self.current_batch_size,
            "batch_size_source": self.batch_size_source,
            "max_oom_retries": self.max_oom_retries,
            "timeout_ms": self.timeout_ms,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
            "padding_side": getattr(self.tokenizer, "padding_side", None),
            "compatibility_repairs": self.compatibility_repairs,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cpu = getattr(self.model, "cpu", None)
        if cpu is not None:
            try:
                cpu()
            except (RuntimeError, AttributeError):
                logger.debug("reranker CPU release failed", exc_info=True)
        self._cleanup_after_oom()


__all__ = ["Reranker", "RerankerOOMError"]
