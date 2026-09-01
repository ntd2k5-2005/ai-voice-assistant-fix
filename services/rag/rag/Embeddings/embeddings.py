"""Production embedding engine with model profiles and bounded OOM recovery."""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .capabilities import (
    CapabilityResolutionError,
    EmbeddingProfile,
    EmbeddingProfileOverrides,
    resolve_embedding_profile,
    valid_model_max_length,
)

from ..errors import EmbeddingError
from ..model_identity import profile_fingerprint, resolved_model_revision

logger = logging.getLogger(__name__)


class EmbeddingOOMError(EmbeddingError, RuntimeError):
    """Embedding still failed after reducing the batch to one item."""


@dataclass(frozen=True)
class EmbeddingIdentity:
    model_name: str
    dimension: int
    normalized: bool
    family: str
    revision: str | None = None
    profile_hash: str = ""


def _is_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return "outofmemory" in name or "out of memory" in message or "cuda oom" in message


def _torch_module():
    try:
        import torch
    except ImportError:  # pragma: no cover - sentence-transformers installs torch
        return None
    return torch


def _resolve_device(device: str | None, model: Any | None = None) -> str:
    requested = (device or "auto").casefold()
    if requested != "auto":
        if requested == "cuda":
            torch = _torch_module()
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for embeddings but is unavailable")
        return requested
    model_device = str(getattr(model, "device", "")) if model is not None else ""
    if model_device and model_device != "None":
        return model_device
    torch = _torch_module()
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


class EmbeddingEngine:
    """A long-lived SentenceTransformer-compatible embedding component.

    A model object can be injected for tests or custom runtimes. Model-specific
    query/document formatting is supplied solely by :mod:`capabilities`.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = "auto",
        batch_size: int | str | None = "auto",
        normalize: bool | None = None,
        auto_batch: bool = True,
        safety_margin: float = 0.8,
        force_offline: bool | None = None,
        *,
        local_files_only: bool | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        dtype: str | None = "auto",
        max_length: int | None = None,
        fallback_max_length: int | None = 512,
        dimension: int | None = None,
        max_oom_retries: int = 8,
        profile: EmbeddingProfile | None = None,
        profile_overrides: EmbeddingProfileOverrides | None = None,
        model: Any | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not 0 < safety_margin <= 1:
            raise ValueError("safety_margin must be in (0, 1]")
        if max_length is not None and valid_model_max_length(max_length) is None:
            raise ValueError("max_length must be between 2 and 1,000,000")
        if fallback_max_length is not None and valid_model_max_length(fallback_max_length) is None:
            raise ValueError("fallback_max_length must be between 2 and 1,000,000")
        if dimension is not None and (isinstance(dimension, bool) or dimension <= 0):
            raise ValueError("dimension must be positive")
        if isinstance(max_oom_retries, bool) or max_oom_retries < 0:
            raise ValueError("max_oom_retries cannot be negative")

        self.model_name = model_name.strip()
        self.profile = profile or resolve_embedding_profile(self.model_name, profile_overrides)
        self.trust_remote_code = bool(trust_remote_code)
        if self.profile.requires_remote_code and not self.trust_remote_code:
            raise CapabilityResolutionError(
                f"embedding profile {self.profile.family!r} requires explicit trust_remote_code=True"
            )

        # ``force_offline`` is retained as a compatibility input, but no global
        # Hugging Face environment variables are changed and no silent online
        # fallback occurs.
        if local_files_only is None:
            local_files_only = bool(force_offline) if force_offline is not None else False
        self.local_files_only = bool(local_files_only)
        self.revision = revision
        self.requested_dtype = dtype or "auto"
        self.normalize = self.profile.normalize if normalize is None else bool(normalize)
        self.auto_batch = bool(auto_batch)
        self.safety_margin = float(safety_margin)
        self.max_oom_retries = int(max_oom_retries)

        self.device = _resolve_device(device, model)
        self.initial_batch_size, self.batch_size_source = self._resolve_batch_size(batch_size)
        self.current_batch_size = self.initial_batch_size
        self.model = model or self._load_model(model_loader)
        # An injected model may carry a more precise device than the caller.
        if (device or "auto").casefold() == "auto":
            self.device = _resolve_device(device, self.model)

        self.resolved_revision = resolved_model_revision(
            self.model,
            tokenizer=getattr(self.model, "tokenizer", None),
            requested=self.revision,
        )
        self.profile_hash = profile_fingerprint(self.profile)

        self.max_length, self.max_length_source = self._resolve_max_length(
            max_length, fallback_max_length
        )
        self.embedding_dim, self.dimension_source = self._resolve_dimension(dimension)
        self.identity = EmbeddingIdentity(
            model_name=self.model_name,
            dimension=self.embedding_dim,
            normalized=self.normalize,
            family=self.profile.family,
            revision=self.resolved_revision,
            profile_hash=self.profile_hash,
        )
        self._closed = False

    def _resolve_batch_size(self, value: int | str | None) -> tuple[int, str]:
        if value is None or (isinstance(value, str) and value.casefold() == "auto"):
            # This is a visible conservative starting point, not a model
            # capability claim; OOM recovery may reduce it at runtime.
            return (64 if self.device.startswith("cuda") else 32), "automatic"
        if isinstance(value, bool):
            raise ValueError("batch_size must be positive or 'auto'")
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("batch_size must be positive or 'auto'") from exc
        if resolved <= 0:
            raise ValueError("batch_size must be positive or 'auto'")
        return resolved, "explicit"

    def _load_model(self, model_loader: Callable[..., Any] | None) -> Any:
        if model_loader is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency error
                raise RuntimeError("sentence-transformers is required to load embedding models") from exc
            model_loader = SentenceTransformer

        model_kwargs: dict[str, Any] = {}
        if self.requested_dtype != "auto":
            model_kwargs["dtype"] = self.requested_dtype
        kwargs: dict[str, Any] = {
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        try:
            return model_loader(self.model_name, **kwargs)
        except Exception as exc:
            raise EmbeddingError(
                f"failed to initialize embedding model {self.model_name!r} "
                f"(device={self.device!r}, local_files_only={self.local_files_only}, "
                f"trust_remote_code={self.trust_remote_code})"
            ) from exc

    def _resolve_max_length(
        self, explicit: int | None, fallback: int | None
    ) -> tuple[int, str]:
        candidates: list[tuple[str, int]] = []
        for source, raw in (
            ("model.max_seq_length", getattr(self.model, "max_seq_length", None)),
            (
                "tokenizer.model_max_length",
                getattr(getattr(self.model, "tokenizer", None), "model_max_length", None),
            ),
            (
                "model.config.max_position_embeddings",
                getattr(getattr(self.model, "config", None), "max_position_embeddings", None),
            ),
        ):
            value = valid_model_max_length(raw)
            if value is not None:
                candidates.append((source, value))

        if explicit is not None:
            capability_limit = min((value for _, value in candidates), default=None)
            if capability_limit is not None and explicit > capability_limit:
                raise CapabilityResolutionError(
                    f"max_length={explicit} exceeds resolved model capability {capability_limit}"
                )
            resolved, source = int(explicit), "explicit"
        elif candidates:
            source, resolved = min(candidates, key=lambda item: item[1])
        elif fallback is not None:
            resolved, source = int(fallback), "configured fallback"
        else:
            raise CapabilityResolutionError(
                f"cannot safely resolve max sequence length for {self.model_name!r}; provide max_length"
            )

        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = resolved
        return resolved, source

    def _dimension_from_metadata(self) -> tuple[int | None, str | None]:
        for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            member = getattr(self.model, name, None)
            if member is None:
                continue
            try:
                value = member() if callable(member) else member
                if not isinstance(value, bool) and int(value) > 0:
                    return int(value), f"model.{name}"
            except (TypeError, ValueError, RuntimeError):
                continue
        for source, value in (
            ("model.embedding_dimension", getattr(self.model, "embedding_dimension", None)),
            ("model.embedding_dim", getattr(self.model, "embedding_dim", None)),
        ):
            try:
                if not isinstance(value, bool) and int(value) > 0:
                    return int(value), source
            except (TypeError, ValueError):
                pass
        return None, None

    def _resolve_dimension(self, explicit: int | None) -> tuple[int, str]:
        metadata_dimension, metadata_source = self._dimension_from_metadata()
        if explicit is not None:
            if metadata_dimension is not None and int(explicit) != metadata_dimension:
                raise CapabilityResolutionError(
                    f"configured dimension {explicit} does not match model dimension {metadata_dimension}"
                )
            return int(explicit), "explicit"
        if metadata_dimension is not None:
            return metadata_dimension, metadata_source or "model metadata"

        try:
            probe = self._call_model(
                [self.profile.format("dimension probe", "document")],
                batch_size=1,
                normalize=False,
            )
            array = np.asarray(probe)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[0] != 1 or array.shape[1] <= 0:
                raise ValueError(f"unexpected probe shape {array.shape}")
            return int(array.shape[1]), "embedding probe"
        except Exception as exc:
            raise CapabilityResolutionError(
                f"cannot determine embedding dimension for {self.model_name!r} from metadata or probe"
            ) from exc

    def _call_model(self, texts: Sequence[str], *, batch_size: int, normalize: bool) -> np.ndarray:
        values = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1 and len(texts) == 1:
            array = array.reshape(1, -1)
        return array

    def _cleanup_after_oom(self) -> None:
        torch = _torch_module()
        if torch is not None and self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _validate_texts(texts: Sequence[str], field_name: str) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise TypeError(f"{field_name} must be a sequence of strings, not one string")
        materialized = list(texts)
        for index, text in enumerate(materialized):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{field_name}[{index}] must be a non-empty string")
        return materialized

    def _adaptive_encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("embedding engine is closed")
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        effective = int(batch_size or self.current_batch_size)
        output: list[np.ndarray] = []
        offset = 0
        total = len(texts)
        oom_retries = 0
        while offset < total:
            size = min(effective, total - offset)
            batch = texts[offset : offset + size]
            try:
                vectors = self._call_model(batch, batch_size=size, normalize=self.normalize)
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                if (
                    size == 1
                    or not self.auto_batch
                    or oom_retries >= self.max_oom_retries
                ):
                    self._cleanup_after_oom()
                    raise EmbeddingOOMError(
                        f"embedding OOM at batch_size={size}; OOM retry policy exhausted"
                    ) from exc
                oom_retries += 1
                reduced = max(1, size // 2)
                effective = max(1, int(reduced * self.safety_margin))
                self.current_batch_size = min(self.current_batch_size, effective)
                logger.warning(
                    "embedding OOM at batch_size=%d; retrying the same input slice with batch_size=%d",
                    size,
                    effective,
                )
                self._cleanup_after_oom()
                continue

            if vectors.shape != (size, self.embedding_dim):
                raise RuntimeError(
                    f"embedding contract violated: got {vectors.shape}, "
                    f"expected ({size}, {self.embedding_dim})"
                )
            if not np.isfinite(vectors).all():
                raise RuntimeError("embedding model returned non-finite vectors")
            if self.normalize:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
            output.append(vectors)
            offset += size
            if progress_callback is not None:
                progress_callback(offset / total)

        return np.vstack(output)

    def _format_many(self, texts: Sequence[str], kind: str) -> list[str]:
        return [self.profile.format(text.strip(), kind) for text in texts]  # type: ignore[arg-type]

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        documents = self._validate_texts(texts, "documents")
        return self._adaptive_encode(
            self._format_many(documents, "document"),
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

    def embed_queries_batch(
        self,
        queries: Sequence[str],
        batch_size: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        values = self._validate_texts(queries, "queries")
        return self._adaptive_encode(
            self._format_many(values, "query"),
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

    def embed_query(self, text: str, batch_size: int | None = None) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query must be a non-empty string")
        return self.embed_queries_batch([text], batch_size=batch_size)[0]

    def embed_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        batch_size: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        rows = [dict(chunk) for chunk in chunks]
        vectors = self.embed_documents(
            [row.get("text") for row in rows],  # type: ignore[list-item]
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
        for row, vector in zip(rows, vectors, strict=True):
            row["vector"] = vector
        return rows

    @staticmethod
    def format_embedding_input(
        texts: Sequence[str], model_name: str, mode: str = "doc"
    ) -> list[str]:
        """Compatibility helper; new code should consume a resolved profile."""

        profile = resolve_embedding_profile(model_name)
        kind = "query" if mode == "query" else "document"
        return [profile.format(text, kind) for text in texts]  # type: ignore[arg-type]

    @property
    def resolved_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "revision": self.resolved_revision,
            "requested_revision": self.revision,
            "profile_hash": self.profile_hash,
            "family": self.profile.family,
            "device": self.device,
            "dtype": str(getattr(self.model, "dtype", self.requested_dtype)),
            "dimension": self.embedding_dim,
            "dimension_source": self.dimension_source,
            "max_length": self.max_length,
            "max_length_source": self.max_length_source,
            "batch_size": self.current_batch_size,
            "batch_size_source": self.batch_size_source,
            "max_oom_retries": self.max_oom_retries,
            "normalize": self.normalize,
            "query_template": self.profile.query_template,
            "document_template": self.profile.document_template,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }

    def get_vram_report(self) -> dict[str, Any]:
        report = {
            "device": self.device,
            "batch_size": self.current_batch_size,
            "model": self.model_name,
        }
        torch = _torch_module()
        if torch is None or not self.device.startswith("cuda") or not torch.cuda.is_available():
            return report
        try:
            free, total = torch.cuda.mem_get_info()
            report.update(
                free_mb=round(free / 1024**2, 1),
                total_mb=round(total / 1024**2, 1),
            )
        except RuntimeError as exc:  # pragma: no cover - hardware dependent
            report["error"] = type(exc).__name__
        return report

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cpu = getattr(self.model, "cpu", None)
        if cpu is not None:
            try:
                cpu()
            except (RuntimeError, AttributeError):
                logger.debug("embedding model CPU release failed", exc_info=True)
        self._cleanup_after_oom()

    def __enter__(self) -> "EmbeddingEngine":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"EmbeddingEngine(model={self.model_name!r}, family={self.profile.family!r}, "
            f"device={self.device!r}, dimension={self.embedding_dim}, "
            f"batch_size={self.current_batch_size})"
        )


__all__ = ["EmbeddingEngine", "EmbeddingIdentity", "EmbeddingOOMError"]
