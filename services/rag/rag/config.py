"""Central configuration for the standalone RAG runtime.

Only user intent lives in these dataclasses. Model-derived values such as the
embedding dimension and effective sequence length are exposed separately by
the model capability layer; they are never copied back into this immutable
configuration object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Literal, Mapping


AutoDevice = Literal["auto", "cpu", "cuda", "mps"]
AutoDtype = Literal["auto", "float32", "float16", "bfloat16"]


def default_runtime_root() -> Path:
    """Resolve one stable writable root for local service state.

    ``RAG_DATA_DIR`` wins when configured.  In an editable/source checkout we
    anchor to the repository containing ``pyproject.toml`` so Streamlit,
    FastAPI, tests, and CLI commands do not create different stores merely
    because their current working directories differ.  Installed wheels fall
    back to a user-owned home directory rather than writing into site-packages.
    """

    configured = os.getenv("RAG_DATA_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser().resolve(strict=False)

    package_file = Path(__file__).resolve(strict=False)
    for parent in package_file.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "rag").is_dir():
            return (parent / ".rag_runtime").resolve(strict=False)

    return (Path.home() / ".robust-rag-service").resolve(strict=False)


def _runtime_path(name: str) -> Path:
    return default_runtime_root() / name


def _positive(name: str, value: int | float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")


def _positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _one_of(name: str, value: Any, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        choices = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"{name} must be one of {choices}")


@dataclass(frozen=True)
class PathsConfig:
    """All writable paths used by the runtime."""

    qdrant: Path = field(default_factory=lambda: _runtime_path("qdrant"))
    sparse: Path = field(default_factory=lambda: _runtime_path("sparse.db"))
    registry: Path = field(default_factory=lambda: _runtime_path("registry.db"))
    benchmark_output: Path = field(default_factory=lambda: _runtime_path("benchmarks"))

    def __post_init__(self) -> None:
        for name in ("qdrant", "sparse", "registry", "benchmark_output"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        resolved = {
            name: getattr(self, name).resolve(strict=False)
            for name in ("qdrant", "sparse", "registry", "benchmark_output")
        }
        if resolved["sparse"] == resolved["registry"]:
            raise ValueError("paths.sparse and paths.registry must be different")
        if resolved["qdrant"] in {resolved["sparse"], resolved["registry"]}:
            raise ValueError("paths.qdrant cannot also be a SQLite file path")
        if resolved["benchmark_output"] == resolved["qdrant"]:
            raise ValueError("paths.benchmark_output must differ from paths.qdrant")


@dataclass(frozen=True)
class OCRConfig:
    strategy: Literal["auto", "never", "always"] = "auto"
    backend: Literal["auto", "tesseract", "rapidocr"] = "auto"
    languages: tuple[str, ...] = ("vie", "eng")
    dpi: int = 200
    min_text_characters: int = 24
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        _one_of("ocr.strategy", self.strategy, ("auto", "never", "always"))
        _one_of("ocr.backend", self.backend, ("auto", "tesseract", "rapidocr"))
        _positive_int("ocr.dpi", self.dpi)
        _positive("ocr.timeout_seconds", self.timeout_seconds)
        _nonnegative_int("ocr.min_text_characters", self.min_text_characters)
        if self.strategy != "never" and not self.languages:
            raise ValueError("ocr.languages cannot be empty when OCR is enabled")


@dataclass(frozen=True)
class ParserConfig:
    max_file_bytes: int = 100 * 1024 * 1024
    max_pages: int = 2_000
    max_units: int = 100_000
    max_archive_members: int = 10_000
    max_archive_uncompressed_bytes: int = 512 * 1024 * 1024
    max_archive_member_bytes: int = 256 * 1024 * 1024
    max_archive_compression_ratio: float = 200.0
    sniff_mime: bool = True
    reject_extension_mismatch: bool = False
    encoding_fallbacks: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1258", "latin-1")
    allowed_extensions: tuple[str, ...] = (
        ".pdf", ".docx", ".xlsx", ".xlsm", ".pptx",
        ".csv", ".tsv", ".txt", ".md", ".markdown",
        ".html", ".htm", ".json", ".jsonl", ".ldjson", ".epub", ".mdx",
        ".py", ".js", ".java", ".c", ".cpp", ".h", ".php", ".go",
        ".ts", ".sh", ".cs", ".kt", ".sql",
    )
    ocr: OCRConfig = field(default_factory=OCRConfig)

    def __post_init__(self) -> None:
        _positive_int("parser.max_file_bytes", self.max_file_bytes)
        _positive_int("parser.max_pages", self.max_pages)
        _positive_int("parser.max_units", self.max_units)
        for name in (
            "max_archive_members",
            "max_archive_uncompressed_bytes",
            "max_archive_member_bytes",
        ):
            _positive_int(f"parser.{name}", getattr(self, name))
        _positive(
            "parser.max_archive_compression_ratio",
            self.max_archive_compression_ratio,
        )
        if self.max_archive_member_bytes > self.max_archive_uncompressed_bytes:
            raise ValueError(
                "parser.max_archive_member_bytes cannot exceed the total archive limit"
            )
        if not self.encoding_fallbacks:
            raise ValueError("parser.encoding_fallbacks cannot be empty")
        normalized = tuple(
            extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
            for extension in self.allowed_extensions
        )
        if any(extension == "." for extension in normalized):
            raise ValueError("parser.allowed_extensions contains an empty extension")
        if len(set(normalized)) != len(normalized):
            raise ValueError("parser.allowed_extensions cannot contain duplicates")
        object.__setattr__(self, "allowed_extensions", normalized)


@dataclass(frozen=True)
class NormalizationConfig:
    unicode_form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFC"
    strip_control_characters: bool = True
    normalize_line_endings: bool = True
    trim_trailing_whitespace: bool = True
    collapse_blank_lines: int = 2

    def __post_init__(self) -> None:
        _one_of("normalization.unicode_form", self.unicode_form, ("NFC", "NFKC", "NFD", "NFKD"))
        _nonnegative_int(
            "normalization.collapse_blank_lines",
            self.collapse_blank_lines,
        )


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk limits; ``max_tokens=None`` delegates to model capabilities."""

    max_tokens: int | None = None
    fallback_max_tokens: int = 384
    overlap_tokens: int = 48
    reserved_tokens: int = 32
    min_chunk_tokens: int = 8
    max_chunks_per_document: int = 100_000
    tokenizer_name: str | None = None
    preserve_tables: bool = True

    def __post_init__(self) -> None:
        _positive_int("chunking.max_tokens", self.max_tokens)
        _positive_int("chunking.fallback_max_tokens", self.fallback_max_tokens)
        _positive_int("chunking.min_chunk_tokens", self.min_chunk_tokens)
        _positive_int("chunking.max_chunks_per_document", self.max_chunks_per_document)
        _nonnegative_int("chunking.overlap_tokens", self.overlap_tokens)
        _nonnegative_int("chunking.reserved_tokens", self.reserved_tokens)
        limit = self.max_tokens or self.fallback_max_tokens
        if self.reserved_tokens >= limit:
            raise ValueError("chunking.reserved_tokens must be smaller than the token limit")
        effective_limit = limit - self.reserved_tokens
        if self.overlap_tokens >= effective_limit:
            raise ValueError("chunking.overlap_tokens must be smaller than the effective token limit")
        if self.min_chunk_tokens > effective_limit:
            raise ValueError("chunking.min_chunk_tokens cannot exceed the effective token limit")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "intfloat/multilingual-e5-small"
    revision: str | None = None
    device: AutoDevice = "auto"
    dtype: AutoDtype = "auto"
    batch_size: int | None = None
    max_length: int | None = None
    dimension: int | None = None
    fallback_max_length: int = 512
    normalize: bool | None = None
    profile_family: str | None = None
    query_template: str | None = None
    document_template: str | None = None
    query_instruction: str | None = None
    document_instruction: str | None = None
    profile_requires_remote_code: bool | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    oom_retries: int = 8
    batch_safety_margin: float = 0.85

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("embedding.model_name cannot be empty")
        _one_of("embedding.device", self.device, ("auto", "cpu", "cuda", "mps"))
        _one_of("embedding.dtype", self.dtype, ("auto", "float32", "float16", "bfloat16"))
        _positive_int("embedding.batch_size", self.batch_size)
        _positive_int("embedding.max_length", self.max_length)
        _positive_int("embedding.dimension", self.dimension)
        _positive_int("embedding.fallback_max_length", self.fallback_max_length)
        if self.query_template is not None and "{text}" not in self.query_template:
            raise ValueError("embedding.query_template must contain {text}")
        if self.document_template is not None and "{text}" not in self.document_template:
            raise ValueError("embedding.document_template must contain {text}")
        if (
            self.query_instruction is not None
            and self.query_template is not None
            and "{instruction}" not in self.query_template
        ):
            raise ValueError("embedding.query_template must consume the configured instruction")
        if (
            self.document_instruction is not None
            and self.document_template is not None
            and "{instruction}" not in self.document_template
        ):
            raise ValueError("embedding.document_template must consume the configured instruction")
        if self.profile_requires_remote_code is True and not self.trust_remote_code:
            raise ValueError("embedding profile requires explicit trust_remote_code=True")
        _nonnegative_int("embedding.oom_retries", self.oom_retries)
        if not 0 < self.batch_safety_margin <= 1:
            raise ValueError("embedding.batch_safety_margin must be in (0, 1]")


@dataclass(frozen=True)
class DenseStoreConfig:
    collection_name: str = "rag_documents"
    distance: Literal["cosine"] = "cosine"

    def __post_init__(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("dense_store.collection_name cannot be empty")
        _one_of("dense_store.distance", self.distance, ("cosine",))


@dataclass(frozen=True)
class SparseConfig:
    analyzer: Literal["unicode61"] = "unicode61"
    query_operator: Literal["or", "and"] = "or"

    def __post_init__(self) -> None:
        _one_of("sparse.analyzer", self.analyzer, ("unicode61",))
        _one_of("sparse.query_operator", self.query_operator, ("or", "and"))

    @property
    def tokenizer(self) -> str:
        """Read-only compatibility alias for the implemented analyzer name."""

        return self.analyzer


@dataclass(frozen=True)
class FusionConfig:
    strategy: Literal["weighted", "rrf"] = "rrf"
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60

    def __post_init__(self) -> None:
        _one_of("fusion.strategy", self.strategy, ("weighted", "rrf"))
        if not math.isfinite(self.dense_weight) or not math.isfinite(self.sparse_weight):
            raise ValueError("fusion weights must be finite")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("fusion weights cannot be negative")
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("at least one fusion weight must be positive")
        _positive_int("fusion.rrf_k", self.rrf_k)


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str | None = "Alibaba-NLP/gte-multilingual-reranker-base"
    revision: str | None = "8215cf04918ba6f7b6a62bb44238ce2953d8831c"
    code_revision: str | None = "40ced75c3017eb27626c9d4ea981bde21a2662f4"
    adapter: Literal["auto", "sequence_classification", "causal_lm"] = "auto"
    profile_family: str | None = None
    score_extraction: Literal[
        "auto", "scalar", "sigmoid", "positive_logit",
        "positive_probability", "yes_no_probability",
    ] | None = None
    positive_label_index: int | None = None
    higher_is_better: bool | None = None
    pair_template: str | None = None
    instruction: str | None = None
    profile_requires_remote_code: bool | None = None
    yes_token_id: int | None = None
    no_token_id: int | None = None
    device: AutoDevice = "auto"
    dtype: AutoDtype = "auto"
    batch_size: int | None = None
    max_length: int | None = None
    fallback_max_length: int = 512
    trust_remote_code: bool = True
    local_files_only: bool = False
    timeout_ms: int | None = None
    failure_policy: Literal["fallback", "raise"] = "fallback"
    oom_retries: int = 8

    def __post_init__(self) -> None:
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("reranker.model_name cannot be blank")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("reranker.revision cannot be blank")
        if self.code_revision is not None and not self.code_revision.strip():
            raise ValueError("reranker.code_revision cannot be blank")
        _one_of("reranker.adapter", self.adapter, ("auto", "sequence_classification", "causal_lm"))
        _one_of("reranker.device", self.device, ("auto", "cpu", "cuda", "mps"))
        _one_of("reranker.dtype", self.dtype, ("auto", "float32", "float16", "bfloat16"))
        _one_of("reranker.failure_policy", self.failure_policy, ("fallback", "raise"))
        if self.score_extraction is not None:
            _one_of(
                "reranker.score_extraction",
                self.score_extraction,
                ("auto", "scalar", "sigmoid", "positive_logit", "positive_probability", "yes_no_probability"),
            )
        _positive_int("reranker.batch_size", self.batch_size)
        _positive_int("reranker.max_length", self.max_length)
        _positive_int("reranker.fallback_max_length", self.fallback_max_length)
        _positive_int("reranker.timeout_ms", self.timeout_ms)
        if self.positive_label_index is not None:
            _nonnegative_int(
                "reranker.positive_label_index",
                self.positive_label_index,
            )
        if self.pair_template is not None and (
            "{query}" not in self.pair_template or "{document}" not in self.pair_template
        ):
            raise ValueError("reranker.pair_template must contain {query} and {document}")
        _nonnegative_int("reranker.oom_retries", self.oom_retries)
        if self.profile_requires_remote_code is True and not self.trust_remote_code:
            raise ValueError("reranker profile requires explicit trust_remote_code=True")
        if (self.yes_token_id is None) != (self.no_token_id is None):
            raise ValueError("reranker yes_token_id and no_token_id must be configured together")


@dataclass(frozen=True)
class RetrievalConfig:
    default_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 5
    candidate_k: int = 50
    max_query_characters: int = 32_000
    parallel_hybrid: bool = True
    allow_sparse_rerank: bool = True

    def __post_init__(self) -> None:
        _one_of("retrieval.default_mode", self.default_mode, ("dense", "sparse", "hybrid"))
        _positive_int("retrieval.top_k", self.top_k)
        _positive_int("retrieval.candidate_k", self.candidate_k)
        if self.candidate_k < self.top_k:
            raise ValueError("retrieval.candidate_k must be >= top_k > 0")
        _positive_int("retrieval.max_query_characters", self.max_query_characters)


@dataclass(frozen=True)
class LifecycleConfig:
    recover_on_startup: bool = True
    cleanup_old_versions: bool = True
    lock_timeout_seconds: float = 30.0
    visibility_retry_count: int = 2

    def __post_init__(self) -> None:
        _positive("lifecycle.lock_timeout_seconds", self.lock_timeout_seconds)
        _nonnegative_int(
            "lifecycle.visibility_retry_count",
            self.visibility_retry_count,
        )


@dataclass(frozen=True)
class ObservabilityConfig:
    """Bounds for opt-in, JSON-safe execution traces and inspection views."""

    max_trace_items: int = 2_000
    max_nested_items: int = 100
    max_text_characters_per_item: int = 4_000
    vector_preview_dimensions: int = 8
    max_inspection_chunks: int = 5_000

    def __post_init__(self) -> None:
        for name in (
            "max_trace_items",
            "max_nested_items",
            "max_text_characters_per_item",
            "vector_preview_dimensions",
            "max_inspection_chunks",
        ):
            _positive_int(f"observability.{name}", getattr(self, name))
        if self.max_inspection_chunks > 10_000:
            raise ValueError(
                "observability.max_inspection_chunks cannot exceed the store safety limit of 10000"
            )


@dataclass(frozen=True)
class RagConfig:
    """Single source of truth for both knowledge and retrieval flows."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    dense_store: DenseStoreConfig = field(default_factory=DenseStoreConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible benchmark snapshot."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    # Read-only migration aids. The nested sections above remain canonical.
    @property
    def qdrant_path(self) -> Path:
        return self.paths.qdrant

    @property
    def sparse_path(self) -> Path:
        return self.paths.sparse

    @property
    def registry_path(self) -> Path:
        return self.paths.registry

    @property
    def embedding_model(self) -> str:
        return self.embedding.model_name

    @property
    def reranker_model(self) -> str | None:
        return self.reranker.model_name

    @property
    def top_k(self) -> int:
        return self.retrieval.top_k

    @property
    def candidate_k(self) -> int:
        return self.retrieval.candidate_k

    @property
    def dense_weight(self) -> float:
        return self.fusion.dense_weight

    @property
    def sparse_weight(self) -> float:
        return self.fusion.sparse_weight

    @property
    def rerank_timeout_ms(self) -> int | None:
        return self.reranker.timeout_ms


__all__ = [
    "ChunkingConfig", "DenseStoreConfig", "EmbeddingConfig", "FusionConfig",
    "LifecycleConfig", "NormalizationConfig", "ObservabilityConfig", "OCRConfig", "ParserConfig",
    "PathsConfig", "RagConfig", "RerankerConfig", "RetrievalConfig",
    "SparseConfig", "default_runtime_root",
]
