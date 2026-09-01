from pathlib import Path

import pytest

from rag.config import (
    ChunkingConfig,
    DenseStoreConfig,
    EmbeddingConfig,
    FusionConfig,
    ObservabilityConfig,
    OCRConfig,
    ParserConfig,
    PathsConfig,
    RagConfig,
    RerankerConfig,
    RetrievalConfig,
    SparseConfig,
)


def test_config_snapshot_is_nested_and_json_compatible(tmp_path):
    config = RagConfig(
        paths=PathsConfig(qdrant=tmp_path / "vectors"),
        embedding=EmbeddingConfig(model_name="example/model", batch_size=7),
        retrieval=RetrievalConfig(top_k=3, candidate_k=11),
    )

    snapshot = config.snapshot()
    assert snapshot["paths"]["qdrant"] == str(tmp_path / "vectors")
    assert snapshot["embedding"]["batch_size"] == 7
    assert snapshot["retrieval"] == {
        "default_mode": "hybrid",
        "top_k": 3,
        "candidate_k": 11,
        "max_query_characters": 32_000,
        "parallel_hybrid": True,
        "allow_sparse_rerank": True,
    }
    assert config.embedding_model == "example/model"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RetrievalConfig(top_k=5, candidate_k=4),
        lambda: RetrievalConfig(default_mode="automatic"),
        lambda: FusionConfig(dense_weight=0, sparse_weight=0),
        lambda: ChunkingConfig(fallback_max_tokens=10, overlap_tokens=10),
        lambda: EmbeddingConfig(model_name=" "),
        lambda: EmbeddingConfig(device="quantum"),
        lambda: EmbeddingConfig(dtype="int8"),
        lambda: EmbeddingConfig(profile_requires_remote_code=True),
        lambda: RerankerConfig(adapter="decoder"),
        lambda: RerankerConfig(yes_token_id=1),
        lambda: RerankerConfig(
            profile_requires_remote_code=True,
            trust_remote_code=False,
        ),
        lambda: OCRConfig(strategy="sometimes"),
        lambda: OCRConfig(dpi=200.5),
        lambda: OCRConfig(min_text_characters=True),
        lambda: ParserConfig(max_pages=2.5),
        lambda: ParserConfig(max_archive_members=True),
        lambda: ParserConfig(max_archive_member_bytes=1024, max_archive_uncompressed_bytes=512),
        lambda: ChunkingConfig(max_tokens=64.5),
        lambda: EmbeddingConfig(batch_size=1.5),
        lambda: EmbeddingConfig(oom_retries=True),
        lambda: FusionConfig(rrf_k=60.5),
        lambda: RerankerConfig(timeout_ms=1.5),
        lambda: RetrievalConfig(top_k=True),
        lambda: RetrievalConfig(max_query_characters=100.5),
        lambda: DenseStoreConfig(distance="euclidean"),
        lambda: SparseConfig(query_operator="xor"),
        lambda: FusionConfig(strategy="rank_average"),
        lambda: FusionConfig(dense_weight=float("nan")),
        lambda: ChunkingConfig(
            max_tokens=64,
            reserved_tokens=48,
            overlap_tokens=20,
        ),
        lambda: PathsConfig(sparse="same.db", registry="same.db"),
    ],
)
def test_invalid_configuration_fails_fast(factory):
    with pytest.raises(ValueError):
        factory()


def test_paths_accept_strings_but_expose_paths():
    paths = PathsConfig(qdrant="vectors", sparse="sparse.db")
    assert paths.qdrant == Path("vectors")
    assert paths.sparse == Path("sparse.db")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_trace_items", 1.5),
        ("max_nested_items", 1.5),
        ("max_text_characters_per_item", 1.5),
        ("vector_preview_dimensions", 1.5),
        ("max_inspection_chunks", 1.5),
        ("max_trace_items", True),
        ("max_nested_items", 0),
        ("max_text_characters_per_item", 0),
        ("vector_preview_dimensions", -1),
        ("max_inspection_chunks", 10_001),
    ],
)
def test_observability_bounds_are_strict_positive_integers(field, value):
    with pytest.raises(ValueError):
        ObservabilityConfig(**{field: value})


def test_default_runtime_paths_do_not_depend_on_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_DATA_DIR", raising=False)
    first = PathsConfig()
    monkeypatch.chdir(tmp_path)
    second = PathsConfig()
    assert first.qdrant == second.qdrant
    assert first.sparse == second.sparse
    assert first.registry == second.registry


def test_runtime_paths_honor_explicit_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "service-data"
    monkeypatch.setenv("RAG_DATA_DIR", str(data_root))
    paths = PathsConfig()
    assert paths.qdrant == data_root / "qdrant"
    assert paths.sparse == data_root / "sparse.db"
    assert paths.registry == data_root / "registry.db"
    assert paths.benchmark_output == data_root / "benchmarks"
