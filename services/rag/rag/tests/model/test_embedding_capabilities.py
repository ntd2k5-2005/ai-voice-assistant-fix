from __future__ import annotations

import numpy as np
import pytest

from rag.Embeddings.adapters import EngineEmbeddingAdapter
from rag.Embeddings.capabilities import (
    CapabilityResolutionError,
    EmbeddingProfileOverrides,
    resolve_embedding_profile,
    valid_model_max_length,
)
from rag.Embeddings.embeddings import EmbeddingEngine, EmbeddingOOMError
from rag.Embeddings.factory import adapter_for


class FakeEmbeddingModel:
    max_seq_length = 128

    def __init__(self, *, dimension=3, oom_above=None):
        self.dimension = dimension
        self.oom_above = oom_above
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise RuntimeError("CUDA out of memory")
        rows = []
        for text in texts:
            digits = "".join(character for character in text if character.isdigit())
            value = float(digits or len(text))
            row = [value, 1.0, 0.0][: self.dimension]
            row.extend([0.0] * (self.dimension - len(row)))
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)


def make_engine(model_name="intfloat/multilingual-e5-small", **kwargs):
    model = kwargs.pop("model", FakeEmbeddingModel())
    return EmbeddingEngine(
        model_name=model_name,
        model=model,
        device="cpu",
        batch_size=kwargs.pop("batch_size", 4),
        **kwargs,
    )


def test_capability_resolver_formats_query_and_document_by_family():
    e5 = resolve_embedding_profile("intfloat/multilingual-e5-small")
    assert e5.format("xin chao", "query") == "query: xin chao"
    assert e5.format("xin chao", "document") == "passage: xin chao"

    instructed = resolve_embedding_profile("intfloat/multilingual-e5-large-instruct")
    assert instructed.format("query", "query").startswith("Instruct:")
    assert instructed.format("passage", "document") == "passage"

    assert resolve_embedding_profile("BAAI/bge-m3").format("raw", "query") == "raw"
    assert resolve_embedding_profile("BAAI/bge-small-en-v1.5").format(
        "needle", "query"
    ).startswith("Represent this sentence")
    qwen = resolve_embedding_profile("Qwen/Qwen3-Embedding-0.6B")
    assert "Query:cau hoi" in qwen.format("cau hoi", "query")
    assert qwen.format("tai lieu", "document") == "tai lieu"
    assert resolve_embedding_profile("sentence-transformers/all-MiniLM-L6-v2").format(
        "raw", "query"
    ) == "raw"


def test_unknown_model_has_validated_explicit_template_override():
    profile = resolve_embedding_profile(
        "private/new-model",
        EmbeddingProfileOverrides(
            family="private",
            query_template="Q={text}",
            document_template="D={text}",
            normalize=False,
        ),
    )
    assert profile.format("x", "query") == "Q=x"
    assert profile.format("x", "document") == "D=x"
    with pytest.raises(CapabilityResolutionError, match="must contain"):
        resolve_embedding_profile(
            "private/new-model",
            EmbeddingProfileOverrides(query_template="missing-placeholder"),
        )


def test_embedding_engine_keeps_query_document_semantics_and_order():
    model = FakeEmbeddingModel()
    engine = make_engine(model=model, normalize=False, batch_size=2)
    documents = engine.embed_documents(["doc1", "doc2", "doc3"])
    query = engine.embed_query("query9")
    assert documents[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert query[0] == 9.0
    encoded_texts = [text for texts, _ in model.calls for text in texts]
    assert encoded_texts[:3] == ["passage: doc1", "passage: doc2", "passage: doc3"]
    assert encoded_texts[-1] == "query: query9"


def test_empty_single_many_and_progress_contracts():
    engine = make_engine(normalize=False, batch_size=3)
    assert engine.embed_documents([]).shape == (0, 3)
    assert engine.embed_documents(["doc1"]).shape == (1, 3)
    progress = []
    result = engine.embed_documents(
        [f"doc{index}" for index in range(1, 9)], progress_callback=progress.append
    )
    assert result.shape == (8, 3)
    assert progress[-1] == 1.0
    assert progress == sorted(progress)


def test_oom_retries_same_slice_with_smaller_batches_without_reordering():
    model = FakeEmbeddingModel(oom_above=2)
    engine = make_engine(
        model=model, normalize=False, batch_size=4, safety_margin=1.0, auto_batch=True
    )
    vectors = engine.embed_documents([f"doc{index}" for index in range(1, 7)])
    assert vectors[:, 0].tolist() == [1, 2, 3, 4, 5, 6]
    assert [len(texts) for texts, _ in model.calls] == [4, 2, 2, 2]
    assert engine.current_batch_size == 2


def test_oom_at_one_item_is_explicit():
    engine = make_engine(model=FakeEmbeddingModel(oom_above=0), batch_size=1)
    with pytest.raises(EmbeddingOOMError, match="retry policy exhausted"):
        engine.embed_documents(["doc1"])


def test_dimension_normalization_and_resolved_diagnostics():
    engine = make_engine(model=FakeEmbeddingModel(dimension=3), normalize=True)
    vector = engine.embed_query("query2")
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert engine.embedding_dim == 3
    assert engine.identity.dimension == 3
    assert engine.identity.normalized is True
    assert engine.resolved_config["dimension_source"].startswith("model.")
    assert engine.resolved_config["max_length"] == 128
    assert len(engine.resolved_config["profile_hash"]) == 64


def test_concrete_loaded_checkpoint_revision_is_exposed():
    model = FakeEmbeddingModel()
    model.config = type("Config", (), {"_commit_hash": "embedding-commit"})()
    engine = make_engine(model=model)
    assert engine.identity.revision == "embedding-commit"
    assert engine.resolved_config["revision"] == "embedding-commit"


def test_dimension_probe_is_used_when_metadata_is_absent():
    class ProbeModel(FakeEmbeddingModel):
        get_sentence_embedding_dimension = None

    engine = make_engine(model=ProbeModel(dimension=4), normalize=False)
    assert engine.embedding_dim == 4
    assert engine.dimension_source == "embedding probe"


def test_dimension_and_max_length_mismatches_fail_fast():
    with pytest.raises(CapabilityResolutionError, match="does not match"):
        make_engine(model=FakeEmbeddingModel(dimension=3), dimension=4)
    with pytest.raises(CapabilityResolutionError, match="exceeds"):
        make_engine(model=FakeEmbeddingModel(), max_length=256)
    assert valid_model_max_length(10**30) is None


def test_configured_max_length_fallback_is_visible_and_only_used_when_needed():
    class SentinelModel(FakeEmbeddingModel):
        max_seq_length = 10**30

    engine = make_engine(model=SentinelModel(), fallback_max_length=384)
    assert engine.max_length == 384
    assert engine.max_length_source == "configured fallback"


def test_remote_code_and_local_only_are_explicit_loader_inputs():
    with pytest.raises(CapabilityResolutionError, match="explicit trust_remote_code"):
        make_engine(model_name="nomic-ai/nomic-embed-text-v1.5")

    captured = {}

    def loader(name, **kwargs):
        captured.update(name=name, **kwargs)
        return FakeEmbeddingModel()

    engine = EmbeddingEngine(
        "nomic-ai/nomic-embed-text-v1.5",
        device="cpu",
        trust_remote_code=True,
        local_files_only=True,
        model_loader=loader,
    )
    assert captured["trust_remote_code"] is True
    assert captured["local_files_only"] is True
    assert engine.resolved_config["trust_remote_code"] is True


def test_adapter_factory_and_batch_query_fallback_are_model_independent():
    engine = make_engine(normalize=False)
    adapter = adapter_for(engine)
    assert type(adapter).__name__ == "E5EmbeddingAdapter"
    assert adapter.embed_queries_batch(["query1", "query2"]).shape == (2, 3)

    class MinimalEngine:
        model_name = "generic/model"
        embedding_dim = 2

        def embed_documents(self, texts):
            return [[1, 0] for _ in texts]

        def embed_query(self, text):
            return [len(text), 0]

    generic = EngineEmbeddingAdapter(MinimalEngine())
    assert generic.embed_queries_batch(["a", "bbb"]) == [[1, 0], [3, 0]]


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="CUDA is not available in this environment"
)
def test_cuda_device_path_when_available():
    model = FakeEmbeddingModel()
    model.device = "cuda"
    engine = EmbeddingEngine("generic/model", model=model, device="cuda", normalize=False)
    assert engine.device == "cuda"
