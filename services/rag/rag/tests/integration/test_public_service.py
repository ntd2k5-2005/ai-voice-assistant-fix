from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest

from rag.config import (
    ChunkingConfig,
    EmbeddingConfig,
    FusionConfig,
    LifecycleConfig,
    ParserConfig,
    PathsConfig,
    RagConfig,
    RetrievalConfig,
)
from rag.contracts import IngestionResult
from rag.lifecycle import DocumentRegistry
from rag.pipeline import create_canonical_pipeline
from rag.retrieval import SQLiteBM25Retriever
from rag.service import RagModule
from rag.errors import SourceTooLargeError
from rag.storage.qdrant import QdrantStore


class PublicApiEmbedder:
    embedding_dim = 2
    model_name = "tests/public-api"
    normalize = True

    def __init__(self):
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts):
        values = list(texts)
        self.document_calls.append(values)
        return [[1.0, 0.0] for _ in values]

    def embed_query(self, query):
        self.query_calls.append(query)
        return [1.0, 0.0]


def build_module(tmp_path, *, config=None, recover=True):
    config = config or RagConfig()
    embedder = PublicApiEmbedder()
    dense = QdrantStore(
        "public-service",
        2,
        str(tmp_path / "qdrant"),
        embedding_model=embedder.model_name,
        normalization=True,
    )
    sparse = SQLiteBM25Retriever(tmp_path / "sparse.db")
    registry = DocumentRegistry(tmp_path / "registry.db")
    module = RagModule.from_components(
        embedder=embedder,
        dense_store=dense,
        sparse_store=sparse,
        registry=registry,
        config=config,
        recover=recover,
    )
    return module, embedder, dense, sparse, registry


class UploadedBytes(BytesIO):
    def __init__(self, payload: bytes, name: str = "upload.md", mime: str = "text/markdown"):
        super().__init__(payload)
        self.name = name
        self.type = mime


def test_public_intake_accepts_bounded_file_like_and_restores_cursor(tmp_path):
    module, _, _, _, _ = build_module(tmp_path)
    uploaded = UploadedBytes("# Tệp tải lên\n\nNội dung hồng ngọc.".encode("utf-8"))
    uploaded.seek(5)
    try:
        result = module.knowledge.ingest(
            uploaded,
            document_id="uploaded-guide",
            include_trace=True,
        )
        assert result.chunk_count > 0
        assert uploaded.tell() == 5
        source = result.diagnostics["trace"]["source"]
        assert source["file_name"] == "upload.md"
        assert source["mime_type"] == "text/markdown"
        assert module.retrieve("hồng ngọc", mode="sparse", top_k=1).candidates
    finally:
        module.close()


def test_public_file_like_intake_rejects_oversize_and_missing_name(tmp_path):
    config = RagConfig(parser=ParserConfig(max_file_bytes=8))
    module, _, _, _, _ = build_module(tmp_path, config=config)
    try:
        with pytest.raises(SourceTooLargeError):
            module.knowledge.ingest(
                UploadedBytes(b"012345678"), document_id="too-large"
            )
        with pytest.raises(ValueError, match="file_name"):
            module.knowledge.ingest(BytesIO(b"hello"), document_id="missing-name")
    finally:
        module.close()


def test_getvalue_only_source_requires_preflightable_size(tmp_path):
    class UnsafeGetValueOnly:
        name = "unsafe.txt"

        def getvalue(self):
            raise AssertionError("unbounded getvalue must not be called")

    module, _, _, _, _ = build_module(tmp_path)
    try:
        with pytest.raises(TypeError, match="read can be bounded"):
            module.knowledge.ingest(
                UnsafeGetValueOnly(),
                document_id="unsafe-upload",
            )
    finally:
        module.close()


def test_public_knowledge_service_add_unchanged_update_reindex_delete(tmp_path):
    module, _, dense, sparse, registry = build_module(tmp_path)
    first_text = "Kiến thức ban đầu chứa từ khóa hồngngọcđỏ cho kiểm thử."
    updated_text = "Kiến thức cập nhật chứa từ khóa lụcbíchxanh cho kiểm thử."
    try:
        added = module.knowledge.ingest(
            first_text,
            source_type="text",
            document_id="guide",
        )
        assert isinstance(added, IngestionResult)
        assert (added.action, added.version) == ("add", 1)
        assert added.chunk_count > 0

        unchanged = module.knowledge.update(
            first_text,
            source_type="text",
            document_id="guide",
        )
        assert (unchanged.action, unchanged.version) == ("unchanged", 1)

        updated = module.knowledge.update(
            updated_text,
            source_type="text",
            document_id="guide",
        )
        assert (updated.action, updated.version) == ("update", 2)
        assert not module.retrieve("hồngngọcđỏ", mode="sparse").candidates
        assert module.retrieve("lụcbíchxanh", mode="sparse").candidates[0].document_id == "guide"

        reindexed = module.knowledge.reindex(
            updated_text,
            source_type="text",
            document_id="guide",
        )
        assert (reindexed.action, reindexed.version) == ("reindex", 3)
        inspected = module.knowledge.inspect("guide")
        assert inspected["status"] == "active"
        assert inspected["version"] == 3

        deleted = module.knowledge.delete("guide")
        assert deleted["action"] == "delete"
        assert registry.get("guide")["status"] == "deleted"
        assert dense.count_document("guide") == 0
        assert sparse.count_document("guide") == 0
        assert not module.retrieve("lụcbíchxanh", mode="sparse").candidates
    finally:
        module.close()


def test_public_sparse_retrieval_never_embeds_the_query(tmp_path):
    module, embedder, _, _, _ = build_module(tmp_path)
    try:
        module.knowledge.ingest(
            "Tài liệu từ vựng có khóa chính xác là quang phổ.",
            source_type="text",
            document_id="lexical",
        )
        embedder.query_calls.clear()
        result = module.retrieve("quang phổ", mode="sparse", top_k=1, candidate_k=2)
        assert result.candidates
        assert result.mode == "sparse"
        assert embedder.query_calls == []
    finally:
        module.close()


def test_from_components_propagates_central_runtime_config(tmp_path):
    config = RagConfig(
        chunking=ChunkingConfig(
            max_tokens=40,
            overlap_tokens=5,
            reserved_tokens=3,
            min_chunk_tokens=1,
        ),
        fusion=FusionConfig(
            strategy="weighted",
            dense_weight=0.7,
            sparse_weight=0.3,
            rrf_k=17,
        ),
        retrieval=RetrievalConfig(
            top_k=2,
            candidate_k=7,
            max_query_characters=321,
            parallel_hybrid=False,
            allow_sparse_rerank=False,
        ),
        lifecycle=LifecycleConfig(
            recover_on_startup=True,
            cleanup_old_versions=False,
            lock_timeout_seconds=2.0,
        ),
    )
    module, _, _, _, _ = build_module(tmp_path, config=config)
    try:
        pipeline = module.pipeline
        engine = pipeline.retrieval_engine
        assert module.config is config
        assert pipeline.parser_registry.config is config.parser
        assert pipeline.chunker.max_tokens == 40
        assert pipeline.chunker.reserved_tokens == 3
        assert pipeline.chunker.overlap_tokens == 5
        assert engine.fusion.dense_weight == 0.7
        assert engine.fusion.sparse_weight == 0.3
        assert engine.max_query_chars == 321
        assert engine.parallel_branches is False
        assert engine.allow_sparse_rerank is False
        assert pipeline.lifecycle.cleanup_old_versions is False
    finally:
        module.close()


def test_from_components_runs_startup_recovery_before_use(tmp_path):
    embedder = PublicApiEmbedder()
    dense = QdrantStore(
        "startup-recovery",
        2,
        str(tmp_path / "qdrant"),
        embedding_model=embedder.model_name,
        normalization=True,
    )
    sparse = SQLiteBM25Retriever(tmp_path / "sparse.db")
    registry = DocumentRegistry(tmp_path / "registry.db")
    registry.begin_ingest(
        "orphan",
        "orphan.txt",
        "interrupted content",
        operation_id="interrupted-operation",
    )
    assert registry.pending()

    module = RagModule.from_components(
        embedder=embedder,
        dense_store=dense,
        sparse_store=sparse,
        registry=registry,
        recover=True,
    )
    try:
        assert registry.pending() == []
        recovered = registry.get("orphan")
        assert recovered["status"] == "failed"
        assert "recovered interrupted ingestion" in recovered["last_error"]
        assert module.knowledge.recover() == []
    finally:
        module.close()


def test_canonical_factory_propagates_model_and_path_config(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class FakeEmbeddingEngine:
        embedding_dim = 7
        max_length = 123
        normalize = False
        model = SimpleNamespace(tokenizer=None)

        def __init__(self, **kwargs):
            captured["embedding"] = kwargs
            self.model_name = kwargs["model_name"]

        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 6 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 6

        def close(self):
            pass

    class FakeDenseStore:
        dim = 7

        def __init__(self, collection_name, dim, persist_dir, **kwargs):
            captured["dense"] = {
                "collection_name": collection_name,
                "dim": dim,
                "persist_dir": persist_dir,
                **kwargs,
            }

        def ensure_embedding_identity(self, model_name, dimension, normalized=None):
            captured.setdefault("identities", []).append(
                (model_name, dimension, normalized)
            )

        def bind_registry(self, registry):
            self.registry = registry

        def search(self, *args, **kwargs):
            return []

        def close(self):
            pass

    monkeypatch.setattr(
        "rag.Embeddings.embeddings.EmbeddingEngine", FakeEmbeddingEngine
    )
    monkeypatch.setattr(
        "rag.Embeddings.factory.adapter_for",
        lambda engine, **_kwargs: engine,
    )
    monkeypatch.setattr("rag.storage.qdrant.QdrantStore", FakeDenseStore)

    config = RagConfig(
        paths=PathsConfig(
            qdrant=tmp_path / "configured-qdrant",
            sparse=tmp_path / "configured-sparse.db",
            registry=tmp_path / "configured-registry.db",
            benchmark_output=tmp_path / "benchmarks",
        ),
        embedding=EmbeddingConfig(
            model_name="configured/model",
            revision="commit-123",
            device="cpu",
            dtype="float32",
            batch_size=9,
            max_length=123,
            dimension=7,
            normalize=False,
            local_files_only=True,
            trust_remote_code=False,
            oom_retries=3,
            batch_safety_margin=0.6,
        ),
        lifecycle=LifecycleConfig(recover_on_startup=False),
    )
    pipeline = create_canonical_pipeline(config)
    try:
        embedding = captured["embedding"]
        assert embedding["model_name"] == "configured/model"
        assert embedding["revision"] == "commit-123"
        assert embedding["device"] == "cpu"
        assert embedding["batch_size"] == 9
        assert embedding["trust_remote_code"] is False
        assert embedding["local_files_only"] is True
        assert embedding["max_oom_retries"] == 3
        assert embedding["safety_margin"] == 0.6
        dense_config = captured["dense"]
        assert dense_config["persist_dir"] == str(tmp_path / "configured-qdrant")
        assert dense_config["dim"] == 7
        assert captured["identities"][0] == ("configured/model", 7, False)
        assert pipeline.registry.path == str(tmp_path / "configured-registry.db")
        assert pipeline.chunker.max_tokens == 123
    finally:
        pipeline.close()
