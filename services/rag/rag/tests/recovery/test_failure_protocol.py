import pytest

from rag.lifecycle import DocumentRegistry, RecoveryManager
from rag.pipeline import RetrievalPipeline
from rag.storage.fts5 import SQLiteBM25Retriever
from rag.storage.qdrant import QdrantStore


class Embedder:
    embedding_dim = 2
    model_name = "tests/failure-protocol"
    normalize = True

    def __init__(self, fail=False):
        self.fail = fail

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("embedding failure")
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class ToggleDense:
    """Failure-capable wrapper that retains the full Qdrant lifecycle API."""

    def __init__(self, path):
        self.inner = QdrantStore(
            "failure-protocol",
            2,
            str(path),
            embedding_model=Embedder.model_name,
            normalization=True,
        )
        self.fail_upsert = False
        self.fail_delete = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def stage_chunks(self, chunks, *, operation_id):
        if self.fail_upsert:
            raise RuntimeError("dense upsert failure")
        return self.inner.stage_chunks(chunks, operation_id=operation_id)

    def delete_document(self, document_id, keep_version=None, only_version=None):
        if self.fail_delete:
            raise RuntimeError("dense delete failure")
        return self.inner.delete_document(
            document_id,
            keep_version=keep_version,
            only_version=only_version,
        )

    def close(self):
        self.inner.close()


class ToggleSparse:
    def __init__(self, path):
        self.inner = SQLiteBM25Retriever(path)
        self.fail_upsert = False
        self.fail_delete = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def stage_chunks(self, chunks, *, operation_id):
        if self.fail_upsert:
            raise RuntimeError("sparse upsert failure")
        return self.inner.stage_chunks(chunks, operation_id=operation_id)

    def delete_document(self, *args, **kwargs):
        if self.fail_delete:
            raise RuntimeError("sparse delete failure")
        return self.inner.delete_document(*args, **kwargs)

    def close(self):
        self.inner.close()


def build_pipeline(tmp_path, embedder=None):
    registry = DocumentRegistry(tmp_path / "registry.db")
    dense = ToggleDense(tmp_path / "qdrant")
    sparse = ToggleSparse(tmp_path / "fts.db")
    pipeline = RetrievalPipeline(
        embedder or Embedder(),
        dense,
        sparse_store=sparse,
        registry=registry,
    )
    return pipeline, registry, dense, sparse


def stored_versions(store, document_id):
    return {
        manifest.version
        for version in range(1, 4)
        for manifest in store.get_manifest(document_id, version).values()
    }


def test_embedding_failure_leaves_new_document_failed(tmp_path):
    pipeline, registry, dense, sparse = build_pipeline(tmp_path, Embedder(fail=True))
    try:
        with pytest.raises(RuntimeError, match="embedding failure"):
            pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "content"}])
        assert registry.get("doc")["status"] == "failed"
        assert dense.count_document("doc") == 0
        assert sparse.count_document("doc") == 0
    finally:
        pipeline.close()


def test_dense_upsert_failure_keeps_previous_active_version(tmp_path):
    pipeline, registry, dense, sparse = build_pipeline(tmp_path)
    try:
        pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "old knowledge"}])
        dense.fail_upsert = True
        with pytest.raises(RuntimeError, match="dense upsert failure"):
            pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "new knowledge"}])
        current = registry.get("doc")
        assert current["status"] == "active" and current["version"] == 1
        assert stored_versions(dense, "doc") == {1}
    finally:
        pipeline.close()


def test_dense_or_sparse_failure_keeps_previous_active_version(tmp_path):
    pipeline, registry, dense, sparse = build_pipeline(tmp_path)
    try:
        pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "old knowledge"}])
        sparse.fail_upsert = True
        with pytest.raises(RuntimeError, match="sparse upsert failure"):
            pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "new knowledge"}])
        current = registry.get("doc")
        assert current["status"] == "active" and current["version"] == 1
        assert stored_versions(dense, "doc") == {1}
        assert sparse.inner.search("old", top_k=1)
    finally:
        pipeline.close()


def test_registry_activation_failure_removes_pending_version(tmp_path):
    pipeline, registry, dense, sparse = build_pipeline(tmp_path)
    try:
        pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "old knowledge"}])
        original_commit = registry.commit

        def fail_commit(*args, **kwargs):
            raise RuntimeError("registry activation failure")

        registry.commit = fail_commit
        with pytest.raises(RuntimeError, match="registry activation failure"):
            pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "new knowledge"}])
        registry.commit = original_commit
        current = registry.get("doc")
        assert current["status"] == "active" and current["version"] == 1
        assert stored_versions(dense, "doc") == {1}
        assert not sparse.inner.search("new", top_k=1)
    finally:
        registry.commit = original_commit
        pipeline.close()


def test_cleanup_failure_keeps_new_active_and_recovery_is_idempotent(tmp_path):
    pipeline, registry, dense, sparse = build_pipeline(tmp_path)
    try:
        pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "old knowledge"}])
        dense.fail_delete = True
        pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "new knowledge"}])
        current = registry.get("doc")
        assert current["status"] == "active" and current["version"] == 2
        assert current["cleanup_status"] == "pending"
        assert "doc" in pipeline.last_ingestion_diagnostics["cleanup_pending"]

        dense.fail_delete = False
        recovery = RecoveryManager(registry, [dense, sparse])
        assert recovery.recover() == ["doc"]
        assert recovery.recover() == []
        current = registry.get("doc")
        assert current["version"] == 2 and current["cleanup_status"] is None
        assert stored_versions(dense, "doc") == {2}
        assert sparse.inner.search("new", top_k=1)
    finally:
        pipeline.close()
