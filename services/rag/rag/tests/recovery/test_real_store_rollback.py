import pytest

from rag.contracts import MetadataFilter
from rag.lifecycle import DocumentRegistry
from rag.pipeline import RetrievalPipeline
from rag.storage.fts5 import SQLiteBM25Retriever
from rag.storage.qdrant import QdrantStore


class Embedder:
    embedding_dim = 2
    model_name = "real-store-recovery-embedder"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query):
        return [1.0, 0.0]


class FailingFts:
    def __init__(self, index):
        self.index = index
        self.fail_upsert = False

    def __getattr__(self, name):
        return getattr(self.index, name)

    def stage_chunks(self, chunks, *, operation_id):
        if self.fail_upsert:
            raise RuntimeError("injected FTS5 write failure")
        self.index.stage_chunks(chunks, operation_id=operation_id)

    def delete_document(self, *args, **kwargs):
        return self.index.delete_document(*args, **kwargs)

    def search(self, *args, **kwargs):
        return self.index.search(*args, **kwargs)

    def close(self):
        self.index.close()


def test_real_qdrant_is_compensated_when_fts5_write_fails(tmp_path):
    store = QdrantStore("rollback", 2, str(tmp_path / "qdrant"), embedding_model="real-store-recovery-embedder")
    registry = DocumentRegistry(tmp_path / "registry.db")
    sparse = FailingFts(SQLiteBM25Retriever(tmp_path / "fts.db"))
    pipeline = RetrievalPipeline(
        Embedder(),
        store,
        sparse_store=sparse,
        registry=registry,
    )
    try:
        pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "old searchable knowledge"}])

        sparse.fail_upsert = True
        with pytest.raises(RuntimeError, match="FTS5 write failure"):
            pipeline.ingest([{"document_id": "doc", "source": "doc.txt", "text": "new searchable knowledge"}])

        assert registry.get("doc")["status"] == "active"
        assert registry.get("doc")["version"] == 1
        dense = store.search([1.0, 0.0], top_k=10)
        assert dense and {item["metadata"]["version"] for item in dense} == {1}
        assert sparse.search("old", top_k=1, filters=MetadataFilter("version", "=", 1))
        assert not sparse.search("new", top_k=1)
    finally:
        pipeline.close()
