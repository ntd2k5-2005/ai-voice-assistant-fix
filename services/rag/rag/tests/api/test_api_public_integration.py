from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.config import ChunkingConfig, EmbeddingConfig, PathsConfig, RagConfig, RerankerConfig, RetrievalConfig
from rag.lifecycle import DocumentRegistry
from rag.retrieval import SQLiteBM25Retriever
from rag.service import RagModule
from rag.storage.qdrant import QdrantStore


class DeterministicEmbedder:
    embedding_dim = 2
    model_name = "tests/api"
    normalize = True
    resolved_config = {
        "model_name": model_name,
        "family": "deterministic-test",
        "dimension": 2,
        "device": "cpu",
        "dtype": "float32",
        "max_length": 128,
        "batch_size": 2,
        "normalize": True,
        "revision": "test-revision",
    }

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query):
        return [1.0, 0.0]


class DeterministicReranker:
    model_name = "tests/reranker"
    resolved_config = {
        "model_name": model_name,
        "family": "sequence-classification",
        "adapter": "sequence_classification",
        "score_extraction": "scalar",
        "device": "cpu",
        "dtype": "float32",
        "max_length": 128,
        "batch_size": 2,
        "revision": "test-revision",
    }

    def rerank_candidates(self, candidates, query, top_k):
        reversed_candidates = list(reversed(candidates))
        return [
            (candidate, float(len(reversed_candidates) - index))
            for index, candidate in enumerate(reversed_candidates[:top_k])
        ]


def build_module(tmp_path):
    config = RagConfig(
        paths=PathsConfig(
            qdrant=tmp_path / "qdrant",
            sparse=tmp_path / "sparse.db",
            registry=tmp_path / "registry.db",
            benchmark_output=tmp_path / "benchmarks",
        ),
        embedding=EmbeddingConfig(
            model_name=DeterministicEmbedder.model_name,
            device="cpu",
            dimension=2,
            max_length=128,
            batch_size=2,
            normalize=True,
        ),
        chunking=ChunkingConfig(
            max_tokens=80,
            fallback_max_tokens=80,
            reserved_tokens=8,
            overlap_tokens=8,
            min_chunk_tokens=1,
        ),
        reranker=RerankerConfig(
            model_name=DeterministicReranker.model_name,
            revision=None,
            code_revision=None,
            device="cpu",
            max_length=128,
            batch_size=2,
            trust_remote_code=False,
        ),
        retrieval=RetrievalConfig(top_k=3, candidate_k=6),
    )
    embedder = DeterministicEmbedder()
    dense = QdrantStore(
        "api-integration",
        2,
        str(config.paths.qdrant),
        embedding_model=embedder.model_name,
        normalization=True,
    )
    sparse = SQLiteBM25Retriever(config.paths.sparse)
    registry = DocumentRegistry(config.paths.registry)
    return RagModule.from_components(
        embedder=embedder,
        dense_store=dense,
        sparse_store=sparse,
        registry=registry,
        reranker=DeterministicReranker(),
        config=config,
    )


def test_http_adapter_reaches_real_public_rag_flow(tmp_path):
    module = build_module(tmp_path)
    app = create_app(module_factory=lambda: module)
    with TestClient(app) as client:
        ingest = client.post(
            "/v1/knowledge/documents",
            files={
                "file": (
                    "knowledge.md",
                    "# Giải pháp\n\nCần luyện tập giao tiếp và phản biện.".encode("utf-8"),
                    "text/markdown",
                )
            },
            data={"document_id": "knowledge"},
        )
        assert ingest.status_code == 201, ingest.text
        result = client.post(
            "/v1/retrieval",
            json={"query": "giải pháp giao tiếp", "mode": "hybrid", "rerank": True, "top_k": 1},
        )
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["contexts"]
        assert payload["contexts"][0]["document_id"] == "knowledge"
        assert 0.0 <= payload["contexts"][0]["score"] <= 1.0
