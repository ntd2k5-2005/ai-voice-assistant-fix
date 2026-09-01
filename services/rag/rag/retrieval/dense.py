"""Dense candidate retrieval over the configured embedding and vector store."""
from __future__ import annotations

import inspect
import time

import numpy as np

from ..contracts import RetrievalCandidate


class DenseRetriever:
    def __init__(self, embedder, store):
        self.embedder, self.store = embedder, store
        self.last_embed_latency_ms = 0.0
        self.last_search_latency_ms = 0.0
        self.last_query_vector: np.ndarray | None = None

        ensure_identity = getattr(store, "ensure_embedding_identity", None)
        dimension = getattr(embedder, "embedding_dim", None)
        model_name = getattr(embedder, "model_name", "")
        identity = getattr(embedder, "identity", None)
        if identity is not None:
            dimension = dimension or getattr(identity, "dimension", None)
            model_name = model_name or getattr(identity, "model_name", "")
        if ensure_identity is not None and dimension is not None:
            parameters = inspect.signature(ensure_identity).parameters
            kwargs = {}
            if "normalization" in parameters and identity is not None:
                kwargs["normalization"] = getattr(identity, "normalized", None)
            if "revision" in parameters and identity is not None:
                kwargs["revision"] = getattr(identity, "revision", None)
            if "profile_hash" in parameters and identity is not None:
                kwargs["profile_hash"] = getattr(identity, "profile_hash", None)
            ensure_identity(str(model_name or ""), int(dimension), **kwargs)

    def embed_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        started = time.perf_counter()
        if not hasattr(self.embedder, "embed_query"):
            raise TypeError("dense embedder must implement embed_query()")
        vector = np.asarray(self.embedder.embed_query(query.strip()), dtype=np.float32)
        self.last_embed_latency_ms = (time.perf_counter() - started) * 1000
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise RuntimeError("embed_query() must return one finite one-dimensional vector")
        expected_dimension = getattr(self.store, "dim", None)
        if expected_dimension is not None and vector.shape[0] != int(expected_dimension):
            raise ValueError(
                f"query embedding dimension {vector.shape[0]} does not match dense index "
                f"dimension {expected_dimension}"
            )
        self.last_query_vector = vector.copy()
        return vector

    def search_vector(
        self,
        vector,
        top_k: int = 10,
        filters=None,
        *,
        visibility_snapshot=None,
    ):
        if top_k <= 0:
            return []
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise RuntimeError("dense search vector must be finite and one-dimensional")
        started = time.perf_counter()
        kwargs = {"top_k": top_k, "query_filter": filters}
        if visibility_snapshot is not None:
            parameters = inspect.signature(self.store.search).parameters
            if "visibility_snapshot" not in parameters and not any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                raise TypeError(
                    "dense store cannot enforce a shared lifecycle visibility snapshot"
                )
            kwargs["visibility_snapshot"] = visibility_snapshot
        hits = self.store.search(vector, **kwargs)
        self.last_search_latency_ms = (time.perf_counter() - started) * 1000
        result = []
        for hit in hits:
            candidate = RetrievalCandidate.from_mapping(hit, "dense")
            candidate.dense_score = candidate.retrieval_score
            result.append(candidate)
        return sorted(result, key=lambda item: (-float(item.dense_score), item.chunk_id))[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters=None,
        *,
        visibility_snapshot=None,
        query_vector=None,
    ):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            return []
        vector = self.embed_query(query) if query_vector is None else query_vector
        return self.search_vector(
            vector,
            top_k,
            filters,
            visibility_snapshot=visibility_snapshot,
        )


__all__ = ["DenseRetriever"]
