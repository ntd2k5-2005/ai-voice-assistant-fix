"""Deterministic in-memory BM25 candidate retrieval."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Optional

from ..contracts import MetadataFilter, RetrievalCandidate
from ..sparse_analyzer import SparseAnalyzer, resolve_sparse_analyzer


def tokenize(text: str) -> list[str]:
    """Compatibility helper backed by the canonical default analyzer."""

    return list(resolve_sparse_analyzer().tokens(text))


def allowed(candidate: RetrievalCandidate, filters) -> bool:
    if isinstance(filters, MetadataFilter):
        filters = [filters]
    if not filters:
        return True
    try:
        predicates = list(filters)
    except TypeError as exc:
        raise TypeError("filters must be MetadataFilter or an iterable of filters") from exc
    if any(not hasattr(predicate, "matches") for predicate in predicates):
        raise TypeError("every sparse filter must implement matches(metadata)")
    return all(predicate.matches(candidate.metadata) for predicate in predicates)


class SparseRetriever:
    """Dependency-free BM25 index used for tests and small benchmarks."""

    def __init__(
        self,
        chunks: Iterable[Mapping[str, Any]] = (),
        k1: float = 1.5,
        b: float = 0.75,
        *,
        analyzer: str | SparseAnalyzer = "unicode61",
    ):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.k1, self.b = k1, b
        self.analyzer = resolve_sparse_analyzer(analyzer)
        self._documents: dict[str, tuple[RetrievalCandidate, list[str]]] = {}
        self._postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self._df: Counter = Counter()
        self._avgdl = 0.0
        self.add(chunks)

    def add(self, chunks: Iterable[Mapping[str, Any] | RetrievalCandidate]) -> None:
        for raw in chunks:
            candidate = (
                RetrievalCandidate(**raw.__dict__)
                if isinstance(raw, RetrievalCandidate)
                else RetrievalCandidate.from_mapping(raw, "sparse")
            )
            terms = list(self.analyzer.tokens(candidate.text))
            # Upserting an empty replacement must remove the stale lexical
            # representation instead of silently retaining its old terms.
            self._documents.pop(candidate.chunk_id, None)
            if terms:
                candidate.retrieval_source = "sparse"
                self._documents[candidate.chunk_id] = (candidate, terms)
        self._rebuild()

    upsert = add

    def _rebuild(self) -> None:
        self._postings.clear()
        self._df.clear()
        for chunk_id, (_, terms) in self._documents.items():
            for term, frequency in Counter(terms).items():
                self._postings[term].append((chunk_id, frequency))
                self._df[term] += 1
        self._avgdl = sum(len(terms) for _, terms in self._documents.values()) / max(len(self._documents), 1)

    def delete_document(self, document_id: str, keep_version=None, only_version=None) -> None:
        for chunk_id, (candidate, _) in list(self._documents.items()):
            if candidate.document_id != str(document_id):
                continue
            version = candidate.metadata.get("version")
            if only_version is not None and version != only_version:
                continue
            if keep_version is not None and version == keep_version:
                continue
            del self._documents[chunk_id]
        self._rebuild()

    def search(self, query: str, top_k: int = 10, filters: Optional[Iterable[MetadataFilter]] = None) -> list[RetrievalCandidate]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            return []
        scores = defaultdict(float)
        total = len(self._documents)
        for term in self.analyzer.tokens(query):
            document_frequency = self._df.get(term, 0)
            if not document_frequency:
                continue
            idf = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
            for chunk_id, term_frequency in self._postings[term]:
                candidate, terms = self._documents[chunk_id]
                if not allowed(candidate, filters):
                    continue
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * len(terms) / max(self._avgdl, 1e-12))
                scores[chunk_id] += idf * term_frequency * (self.k1 + 1) / denominator
        result = []
        for chunk_id, score in scores.items():
            candidate = RetrievalCandidate(**self._documents[chunk_id][0].__dict__)
            candidate.retrieval_score = float(score)
            candidate.sparse_score = float(score)
            candidate.final_score = float(score)
            candidate.retrieval_source = "sparse"
            result.append(candidate)
        return sorted(result, key=lambda item: (-item.retrieval_score, item.chunk_id))[:top_k]


from ..storage.fts5 import SQLiteBM25Retriever


__all__ = ["SparseRetriever", "SQLiteBM25Retriever", "tokenize", "allowed"]
