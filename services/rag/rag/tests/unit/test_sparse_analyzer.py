from __future__ import annotations

import pytest

from rag.config import RagConfig, SparseConfig
from rag.retrieval.bm25 import SparseRetriever
from rag.sparse_analyzer import (
    Unicode61Analyzer,
    resolve_sparse_analyzer,
)


def test_unicode61_analysis_is_deterministic_and_preserves_caller_text():
    analyzer = Unicode61Analyzer()
    original = "Thủ đô HÀ NỘI — dữ liệu_2026"

    assert analyzer.tokens(original) == (
        "thu",
        "đo",
        "ha",
        "noi",
        "du",
        "lieu_2026",
    )
    assert analyzer.index_text(original) == "thu đo ha noi du lieu_2026"
    assert original == "Thủ đô HÀ NỘI — dữ liệu_2026"


def test_in_memory_bm25_uses_same_analyzer_for_documents_and_queries():
    retriever = SparseRetriever(
        [
            {
                "chunk_id": "capital",
                "document_id": "vn",
                "text": "Thủ đô của Việt Nam là Hà Nội.",
                "metadata": {"version": 1},
            }
        ]
    )

    hits = retriever.search("THỦ ĐÔ HÀ NỘI", top_k=1)
    assert hits
    assert hits[0].chunk_id == "capital"
    assert hits[0].text == "Thủ đô của Việt Nam là Hà Nội."


def test_sparse_config_exposes_only_real_analyzers():
    config = SparseConfig()
    assert config.analyzer == "unicode61"
    assert config.tokenizer == config.analyzer
    assert RagConfig(sparse=config).snapshot()["sparse"] == {
        "analyzer": "unicode61",
        "query_operator": "or",
    }
    with pytest.raises(ValueError, match="unsupported sparse analyzer"):
        resolve_sparse_analyzer("pretend-vietnamese-segmenter")
    with pytest.raises(ValueError, match="sparse.analyzer"):
        SparseConfig(analyzer="pretend")
