import numpy as np
import pytest

from rag.processing.chunking import UltimateChunker
from rag.storage.qdrant import QdrantStore


def test_chunk_ids_are_stable_and_qdrant_contract_is_present():
    doc = {"text": "Tiêu đề\n\nNội dung tiếng Việt.", "source": "a.txt", "page": 1}
    first = UltimateChunker(max_chars=100, overlap_chars=10, hard_max_chars=120).chunk_document(doc)
    second = UltimateChunker(max_chars=100, overlap_chars=10, hard_max_chars=120).chunk_document(doc)
    assert first[0]["_id"] == second[0]["_id"]
    assert first[0]["metadata"]["chunk_id"] == first[0]["_id"]
    assert first[0]["metadata"]["chunk_index"] == 0


def test_qdrant_rejects_wrong_dimension(tmp_path):
    store = QdrantStore("test", dim=3, persist_dir=str(tmp_path))
    with pytest.raises(ValueError):
        store.upsert_chunks([{"_id": "x", "text": "x", "vector": np.zeros(2)}])


def test_qdrant_rejects_non_finite_vectors(tmp_path):
    store = QdrantStore("test", dim=2, persist_dir=str(tmp_path))
    with pytest.raises(ValueError):
        store.upsert_chunks([{"_id": "x", "text": "x", "vector": [np.nan, 1]}])
