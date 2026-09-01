import json

from rag.contracts import MetadataFilter
from rag.storage.fts5 import SQLiteBM25Retriever


def test_fts5_persists_json_metadata_and_filters_in_sql(tmp_path):
    index = SQLiteBM25Retriever(tmp_path / "fts.db")
    index.upsert([
        {"_id": "a", "document_id": "doc-a", "text": "Lan is an engineer", "metadata": {"version": 2, "sheet_name": "Employees", "row_number": 2, "tags": ["hr", "active"]}},
        {"_id": "b", "document_id": "doc-b", "text": "Lan is a student", "metadata": {"version": 1, "sheet_name": "Students", "row_number": 9, "tags": ["archive"]}},
    ])
    raw = index.db.execute("SELECT metadata_json FROM chunk_metadata WHERE chunk_id='a'").fetchone()[0]
    assert json.loads(raw)["sheet_name"] == "Employees"
    assert "'sheet_name'" not in raw

    rows = index.search("Lan", top_k=1, filters=[MetadataFilter("sheet_name", "=", "Employees"), MetadataFilter("row_number", ">", 1)])
    assert [row.chunk_id for row in rows] == ["a"]
    rows = index.search("Lan", top_k=1, filters=MetadataFilter("tags", "contains", "active"))
    assert [row.chunk_id for row in rows] == ["a"]
    index.close()

    reopened = SQLiteBM25Retriever(tmp_path / "fts.db")
    assert reopened.search("engineer", top_k=1)[0].metadata["sheet_name"] == "Employees"
    reopened.close()


def test_fts5_uses_canonical_analyzer_but_returns_unmodified_original_text(tmp_path):
    original = "Thủ đô của Việt Nam là Hà Nội — dữ liệu 2026."
    index = SQLiteBM25Retriever(tmp_path / "unicode.db", analyzer="unicode61")
    index.upsert(
        [
            {
                "chunk_id": "capital",
                "document_id": "vietnam",
                "text": original,
                "metadata": {"version": 1},
            }
        ]
    )

    stored = index.db.execute(
        """SELECT f.text,m.original_text
             FROM chunks_fts AS f JOIN chunk_metadata AS m
               ON f.chunk_id=m.chunk_id
            WHERE f.chunk_id='capital'"""
    ).fetchone()
    assert stored[0] == "thu đo cua viet nam la ha noi du lieu 2026"
    assert stored[1] == original
    hits = index.search("THỦ ĐÔ HÀ NỘI", top_k=1)
    assert hits
    assert hits[0].chunk_id == "capital"
    assert hits[0].text == original
    assert index.index_identity["analyzer"] == "unicode61"
    index.close()
