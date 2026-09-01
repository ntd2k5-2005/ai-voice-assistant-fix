from rag.pipeline import RetrievalPipeline


class FakeRegistry:
    def inspect(self, document_id=None):
        rows = {
            "active-doc": {
                "document_id": "active-doc",
                "status": "active",
                "version": 2,
            },
            "deleted-doc": {
                "document_id": "deleted-doc",
                "status": "deleted",
                "version": 1,
            },
        }
        if document_id is not None:
            value = rows.get(document_id)
            return dict(value) if value else None
        return [dict(value) for value in rows.values()]

    def visible_version(self, document_id):
        return 2 if document_id == "active-doc" else None


class FakeStore:
    def count_document(self, document_id, *, version=None):
        assert document_id == "active-doc"
        assert version == 2
        return 133


def make_pipeline():
    pipeline = object.__new__(RetrievalPipeline)
    pipeline.registry = FakeRegistry()
    pipeline.store = FakeStore()
    return pipeline


def test_inspect_can_add_chunk_counts_without_loading_chunks():
    pipeline = make_pipeline()
    rows = pipeline.inspect(include_chunk_counts=True)
    by_id = {row["document_id"]: row for row in rows}
    assert by_id["active-doc"]["chunk_count"] == 133
    assert by_id["deleted-doc"]["chunk_count"] == 0


def test_single_document_count_is_available_for_api_detail():
    pipeline = make_pipeline()
    row = pipeline.inspect("active-doc", include_chunk_counts=True)
    assert row["chunk_count"] == 133
