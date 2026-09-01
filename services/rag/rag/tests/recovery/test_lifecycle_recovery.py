from rag.lifecycle import DocumentRegistry, RecoveryManager


class FakeStore:
    def __init__(self):
        self.deleted = []

    def delete_document(self, document_id, **kwargs):
        self.deleted.append((document_id, kwargs))


def test_restart_recovery_removes_only_the_pending_version(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.db")
    registry.register("doc", "doc.txt", "old content")
    state = registry.begin_ingest("doc", "doc.txt", "new content")
    assert state["status"] == "indexing"
    assert state["version"] == 1
    assert state["pending_version"] == 2

    store = FakeStore()
    recovered = RecoveryManager(registry, [store]).recover()
    assert recovered == ["doc"]
    assert store.deleted == [("doc", {"only_version": 2})]
    current = registry.get("doc")
    assert current["status"] == "active"
    assert current["version"] == 1
    assert current["hash"] == registry.content_hash("old content")
    registry.close()
