from __future__ import annotations

import multiprocessing
from pathlib import Path
from queue import Empty

import pytest

from rag.contracts import MetadataFilter
from rag.errors import ConcurrentUpdateError, StorageConsistencyError
from rag.lifecycle import DocumentRegistry, StorageLifecycleCoordinator
from rag.storage import QdrantStore, SQLiteBM25Retriever
from rag.storage.identity import ChunkIdentityError


class SimulatedProcessCrash(BaseException):
    """Bypass normal Exception compensation to exercise startup recovery."""


def _chunk(document_id: str, version: int, text: str, suffix: str = "0"):
    return {
        "chunk_id": f"{document_id}:v{version}:{suffix}",
        "document_id": document_id,
        "version": version,
        "text": text,
        "vector": [1.0, 0.0],
        "metadata": {
            "document_id": document_id,
            "version": version,
            "source": f"{document_id}.txt",
            "section": "body",
        },
    }


def _open_protocol(
    root: Path, failure_injector=None, *, cleanup_old_versions: bool = True
):
    registry = DocumentRegistry(root / "registry.db")
    dense = QdrantStore(
        "lifecycle",
        2,
        str(root / "qdrant"),
        embedding_model="lifecycle-test-model",
        normalization=True,
    )
    sparse = SQLiteBM25Retriever(root / "sparse.db")
    coordinator = StorageLifecycleCoordinator(
        registry,
        dense,
        sparse,
        failure_injector=failure_injector,
        cleanup_old_versions=cleanup_old_versions,
    )
    return registry, dense, sparse, coordinator


def _close_protocol(registry, dense, sparse):
    dense.close()
    sparse.close()
    registry.close()


@pytest.mark.parametrize(
    "phase",
    [
        "registry_begin",
        "dense_write",
        "dense_verify",
        "sparse_write",
        "sparse_verify",
        "activation",
        "old_version_cleanup",
    ],
)
def test_crash_at_every_protocol_phase_recovers_without_partial_visibility(
    tmp_path, phase
):
    registry, dense, sparse, coordinator = _open_protocol(tmp_path)
    first = coordinator.begin_ingest(
        "doc", "doc.txt", "old searchable knowledge", operation_id="seed"
    )
    coordinator.stage_and_activate(
        first,
        [_chunk("doc", first["pending_version"], "old searchable knowledge")],
        content="old searchable knowledge",
    )

    def crash(selected, _context):
        if selected == phase:
            raise SimulatedProcessCrash(selected)

    coordinator.failure_injector = crash
    if phase == "registry_begin":
        with pytest.raises(SimulatedProcessCrash):
            coordinator.begin_ingest(
                "doc", "doc.txt", "new searchable knowledge", operation_id="update"
            )
    else:
        coordinator.failure_injector = None
        update = coordinator.begin_ingest(
            "doc", "doc.txt", "new searchable knowledge", operation_id="update"
        )
        coordinator.failure_injector = crash
        with pytest.raises(SimulatedProcessCrash):
            coordinator.stage_and_activate(
                update,
                [
                    _chunk(
                        "doc",
                        update["pending_version"],
                        "new searchable knowledge",
                    )
                ],
                content="new searchable knowledge",
            )

    # The registry gate is authoritative even if a crash lands between the two
    # physical store activations.
    expected_before_restart = 2 if phase == "old_version_cleanup" else 1
    dense_hits = dense.search([1.0, 0.0], top_k=10)
    sparse_hits = sparse.search(
        "new" if expected_before_restart == 2 else "old", top_k=10
    )
    assert dense_hits, "dense visibility must not disappear during version switching"
    assert sparse_hits, "sparse visibility must not disappear during version switching"
    assert all(
        hit["metadata"]["version"] == expected_before_restart for hit in dense_hits
    )
    assert all(
        hit.metadata["version"] == expected_before_restart for hit in sparse_hits
    )
    assert not sparse.search(
        "old" if expected_before_restart == 2 else "new", top_k=10
    )
    _close_protocol(registry, dense, sparse)

    registry, dense, sparse, coordinator = _open_protocol(tmp_path)
    assert coordinator.recover() == ["doc"]
    assert coordinator.recover() == []
    expected_after_recovery = 2 if phase == "old_version_cleanup" else 1
    row = registry.get("doc")
    assert row["status"] == "active"
    assert row["version"] == expected_after_recovery
    assert dense.count_document("doc") == 1
    assert sparse.count_document("doc") == 1
    assert dense.verify_version("doc", expected_after_recovery, state="active")
    assert sparse.verify_version("doc", expected_after_recovery, state="active")
    coordinator.verify_parity("doc", expected_after_recovery, state="active")
    _close_protocol(registry, dense, sparse)


def test_normal_failure_compensates_both_stores_and_preserves_old_version(tmp_path):
    registry, dense, sparse, coordinator = _open_protocol(tmp_path)
    seed = coordinator.begin_ingest("doc", "doc.txt", "old", operation_id="seed")
    coordinator.stage_and_activate(
        seed, [_chunk("doc", 1, "old")], content="old"
    )
    update = coordinator.begin_ingest(
        "doc", "doc.txt", "new", operation_id="will-fail"
    )

    def fail(phase, _context):
        if phase == "sparse_verify":
            raise RuntimeError("injected sparse verification failure")

    coordinator.failure_injector = fail
    with pytest.raises(RuntimeError, match="injected sparse verification"):
        coordinator.stage_and_activate(
            update, [_chunk("doc", 2, "new")], content="new"
        )
    assert registry.get("doc")["version"] == 1
    assert registry.get("doc")["status"] == "active"
    assert dense.count_document("doc", version=2) == 0
    assert sparse.count_document("doc", version=2) == 0
    assert dense.verify_version("doc", 1, state="active")
    assert sparse.verify_version("doc", 1, state="active")
    _close_protocol(registry, dense, sparse)


def test_operation_ids_make_ingest_reindex_and_delete_idempotent(tmp_path):
    registry, dense, sparse, coordinator = _open_protocol(tmp_path)
    add = coordinator.begin_ingest(
        "doc", "doc.txt", "same content", operation_id="add-once"
    )
    coordinator.stage_and_activate(
        add, [_chunk("doc", 1, "same content")], content="same content"
    )
    replay = coordinator.begin_ingest(
        "doc", "doc.txt", "same content", operation_id="add-once"
    )
    assert replay["replayed"] and replay["unchanged"] and replay["version"] == 1

    reindex = coordinator.begin_reindex(
        "doc", "doc.txt", "same content", operation_id="reindex-once"
    )
    assert reindex["pending_version"] == 2
    coordinator.stage_and_activate(
        reindex,
        [_chunk("doc", 2, "same content")],
        content="same content",
    )
    reindex_replay = coordinator.begin_reindex(
        "doc", "doc.txt", "same content", operation_id="reindex-once"
    )
    assert reindex_replay["replayed"]
    assert reindex_replay["version"] == 2

    deleted = coordinator.delete("doc", operation_id="delete-once")
    assert deleted["status"] == "deleted"
    delete_replay = coordinator.delete("doc", operation_id="delete-once")
    assert delete_replay["unchanged"] and delete_replay["replayed"]
    assert dense.count_document("doc") == 0
    assert sparse.count_document("doc") == 0
    _close_protocol(registry, dense, sparse)


def test_cleanup_policy_can_retain_only_stale_invisible_old_versions(tmp_path):
    registry, dense, sparse, coordinator = _open_protocol(
        tmp_path, cleanup_old_versions=False
    )
    add = coordinator.begin_ingest("doc", "doc.txt", "old", operation_id="add")
    coordinator.stage_and_activate(
        add, [_chunk("doc", 1, "old")], content="old"
    )
    update = coordinator.begin_ingest(
        "doc", "doc.txt", "new", operation_id="update"
    )
    result = coordinator.stage_and_activate(
        update, [_chunk("doc", 2, "new")], content="new"
    )

    assert not result["cleanup_old_versions"]
    assert result["old_versions_retained"]
    assert not result["cleanup_pending"]
    assert registry.get("doc")["cleanup_status"] is None
    assert dense.count_document("doc") == 2
    assert sparse.count_document("doc") == 2
    coordinator.verify_parity("doc", 1, state="stale", operation_id="add")
    coordinator.verify_parity("doc", 2, state="active", operation_id="update")
    assert {hit["metadata"]["version"] for hit in dense.search([1.0, 0.0], 10)} == {2}
    assert not sparse.search("old", 10)
    assert {hit.metadata["version"] for hit in sparse.search("new", 10)} == {2}
    assert coordinator.recover() == []
    _close_protocol(registry, dense, sparse)


def test_sqlite_timeout_is_propagated_to_connection_and_busy_pragma(tmp_path):
    registry = DocumentRegistry(
        tmp_path / "registry.db", timeout_seconds=0.125
    )
    sparse = SQLiteBM25Retriever(
        tmp_path / "sparse.db", timeout_seconds=0.275
    )
    assert registry.timeout_seconds == 0.125
    assert registry.db.execute("PRAGMA busy_timeout").fetchone()[0] == 125
    assert sparse.timeout_seconds == 0.275
    assert sparse.db.execute("PRAGMA busy_timeout").fetchone()[0] == 275
    registry.close()
    sparse.close()
    with pytest.raises(ValueError, match="timeout_seconds"):
        DocumentRegistry(tmp_path / "invalid.db", timeout_seconds=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        SQLiteBM25Retriever(tmp_path / "invalid-fts.db", timeout_seconds=float("inf"))


def test_interrupted_delete_is_hidden_and_recovered_idempotently(tmp_path):
    registry, dense, sparse, coordinator = _open_protocol(tmp_path)
    add = coordinator.begin_ingest("doc", "doc.txt", "secret", operation_id="add")
    coordinator.stage_and_activate(
        add, [_chunk("doc", 1, "secret")], content="secret"
    )

    def crash(phase, _context):
        if phase == "delete_dense":
            raise SimulatedProcessCrash(phase)

    coordinator.failure_injector = crash
    with pytest.raises(SimulatedProcessCrash):
        coordinator.delete("doc", operation_id="delete")
    assert registry.get("doc")["status"] == "deleting"
    assert not dense.search([1.0, 0.0], top_k=10)
    assert not sparse.search("secret", top_k=10)
    _close_protocol(registry, dense, sparse)

    registry, dense, sparse, coordinator = _open_protocol(tmp_path)
    assert coordinator.recover() == ["doc"]
    assert coordinator.recover() == []
    assert registry.get("doc")["status"] == "deleted"
    assert dense.count_document("doc") == 0
    assert sparse.count_document("doc") == 0
    _close_protocol(registry, dense, sparse)


def _concurrent_begin_worker(path: str, content: str, start, results):
    registry = DocumentRegistry(path)
    try:
        start.wait(10)
        operation = registry.begin_ingest(
            "doc", "doc.txt", content, operation_id=f"op-{content}"
        )
        results.put(("ok", operation["operation_id"], operation["pending_version"]))
    except BaseException as exc:  # result must survive the spawned process
        results.put(("error", type(exc).__name__, str(exc)))
    finally:
        registry.close()


def test_multiprocess_version_allocation_has_one_owner(tmp_path):
    path = str(tmp_path / "registry.db")
    registry = DocumentRegistry(path)
    registry.register("doc", "doc.txt", "base")
    registry.close()

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_begin_worker,
            args=(path, content, start, results),
        )
        for content in ("update-a", "update-b")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    received = []
    for _ in processes:
        try:
            received.append(results.get(timeout=5))
        except Empty:  # pragma: no cover - diagnostics for broken spawn
            pytest.fail("worker did not publish its lifecycle result")
    assert [item[0] for item in received].count("ok") == 1
    errors = [item for item in received if item[0] == "error"]
    assert len(errors) == 1 and errors[0][1] == ConcurrentUpdateError.__name__
    assert next(item for item in received if item[0] == "ok")[2] == 2

    registry = DocumentRegistry(path)
    assert len(registry.pending()) == 1
    from rag.lifecycle import RecoveryManager

    assert RecoveryManager(registry).recover() == ["doc"]
    assert registry.get("doc")["version"] == 1
    registry.close()


def test_dense_and_sparse_metadata_filters_have_the_same_semantics(tmp_path):
    dense = QdrantStore("filters", 2, str(tmp_path / "qdrant"))
    sparse = SQLiteBM25Retriever(tmp_path / "sparse.db")
    chunks = [
        {
            "chunk_id": "a",
            "document_id": "a-doc",
            "version": 1,
            "text": "common token alpha",
            "vector": [1.0, 0.0],
            "metadata": {
                "document_id": "a-doc",
                "version": 1,
                "group": "a",
                "score_value": 1,
                "tags": ["active", "hr"],
                "note": "alphabet",
            },
        },
        {
            "chunk_id": "b",
            "document_id": "b-doc",
            "version": 1,
            "text": "common token beta",
            "vector": [0.9, 0.1],
            "metadata": {
                "document_id": "b-doc",
                "version": 1,
                "group": "b",
                "score_value": 2,
                "tags": ["inactive"],
                "note": "beta",
            },
        },
        {
            "chunk_id": "c",
            "document_id": "c-doc",
            "version": 1,
            "text": "common token gamma",
            "vector": [0.8, 0.2],
            "metadata": {
                "document_id": "c-doc",
                "version": 1,
                "score_value": 3,
                "tags": [],
                "note": "gamma",
            },
        },
    ]
    dense.upsert_chunks(chunks)
    sparse.upsert(chunks)
    cases = [
        (MetadataFilter("group", "!=", "a"), {"b", "c"}),
        (MetadataFilter("tags", "contains", "active"), {"a"}),
        (MetadataFilter("note", "contains", "alp"), {"a"}),
        (MetadataFilter("score_value", ">=", 2), {"b", "c"}),
        (MetadataFilter("group", "in", ["a", "b"]), {"a", "b"}),
        (MetadataFilter("missing", "!=", "x"), {"a", "b", "c"}),
    ]
    for predicate, expected in cases:
        dense_ids = {
            hit["chunk_id"]
            for hit in dense.search(
                [1.0, 0.0], top_k=10, query_filter=predicate
            )
        }
        sparse_ids = {
            hit.chunk_id
            for hit in sparse.search("common token", top_k=10, filters=predicate)
        }
        assert dense_ids == sparse_ids == expected
    dense.close()
    sparse.close()


def test_conflicting_chunk_identity_fails_before_either_store_write(tmp_path):
    dense = QdrantStore("identity", 2, str(tmp_path / "qdrant"))
    sparse = SQLiteBM25Retriever(tmp_path / "sparse.db")
    bad = {
        "chunk_id": "chunk",
        "document_id": "top-level-doc",
        "version": 1,
        "text": "bad identity",
        "vector": [1.0, 0.0],
        "metadata": {"document_id": "other-doc", "version": 1},
    }
    with pytest.raises((ChunkIdentityError, StorageConsistencyError)):
        dense.upsert_chunks([bad])
    with pytest.raises((ChunkIdentityError, StorageConsistencyError)):
        sparse.upsert([bad])
    assert dense.count_document("top-level-doc") == 0
    assert sparse.count_document("top-level-doc") == 0
    dense.close()
    sparse.close()
