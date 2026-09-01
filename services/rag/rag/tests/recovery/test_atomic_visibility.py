"""Non-vacuous proofs for the cross-store atomic visibility protocol."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from threading import Event, Thread

import pytest

from rag.lifecycle import DocumentRegistry, StorageLifecycleCoordinator
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import RetrievalEngine
from rag.storage.fts5 import SQLiteBM25Retriever
from rag.storage.qdrant import QdrantStore


class SimulatedProcessCrash(BaseException):
    """Escape normal compensation so restart recovery sees on-disk state."""


_PRECOMMIT_PHASES = {
    "registry_begin",
    "before_stage",
    "dense_write",
    "dense_verify",
    "sparse_write",
    "sparse_verify",
    "verification",
    "dense_ready",
    "sparse_ready",
    "activation",
    "precommit",
}
_POSTCOMMIT_PHASES = {
    "postcommit",
    "old_version_cleanup",
    "dense_cleanup",
    "sparse_cleanup",
}
_ALL_UPDATE_PHASES = tuple(sorted(_PRECOMMIT_PHASES | _POSTCOMMIT_PHASES))


def _chunk(document_id: str, version: int, suffix: str, generation: str):
    return {
        "chunk_id": f"{document_id}:v{version}:{suffix}",
        "document_id": document_id,
        "version": version,
        "text": f"atomicanchor {generation} knowledge part {suffix}",
        "vector": [1.0, 0.0],
        "metadata": {
            "document_id": document_id,
            "version": version,
            "source": f"{document_id}.txt",
            "section": "body",
            "generation": generation,
        },
    }


def _version_chunks(document_id: str, version: int, generation: str):
    return [
        _chunk(document_id, version, "a", generation),
        _chunk(document_id, version, "b", generation),
    ]


def _open(root: Path, failure_injector=None, *, cleanup=True):
    registry = DocumentRegistry(root / "registry.db")
    dense = QdrantStore(
        "atomic-visibility",
        2,
        str(root / "qdrant"),
        embedding_model="atomic-test-model",
        normalization=True,
    )
    sparse = SQLiteBM25Retriever(root / "sparse.db")
    coordinator = StorageLifecycleCoordinator(
        registry,
        dense,
        sparse,
        failure_injector=failure_injector,
        cleanup_old_versions=cleanup,
    )
    return registry, dense, sparse, coordinator


def _close(registry, dense, sparse):
    dense.close()
    sparse.close()
    registry.close()


def _seed(coordinator: StorageLifecycleCoordinator):
    operation = coordinator.begin_ingest(
        "doc", "doc.txt", "old complete content", operation_id="seed"
    )
    coordinator.stage_and_activate(
        operation,
        _version_chunks("doc", 1, "old"),
        content="old complete content",
    )


def _assert_complete_visible_version(
    dense, sparse, expected_version: int, *, visibility_snapshot=None
):
    """Reject empty, partial, mixed, stale, and pending visibility."""

    expected_ids = {
        f"doc:v{expected_version}:a",
        f"doc:v{expected_version}:b",
    }
    dense_hits = dense.search(
        [1.0, 0.0],
        top_k=20,
        visibility_snapshot=visibility_snapshot,
    )
    sparse_hits = sparse.search(
        "atomicanchor",
        top_k=20,
        visibility_snapshot=visibility_snapshot,
    )

    # These assertions are intentionally first: an empty list must never make
    # the later all()/set assertions pass vacuously.
    assert dense_hits, "dense retrieval unexpectedly exposed no committed version"
    assert sparse_hits, "sparse retrieval unexpectedly exposed no committed version"
    assert {hit["chunk_id"] for hit in dense_hits} == expected_ids
    assert {hit.chunk_id for hit in sparse_hits} == expected_ids
    assert {int(hit["metadata"]["version"]) for hit in dense_hits} == {
        expected_version
    }
    assert {int(hit.metadata["version"]) for hit in sparse_hits} == {
        expected_version
    }
    assert all(
        hit["metadata"]["lifecycle_state"] == "active" for hit in dense_hits
    )
    assert all(hit.metadata["lifecycle_state"] == "active" for hit in sparse_hits)


def _assert_no_vector_keys(value):
    if isinstance(value, Mapping):
        assert not ({"vector", "vectors"} & set(value))
        for nested in value.values():
            _assert_no_vector_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_vector_keys(nested)


@pytest.mark.parametrize("observed_phase", _ALL_UPDATE_PHASES)
def test_every_update_checkpoint_has_one_complete_visible_version(
    tmp_path, observed_phase
):
    """Observe live reads at every protocol checkpoint without crashing it."""

    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        _seed(coordinator)
        observed = []

        def inspect(phase, _context):
            if phase != observed_phase:
                return
            expected = 1 if phase in _PRECOMMIT_PHASES else 2
            _assert_complete_visible_version(dense, sparse, expected)
            observed.append(phase)

        coordinator.failure_injector = inspect
        update = coordinator.begin_ingest(
            "doc", "doc.txt", "new complete content", operation_id="update"
        )
        coordinator.stage_and_activate(
            update,
            _version_chunks("doc", 2, "new"),
            content="new complete content",
        )
        assert observed == [observed_phase]
        _assert_complete_visible_version(dense, sparse, 2)
    finally:
        _close(registry, dense, sparse)


@pytest.mark.parametrize("crash_phase", _ALL_UPDATE_PHASES)
def test_crash_at_every_update_checkpoint_recovers_one_complete_version(
    tmp_path, crash_phase
):
    registry, dense, sparse, coordinator = _open(tmp_path)
    _seed(coordinator)

    def crash(phase, _context):
        if phase == crash_phase:
            raise SimulatedProcessCrash(phase)

    coordinator.failure_injector = crash
    try:
        if crash_phase == "registry_begin":
            with pytest.raises(SimulatedProcessCrash):
                coordinator.begin_ingest(
                    "doc",
                    "doc.txt",
                    "new complete content",
                    operation_id="update",
                )
        else:
            coordinator.failure_injector = None
            update = coordinator.begin_ingest(
                "doc", "doc.txt", "new complete content", operation_id="update"
            )
            coordinator.failure_injector = crash
            with pytest.raises(SimulatedProcessCrash):
                coordinator.stage_and_activate(
                    update,
                    _version_chunks("doc", 2, "new"),
                    content="new complete content",
                )
        expected = 1 if crash_phase in _PRECOMMIT_PHASES else 2
        _assert_complete_visible_version(dense, sparse, expected)
    finally:
        _close(registry, dense, sparse)

    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        assert coordinator.recover() == ["doc"]
        assert coordinator.recover() == []
        expected = 1 if crash_phase in _PRECOMMIT_PHASES else 2
        _assert_complete_visible_version(dense, sparse, expected)
        assert dense.count_document("doc") == 2
        assert sparse.count_document("doc") == 2
        assert dense.verify_version("doc", expected, state="active")
        assert sparse.verify_version("doc", expected, state="active")
        coordinator.verify_parity("doc", expected, state="active")
    finally:
        _close(registry, dense, sparse)


def test_cleanup_failure_keeps_new_version_visible_and_restart_repairs_it(tmp_path):
    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        _seed(coordinator)

        def fail_cleanup(phase, _context):
            if phase == "old_version_cleanup":
                raise RuntimeError("injected cleanup failure")

        coordinator.failure_injector = fail_cleanup
        update = coordinator.begin_ingest(
            "doc", "doc.txt", "new complete content", operation_id="update"
        )
        result = coordinator.stage_and_activate(
            update,
            _version_chunks("doc", 2, "new"),
            content="new complete content",
        )
        assert result["cleanup_pending"]
        assert "injected cleanup failure" in result["cleanup_error"]
        assert registry.get("doc")["version"] == 2
        _assert_complete_visible_version(dense, sparse, 2)
    finally:
        _close(registry, dense, sparse)

    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        assert coordinator.recover() == ["doc"]
        assert coordinator.recover() == []
        _assert_complete_visible_version(dense, sparse, 2)
        assert dense.count_document("doc") == 2
        assert sparse.count_document("doc") == 2
    finally:
        _close(registry, dense, sparse)


def test_partial_cross_store_cleanup_never_changes_logical_visibility(tmp_path):
    registry, dense, sparse, coordinator = _open(tmp_path)
    original_sparse_delete = sparse.delete_document
    try:
        _seed(coordinator)

        def fail_sparse_cleanup(document_id, keep_version=None, only_version=None):
            if keep_version is not None:
                raise RuntimeError("sparse cleanup failed after dense cleanup")
            return original_sparse_delete(
                document_id,
                keep_version=keep_version,
                only_version=only_version,
            )

        sparse.delete_document = fail_sparse_cleanup
        update = coordinator.begin_ingest(
            "doc", "doc.txt", "new complete content", operation_id="update"
        )
        result = coordinator.stage_and_activate(
            update,
            _version_chunks("doc", 2, "new"),
            content="new complete content",
        )
        assert result["cleanup_pending"]
        assert "sparse cleanup failed" in result["cleanup_error"]
        assert dense.count_document("doc") == 2
        assert sparse.count_document("doc") == 4
        _assert_complete_visible_version(dense, sparse, 2)

        sparse.delete_document = original_sparse_delete
        assert coordinator.recover() == ["doc"]
        assert coordinator.recover() == []
        assert dense.count_document("doc") == 2
        assert sparse.count_document("doc") == 2
        _assert_complete_visible_version(dense, sparse, 2)
    finally:
        sparse.delete_document = original_sparse_delete
        _close(registry, dense, sparse)


def test_postcommit_exception_never_rolls_back_the_committed_target(tmp_path):
    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        _seed(coordinator)

        def fail_after_commit(phase, _context):
            if phase == "postcommit":
                raise RuntimeError("failure after durable CAS")

        coordinator.failure_injector = fail_after_commit
        update = coordinator.begin_ingest(
            "doc", "doc.txt", "new complete content", operation_id="update"
        )
        with pytest.raises(RuntimeError, match="failure after durable CAS"):
            coordinator.stage_and_activate(
                update,
                _version_chunks("doc", 2, "new"),
                content="new complete content",
            )

        row = registry.get("doc")
        assert row["version"] == 2
        assert row["cleanup_status"] == "pending"
        _assert_complete_visible_version(dense, sparse, 2)
        assert dense.count_document("doc") == 4
        assert sparse.count_document("doc") == 4

        coordinator.failure_injector = None
        assert coordinator.recover() == ["doc"]
        assert coordinator.recover() == []
        _assert_complete_visible_version(dense, sparse, 2)
        assert dense.count_document("doc") == 2
        assert sparse.count_document("doc") == 2
    finally:
        _close(registry, dense, sparse)


def test_restart_recovery_honors_retain_old_versions_policy(tmp_path):
    registry, dense, sparse, coordinator = _open(tmp_path, cleanup=False)
    _seed(coordinator)

    def crash_before_retirement(phase, _context):
        if phase == "old_version_cleanup":
            raise SimulatedProcessCrash(phase)

    coordinator.failure_injector = crash_before_retirement
    try:
        update = coordinator.begin_ingest(
            "doc", "doc.txt", "new complete content", operation_id="update"
        )
        with pytest.raises(SimulatedProcessCrash):
            coordinator.stage_and_activate(
                update,
                _version_chunks("doc", 2, "new"),
                content="new complete content",
            )
        _assert_complete_visible_version(dense, sparse, 2)
    finally:
        _close(registry, dense, sparse)

    registry, dense, sparse, coordinator = _open(tmp_path, cleanup=False)
    try:
        assert coordinator.recover() == ["doc"]
        assert coordinator.recover() == []
        _assert_complete_visible_version(dense, sparse, 2)
        assert dense.count_document("doc") == 4
        assert sparse.count_document("doc") == 4
        coordinator.verify_parity("doc", 1, state="stale")
        coordinator.verify_parity("doc", 2, state="active")
    finally:
        _close(registry, dense, sparse)


def test_reads_during_update_switch_once_without_empty_or_mixed_results(tmp_path):
    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        _seed(coordinator)
        at_precommit = Event()
        allow_commit = Event()
        at_postcommit = Event()
        allow_cleanup = Event()
        failures: list[BaseException] = []

        def pause(phase, _context):
            if phase == "precommit":
                at_precommit.set()
                assert allow_commit.wait(5), "test did not release precommit"
            elif phase == "postcommit":
                at_postcommit.set()
                assert allow_cleanup.wait(5), "test did not release postcommit"

        def update_document():
            try:
                operation = coordinator.begin_ingest(
                    "doc",
                    "doc.txt",
                    "new complete content",
                    operation_id="concurrent-update",
                )
                coordinator.stage_and_activate(
                    operation,
                    _version_chunks("doc", 2, "new"),
                    content="new complete content",
                )
            except BaseException as exc:  # publish worker assertion/error
                failures.append(exc)

        coordinator.failure_injector = pause
        worker = Thread(target=update_document, name="atomic-update")
        worker.start()
        assert at_precommit.wait(5)
        for _ in range(20):
            _assert_complete_visible_version(dense, sparse, 1)
        allow_commit.set()
        assert at_postcommit.wait(5)
        for _ in range(20):
            _assert_complete_visible_version(dense, sparse, 2)
        allow_cleanup.set()
        worker.join(5)
        assert not worker.is_alive()
        assert not failures
        _assert_complete_visible_version(dense, sparse, 2)
    finally:
        allow_commit.set() if "allow_commit" in locals() else None
        allow_cleanup.set() if "allow_cleanup" in locals() else None
        _close(registry, dense, sparse)


class _FlippingRegistry:
    """Hostile registry proving each search snapshots a document only once."""

    def __init__(self):
        self.calls = 0

    def visible_version(self, _document_id):
        self.calls += 1
        return 1 if self.calls == 1 else 2


def test_each_store_uses_one_registry_snapshot_per_document_and_search(tmp_path):
    dense = QdrantStore("snapshot", 2, str(tmp_path / "qdrant"))
    sparse = SQLiteBM25Retriever(tmp_path / "sparse.db")
    try:
        chunks = _version_chunks("doc", 1, "old") + _version_chunks(
            "doc", 2, "new"
        )
        dense.upsert_chunks(chunks)
        sparse.upsert(chunks)

        dense_registry = _FlippingRegistry()
        dense.bind_registry(dense_registry)
        dense_hits = dense.search([1.0, 0.0], top_k=20)
        assert dense_hits
        assert {hit["metadata"]["version"] for hit in dense_hits} == {1}
        assert dense_registry.calls == 1

        sparse_registry = _FlippingRegistry()
        sparse.bind_registry(sparse_registry)
        sparse_hits = sparse.search("atomicanchor", top_k=20)
        assert sparse_hits
        assert {hit.metadata["version"] for hit in sparse_hits} == {1}
        assert sparse_registry.calls == 1
    finally:
        dense.close()
        sparse.close()


def test_shared_visibility_snapshot_spans_dense_and_sparse_and_has_cas_epoch(
    tmp_path,
):
    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        _seed(coordinator)
        old_snapshot = registry.visibility_snapshot()
        assert old_snapshot.versions == {"doc": 1}
        assert registry.is_visibility_snapshot_current(old_snapshot)

        observed = []

        def inspect_precommit(phase, _context):
            if phase == "precommit":
                # Both generations are physically searchable at hand-off;
                # the immutable registry snapshot is the sole logical gate.
                coordinator.verify_parity("doc", 1, state="active")
                coordinator.verify_parity("doc", 2, state="active")
                _assert_complete_visible_version(
                    dense,
                    sparse,
                    1,
                    visibility_snapshot=old_snapshot,
                )
                assert registry.is_visibility_snapshot_current(old_snapshot)
                observed.append(phase)

        coordinator.failure_injector = inspect_precommit
        update = coordinator.begin_ingest(
            "doc", "doc.txt", "new complete content", operation_id="update"
        )
        coordinator.stage_and_activate(
            update,
            _version_chunks("doc", 2, "new"),
            content="new complete content",
        )
        assert observed == ["precommit"]
        assert not registry.is_visibility_snapshot_current(old_snapshot)

        new_snapshot = registry.visibility_snapshot()
        assert new_snapshot.epoch == old_snapshot.epoch + 1
        assert new_snapshot.versions == {"doc": 2}
        assert registry.is_visibility_snapshot_current(new_snapshot)
        _assert_complete_visible_version(
            dense, sparse, 2, visibility_snapshot=new_snapshot
        )
    finally:
        _close(registry, dense, sparse)


def test_hybrid_retrieval_crossing_cas_retries_without_mixed_versions(tmp_path):
    """Force CAS between dense and sparse branches of one logical query."""

    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        _seed(coordinator)
        embed_calls = 0
        update_calls = 0

        class Embedder:
            embedding_dim = 2
            model_name = "atomic-test-model"

            def embed_query(self, _query):
                nonlocal embed_calls
                embed_calls += 1
                return [1.0, 0.0]

        def commit_update_between_branches():
            nonlocal update_calls
            update_calls += 1
            operation = coordinator.begin_ingest(
                "doc",
                "doc.txt",
                "new complete content",
                operation_id="cross-cas-update",
            )
            coordinator.stage_and_activate(
                operation,
                _version_chunks("doc", 2, "new"),
                content="new complete content",
            )

        class UpdateAfterFirstDenseSearch:
            def __init__(self, inner):
                self.inner = inner
                self.triggered = False

            def __getattr__(self, name):
                return getattr(self.inner, name)

            def search(self, *args, **kwargs):
                hits = self.inner.search(*args, **kwargs)
                if not self.triggered:
                    self.triggered = True
                    commit_update_between_branches()
                return hits

        wrapped_dense = UpdateAfterFirstDenseSearch(dense)
        engine = RetrievalEngine(
            dense=DenseRetriever(Embedder(), wrapped_dense),
            sparse=sparse,
            visibility_registry=registry,
            visibility_retries=2,
            parallel_branches=False,
        )
        result = engine.retrieve(
            "atomicanchor",
            mode="hybrid",
            top_k=2,
            candidate_k=10,
        )

        assert update_calls == 1
        assert embed_calls == 1, "visibility retry must reuse the query embedding"
        assert result.candidates
        assert {candidate.metadata["version"] for candidate in result.candidates} == {2}
        assert {candidate.chunk_id for candidate in result.candidates} == {
            "doc:v2:a",
            "doc:v2:b",
        }
        assert result.diagnostics["visibility_retries"] == 1
        assert result.diagnostics["visibility_epoch"] == registry.visibility_snapshot().epoch
    finally:
        _close(registry, dense, sparse)


def test_list_chunks_has_cross_store_parity_and_logical_active_visibility(tmp_path):
    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        _seed(coordinator)
        observations = {}

        def inspect_handoff(phase, _context):
            if phase not in {"precommit", "postcommit"}:
                return
            expected = 1 if phase == "precommit" else 2
            dense_rows = dense.list_chunks("doc")
            sparse_rows = sparse.list_chunks("doc")
            assert dense_rows and sparse_rows
            expected_ids = [f"doc:v{expected}:a", f"doc:v{expected}:b"]
            assert [row["chunk_id"] for row in dense_rows] == expected_ids
            assert [row["chunk_id"] for row in sparse_rows] == expected_ids
            assert {row["version"] for row in dense_rows} == {expected}
            assert {row["version"] for row in sparse_rows} == {expected}
            assert {row["state"] for row in dense_rows} == {"active"}
            assert {row["state"] for row in sparse_rows} == {"active"}
            assert [row["text"] for row in dense_rows] == [
                row["text"] for row in sparse_rows
            ]
            assert [row["metadata"]["generation"] for row in dense_rows] == [
                row["metadata"]["generation"] for row in sparse_rows
            ]
            for row in [*dense_rows, *sparse_rows]:
                assert json.loads(json.dumps(row, ensure_ascii=False)) == row
                _assert_no_vector_keys(row)

            hidden_version = 2 if expected == 1 else 1
            assert dense.list_chunks("doc", version=hidden_version) == []
            assert sparse.list_chunks("doc", version=hidden_version) == []
            observations[phase] = expected

        coordinator.failure_injector = inspect_handoff
        update = coordinator.begin_ingest(
            "doc", "doc.txt", "new complete content", operation_id="update"
        )
        coordinator.stage_and_activate(
            update,
            _version_chunks("doc", 2, "new"),
            content="new complete content",
        )
        assert observations == {"precommit": 1, "postcommit": 2}

        assert [row["chunk_id"] for row in dense.list_chunks("doc", limit=1)] == [
            "doc:v2:a"
        ]
        assert [row["chunk_id"] for row in sparse.list_chunks("doc", limit=1)] == [
            "doc:v2:a"
        ]
    finally:
        _close(registry, dense, sparse)


@pytest.mark.parametrize("invalid_limit", [True, 1.5, 0, -1, 10_001])
def test_list_chunks_rejects_invalid_or_unbounded_limits(tmp_path, invalid_limit):
    registry, dense, sparse, coordinator = _open(tmp_path)
    try:
        _seed(coordinator)
        expected_error = TypeError if invalid_limit in {True, 1.5} else ValueError
        for store in (dense, sparse):
            with pytest.raises(expected_error):
                store.list_chunks("doc", limit=invalid_limit)
            with pytest.raises(ValueError, match="lifecycle state"):
                store.list_chunks("doc", state="unknown")
            with pytest.raises(TypeError, match="version"):
                store.list_chunks("doc", version=True)
            with pytest.raises(ValueError, match="version"):
                store.list_chunks("doc", version=0)
    finally:
        _close(registry, dense, sparse)
