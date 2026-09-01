"""Cross-store lifecycle protocol for dense + sparse representations."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from time import perf_counter
from typing import Any

from ..errors import LifecycleError, StorageConsistencyError, StorageError
from ..storage.identity import ACTIVE, PENDING, prepare_chunks
from .manager import DocumentRegistry


FailureInjector = Callable[[str, Mapping[str, Any]], None]


class StorageLifecycleCoordinator:
    """Execute the recoverable dense+sparse version-switch protocol.

    Pending chunks are staged and verified in both stores.  The target is then
    made physically searchable while the old version remains searchable too;
    the registry still selects only the old version.  One registry CAS selects
    the target, after which old rows may be retired or deleted.  Thus neither
    sequential store transitions nor cleanup failures can create an empty
    logical visibility window.  Startup code must call ``recover`` before
    serving reads after an unclean shutdown.
    """

    CHECKPOINTS = (
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
        "postcommit",
        "old_version_cleanup",
        "dense_cleanup",
        "sparse_cleanup",
    )

    def __init__(
        self,
        registry: DocumentRegistry,
        dense_store,
        sparse_store,
        *,
        failure_injector: FailureInjector | None = None,
        cleanup_old_versions: bool = True,
    ):
        self.registry = registry
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.failure_injector = failure_injector
        self.cleanup_old_versions = bool(cleanup_old_versions)
        for store in (dense_store, sparse_store):
            binder = getattr(store, "bind_registry", None)
            if binder:
                binder(registry)

    @property
    def stores(self) -> tuple[Any, Any]:
        return self.dense_store, self.sparse_store

    def _checkpoint(self, phase: str, **context: Any) -> None:
        if phase not in self.CHECKPOINTS and not phase.startswith("delete_"):
            raise LifecycleError(f"unknown failure checkpoint {phase!r}")
        if self.failure_injector is not None:
            self.failure_injector(phase, context)

    def begin_ingest(
        self,
        document_id: str,
        filename: str,
        content: str,
        *,
        force: bool = False,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        operation = self.registry.begin_ingest(
            document_id,
            filename,
            content,
            force=force,
            operation_id=operation_id,
        )
        self._checkpoint("registry_begin", operation=operation)
        return operation

    def begin_reindex(
        self,
        document_id: str,
        filename: str,
        content: str,
        *,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        return self.begin_ingest(
            document_id,
            filename,
            content,
            force=True,
            operation_id=operation_id,
        )

    @staticmethod
    def _validate_operation_chunks(
        operation: Mapping[str, Any], chunks: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if operation.get("unchanged"):
            return []
        document_id = str(operation.get("document_id") or "")
        version = int(operation.get("pending_version") or 0)
        operation_id = str(operation.get("operation_id") or "")
        if not document_id or version <= 0 or not operation_id:
            raise LifecycleError("invalid pending lifecycle operation")
        prepared = prepare_chunks(
            chunks, state=PENDING, operation_id=operation_id
        )
        if not prepared:
            raise StorageConsistencyError("cannot activate an empty chunk version")
        for chunk in prepared:
            if (
                chunk["document_id"] != document_id
                or int(chunk["version"]) != version
            ):
                raise StorageConsistencyError(
                    "every staged chunk must match the operation document/version"
                )
        return prepared

    def _verify_store(
        self,
        store,
        chunks: list[Mapping[str, Any]],
        *,
        state: str,
        operation_id: str,
        label: str,
    ) -> None:
        verifier = getattr(store, "verify_chunks", None)
        if not verifier:
            raise StorageError(f"{label} store does not implement verify_chunks")
        if not verifier(chunks, state=state, operation_id=operation_id):
            raise StorageConsistencyError(
                f"{label} store verification failed for staged chunks"
            )

    def verify_parity(
        self,
        document_id: str,
        version: int,
        *,
        state: str | None = None,
        operation_id: str | None = None,
    ) -> bool:
        """Raise on any logical identity/state drift across representations."""

        dense_get = getattr(self.dense_store, "get_manifest", None)
        sparse_get = getattr(self.sparse_store, "get_manifest", None)
        if not dense_get or not sparse_get:
            raise StorageError("both stores must expose get_manifest")
        dense = dense_get(document_id, version)
        sparse = sparse_get(document_id, version)
        if not dense or dense.keys() != sparse.keys():
            raise StorageConsistencyError(
                f"dense/sparse chunk sets differ for {document_id!r}/v{int(version)}"
            )
        for chunk_id in dense:
            left, right = dense[chunk_id], sparse[chunk_id]
            if (
                left.chunk_id != right.chunk_id
                or left.document_id != right.document_id
                or left.version != right.version
                or left.identity_hash != right.identity_hash
                or left.state != right.state
                or left.operation_id != right.operation_id
            ):
                raise StorageConsistencyError(
                    f"dense/sparse identity differs for chunk {chunk_id!r}"
                )
            if state is not None and left.state != state:
                raise StorageConsistencyError(
                    f"chunk {chunk_id!r} is {left.state!r}, expected {state!r}"
                )
            if operation_id is not None and left.operation_id != str(operation_id):
                raise StorageConsistencyError(
                    f"chunk {chunk_id!r} belongs to another operation"
                )
        return True

    def _rollback_pending(
        self,
        operation: Mapping[str, Any],
        cause: BaseException,
    ) -> list[str]:
        document_id = str(operation["document_id"])
        version = int(operation["pending_version"])
        previous_version = int(operation.get("version") or 0) or None
        errors: list[str] = []
        for label, store in (
            ("sparse", self.sparse_store),
            ("dense", self.dense_store),
        ):
            try:
                rollback = getattr(store, "rollback_version", None)
                if rollback:
                    rollback(
                        document_id,
                        version,
                        restore_version=previous_version,
                    )
                else:
                    store.delete_document(document_id, only_version=version)
            except Exception as exc:  # keep registry pending for startup recovery
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
        if errors:
            self.registry.mark_pending_error(
                document_id,
                "rollback pending after "
                f"{type(cause).__name__}: {cause}; " + "; ".join(errors),
            )
        else:
            self.registry.abort_ingest(
                document_id,
                f"ingestion failed: {type(cause).__name__}: {cause}",
                operation_id=str(operation["operation_id"]),
            )
        return errors

    def stage_and_activate(
        self,
        operation: Mapping[str, Any],
        chunks: Iterable[Mapping[str, Any]],
        *,
        content: str,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Persist, verify, activate, CAS-commit, then clean the old version."""

        total_started = perf_counter()
        timings_ms: dict[str, float] = {
            "dense_write": 0.0,
            "dense_verify": 0.0,
            "sparse_write": 0.0,
            "sparse_verify": 0.0,
            "physical_readiness": 0.0,
            "registry_cas": 0.0,
            "old_version_cleanup": 0.0,
        }
        if operation.get("unchanged"):
            return {
                "document_id": str(operation["document_id"]),
                "version": int(operation["version"]),
                "operation_id": str(operation["operation_id"]),
                "action": str(operation.get("action") or "unchanged"),
                "unchanged": True,
                "cleanup_pending": False,
                "cleanup_error": None,
                "cleanup_old_versions": self.cleanup_old_versions,
                "old_versions_retained": False,
                "timings_ms": {**timings_ms, "total": 0.0},
                "chunks": [],
            }
        prepared = self._validate_operation_chunks(operation, chunks)
        document_id = str(operation["document_id"])
        version = int(operation["pending_version"])
        operation_id = str(operation["operation_id"])
        filename = str(filename or operation.get("pending_filename") or document_id)
        try:
            self._checkpoint("before_stage", operation=operation)
            phase_started = perf_counter()
            self.dense_store.stage_chunks(prepared, operation_id=operation_id)
            timings_ms["dense_write"] = (perf_counter() - phase_started) * 1000
            self.registry.record_operation_phase(operation_id, "dense_staged")
            self._checkpoint("dense_write", operation=operation)

            phase_started = perf_counter()
            self._verify_store(
                self.dense_store,
                prepared,
                state=PENDING,
                operation_id=operation_id,
                label="dense",
            )
            timings_ms["dense_verify"] = (perf_counter() - phase_started) * 1000
            self.registry.record_operation_phase(operation_id, "dense_verified")
            self._checkpoint("dense_verify", operation=operation)

            phase_started = perf_counter()
            self.sparse_store.stage_chunks(prepared, operation_id=operation_id)
            timings_ms["sparse_write"] = (perf_counter() - phase_started) * 1000
            self.registry.record_operation_phase(operation_id, "sparse_staged")
            self._checkpoint("sparse_write", operation=operation)

            phase_started = perf_counter()
            self._verify_store(
                self.sparse_store,
                prepared,
                state=PENDING,
                operation_id=operation_id,
                label="sparse",
            )
            self.registry.record_operation_phase(operation_id, "sparse_verified")
            self.verify_parity(
                document_id,
                version,
                state=PENDING,
                operation_id=operation_id,
            )
            timings_ms["sparse_verify"] = (perf_counter() - phase_started) * 1000
            self._checkpoint("sparse_verify", operation=operation)
            self._checkpoint("verification", operation=operation)

            self.registry.record_operation_phase(operation_id, "activating")
            # Physical readiness deliberately does not retire the committed
            # old version.  Until registry.commit() succeeds the registry gate
            # continues to select the old version in both stores.
            phase_started = perf_counter()
            self.dense_store.activate_version(
                document_id, version, operation_id=operation_id
            )
            self._checkpoint("dense_ready", operation=operation)
            self.sparse_store.activate_version(
                document_id, version, operation_id=operation_id
            )
            self._checkpoint("sparse_ready", operation=operation)
            self.verify_parity(
                document_id,
                version,
                state=ACTIVE,
                operation_id=operation_id,
            )
            timings_ms["physical_readiness"] = (
                perf_counter() - phase_started
            ) * 1000
            self.registry.record_operation_phase(operation_id, "activated")
            self._checkpoint("activation", operation=operation)
            self._checkpoint("precommit", operation=operation)

            phase_started = perf_counter()
            registry_row = self.registry.commit(
                document_id,
                filename,
                content,
                version,
                operation_id=operation_id,
            )
            timings_ms["registry_cas"] = (perf_counter() - phase_started) * 1000
        except Exception as exc:
            rollback_errors = self._rollback_pending(operation, exc)
            if rollback_errors:
                raise StorageConsistencyError(
                    "ingestion failed and compensation remains pending: "
                    + "; ".join(rollback_errors)
                ) from exc
            raise

        # From this point onward the new version is committed.  A failure must
        # never run pre-commit compensation, because deleting the target would
        # make the registry point at missing data.  Recovery instead completes
        # only retirement/cleanup of the old version.
        self._checkpoint(
            "postcommit", operation=operation, registry_row=registry_row
        )

        cleanup_pending = registry_row.get("cleanup_status") == "pending"
        cleanup_error = None
        old_versions_retained = False
        if cleanup_pending:
            cleanup_started = perf_counter()
            try:
                self._checkpoint("old_version_cleanup", operation=operation)
                if not self.dense_store.verify_version(
                    document_id,
                    version,
                    state=ACTIVE,
                    operation_id=operation_id,
                ) or not self.sparse_store.verify_version(
                    document_id,
                    version,
                    state=ACTIVE,
                    operation_id=operation_id,
                ):
                    raise StorageConsistencyError(
                        "committed version is not searchable before old cleanup"
                    )
                if self.cleanup_old_versions:
                    self.dense_store.delete_document(
                        document_id, keep_version=registry_row["version"]
                    )
                    self._checkpoint("dense_cleanup", operation=operation)
                    self.sparse_store.delete_document(
                        document_id, keep_version=registry_row["version"]
                    )
                    self._checkpoint("sparse_cleanup", operation=operation)
                else:
                    old_version = int(registry_row["cleanup_version"])
                    self.dense_store.deactivate_version(
                        document_id, old_version, state="stale"
                    )
                    self._checkpoint("dense_cleanup", operation=operation)
                    self.sparse_store.deactivate_version(
                        document_id, old_version, state="stale"
                    )
                    self._checkpoint("sparse_cleanup", operation=operation)
                    self.verify_parity(
                        document_id, old_version, state="stale"
                    )
                    old_versions_retained = True
                if not self.dense_store.verify_version(
                    document_id,
                    version,
                    state=ACTIVE,
                    operation_id=operation_id,
                ) or not self.sparse_store.verify_version(
                    document_id,
                    version,
                    state=ACTIVE,
                    operation_id=operation_id,
                ):
                    raise StorageConsistencyError(
                        "active version failed verification after stale cleanup"
                    )
                self.registry.complete_cleanup(
                    document_id, registry_row["cleanup_version"]
                )
                cleanup_pending = False
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
                self.registry.mark_cleanup_failed(document_id, cleanup_error)
            finally:
                timings_ms["old_version_cleanup"] = (
                    perf_counter() - cleanup_started
                ) * 1000

        timings_ms["total"] = (perf_counter() - total_started) * 1000

        return {
            "document_id": document_id,
            "version": version,
            "operation_id": operation_id,
            "action": str(operation.get("action") or "ingest"),
            "unchanged": False,
            "cleanup_pending": cleanup_pending,
            "cleanup_error": cleanup_error,
            "cleanup_old_versions": self.cleanup_old_versions,
            "old_versions_retained": old_versions_retained,
            "timings_ms": timings_ms,
            "chunks": prepare_chunks(
                prepared, state=ACTIVE, operation_id=operation_id
            ),
        }

    def delete(
        self, document_id: str, *, operation_id: str | None = None
    ) -> dict[str, Any]:
        operation = self.registry.begin_delete(
            document_id, operation_id=operation_id
        )
        self._checkpoint("delete_registry_begin", operation=operation)
        if operation.get("unchanged"):
            return operation
        operation_id = str(operation["operation_id"])
        try:
            self.dense_store.delete_document(document_id)
            self._checkpoint("delete_dense", operation=operation)
            self.sparse_store.delete_document(document_id)
            self._checkpoint("delete_sparse", operation=operation)
            for label, store in self.stores_with_labels:
                counter = getattr(store, "count_document", None)
                if counter and counter(document_id) != 0:
                    raise StorageConsistencyError(
                        f"{label} store still contains deleted document {document_id!r}"
                    )
            row = self.registry.complete_delete(
                document_id, operation_id=operation_id
            )
        except Exception as exc:
            self.registry.mark_delete_error(
                document_id, f"{type(exc).__name__}: {exc}"
            )
            raise
        return {
            **row,
            "operation_id": operation_id,
            "action": "delete",
            "unchanged": False,
        }

    @property
    def stores_with_labels(self):
        return (("dense", self.dense_store), ("sparse", self.sparse_store))

    def recover(self) -> list[str]:
        """Run idempotent startup recovery before accepting retrievals."""

        from .recovery import RecoveryManager

        return RecoveryManager(
            self.registry,
            self.stores,
            cleanup_old_versions=self.cleanup_old_versions,
        ).recover()


__all__ = ["FailureInjector", "StorageLifecycleCoordinator"]
