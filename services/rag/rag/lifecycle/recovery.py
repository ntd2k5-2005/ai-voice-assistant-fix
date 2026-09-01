"""Idempotent startup recovery for interrupted lifecycle operations."""
from __future__ import annotations

from .manager import DocumentRegistry


class RecoveryManager:
    """Repair pending writes, deletes, and post-commit old-version cleanup.

    Recovery always preserves the registry's committed version. It is safe to
    call repeatedly and is intended to run before a service starts accepting
    retrieval traffic after an unclean shutdown.
    """

    def __init__(
        self,
        registry: DocumentRegistry,
        stores=(),
        *,
        cleanup_old_versions: bool = True,
    ):
        self.registry = registry
        self.stores = tuple(stores)
        self.cleanup_old_versions = bool(cleanup_old_versions)
        for store in self.stores:
            binder = getattr(store, "bind_registry", None)
            if binder:
                binder(registry)

    @staticmethod
    def _append_once(recovered: list[str], document_id: str) -> None:
        if document_id not in recovered:
            recovered.append(document_id)

    def _recover_pending(self, row: dict, recovered: list[str]) -> None:
        document_id = row["document_id"]
        version = int(row["pending_version"])
        restore_version = int(row["version"]) if row.get("hash") else None
        try:
            for store in self.stores:
                rollback = getattr(store, "rollback_version", None)
                if rollback:
                    rollback(
                        document_id,
                        version,
                        restore_version=restore_version,
                    )
                else:
                    delete = getattr(store, "delete_document", None)
                    if delete:
                        delete(document_id, only_version=version)
            for store in self.stores:
                counter = getattr(store, "count_document", None)
                if counter and counter(document_id, version=version) != 0:
                    raise RuntimeError("pending version remains after rollback")
            self.registry.abort_ingest(
                document_id,
                "recovered interrupted ingestion",
                operation_id=row.get("pending_operation_id"),
            )
            self._append_once(recovered, document_id)
        except Exception as exc:
            self.registry.mark_pending_error(
                document_id,
                f"pending recovery failed: {type(exc).__name__}: {exc}",
            )

    def _recover_delete(self, row: dict, recovered: list[str]) -> None:
        document_id = row["document_id"]
        try:
            for store in self.stores:
                delete = getattr(store, "delete_document", None)
                if delete:
                    delete(document_id)
            for store in self.stores:
                counter = getattr(store, "count_document", None)
                if counter and counter(document_id) != 0:
                    raise RuntimeError("document remains after delete recovery")
            self.registry.complete_delete(
                document_id, operation_id=row.get("pending_operation_id")
            )
            self._append_once(recovered, document_id)
        except Exception as exc:
            self.registry.mark_delete_error(
                document_id,
                f"delete recovery failed: {type(exc).__name__}: {exc}",
            )

    def _recover_cleanup(self, row: dict, recovered: list[str]) -> None:
        document_id = row["document_id"]
        cleanup_version = int(row["cleanup_version"])
        active_version = int(row["version"])
        try:
            # Re-establish and verify the committed target before touching the
            # old physical fallback.  This ordering is intentionally
            # conservative if a store was externally damaged after CAS.
            for store in self.stores:
                activate = getattr(store, "activate_version", None)
                if activate:
                    activate(document_id, active_version)
            for store in self.stores:
                verifier = getattr(store, "verify_version", None)
                if verifier and not verifier(
                    document_id, active_version, state="active"
                ):
                    raise RuntimeError(
                        "committed version is not physically searchable"
                    )
            for store in self.stores:
                if self.cleanup_old_versions:
                    delete = getattr(store, "delete_document", None)
                    if delete:
                        # Delete every non-current version: a later activation
                        # may already have superseded the recorded cleanup one.
                        delete(document_id, keep_version=active_version)
                else:
                    deactivate = getattr(store, "deactivate_version", None)
                    if deactivate:
                        deactivate(document_id, cleanup_version, state="stale")
            for store in self.stores:
                verifier = getattr(store, "verify_version", None)
                if verifier and not verifier(
                    document_id, active_version, state="active"
                ):
                    raise RuntimeError("active version missing after cleanup recovery")
            self.registry.complete_cleanup(document_id, cleanup_version)
            self._append_once(recovered, document_id)
        except Exception as exc:
            self.registry.mark_cleanup_failed(
                document_id,
                f"cleanup recovery failed: {type(exc).__name__}: {exc}",
            )

    def recover(self) -> list[str]:
        recovered: list[str] = []
        for row in self.registry.pending():
            self._recover_pending(row, recovered)
        for row in self.registry.deleting():
            self._recover_delete(row, recovered)
        for row in self.registry.cleanup_pending():
            self._recover_cleanup(row, recovered)
        return recovered


__all__ = ["DocumentRegistry", "RecoveryManager"]
