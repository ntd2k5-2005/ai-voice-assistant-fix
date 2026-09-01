"""Durable document lifecycle registry with operation-level compare-and-set."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sqlite3
from threading import RLock
import time
from types import MappingProxyType
import uuid

from ..errors import ConcurrentUpdateError, LifecycleError


_COLUMNS = (
    "document_id,filename,hash,version,status,created_at,updated_at,"
    "pending_hash,pending_version,pending_filename,last_error,"
    "cleanup_version,cleanup_status,cleanup_error,pending_operation_id,"
    "pending_action"
)
_OPERATION_COLUMNS = (
    "operation_id,document_id,action,content_hash,target_version,"
    "previous_version,status,error,created_at,updated_at"
)
_IN_PROGRESS = frozenset(
    {
        "begun",
        "dense_staged",
        "dense_verified",
        "sparse_staged",
        "sparse_verified",
        "activating",
        "activated",
        "delete_pending",
    }
)


@dataclass(frozen=True)
class VisibilitySnapshot:
    """Immutable logical-version selection shared by retrieval branches."""

    epoch: int
    versions: Mapping[str, int]

    def __post_init__(self) -> None:
        if int(self.epoch) < 0:
            raise ValueError("visibility epoch cannot be negative")
        canonical = {
            str(document_id): int(version)
            for document_id, version in self.versions.items()
        }
        object.__setattr__(self, "epoch", int(self.epoch))
        object.__setattr__(self, "versions", MappingProxyType(canonical))

    def version_for(self, document_id: str) -> int | None:
        return self.versions.get(str(document_id))


class DocumentRegistry:
    """Persistent source of truth for active and pending document versions.

    SQLite ``BEGIN IMMEDIATE`` serializes version allocation across processes.
    A caller-supplied operation id is durable, making retries distinguishable
    from a competing update and allowing a completed reindex/delete to replay
    as a no-op.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        timeout_seconds: float = 30.0,
    ):
        self.path = str(path)
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        self.db.row_factory = sqlite3.Row
        self._lock = RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            if self.path != ":memory:":
                self.db.execute("PRAGMA journal_mode=WAL")
                self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute(
                f"PRAGMA busy_timeout={max(1, round(self.timeout_seconds * 1000))}"
            )
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS documents(
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    pending_hash TEXT,
                    pending_version INTEGER,
                    pending_filename TEXT,
                    last_error TEXT,
                    cleanup_version INTEGER,
                    cleanup_status TEXT,
                    cleanup_error TEXT,
                    pending_operation_id TEXT,
                    pending_action TEXT
                )"""
            )
            existing = {
                row[1] for row in self.db.execute("PRAGMA table_info(documents)")
            }
            additions = (
                ("pending_hash", "TEXT"),
                ("pending_version", "INTEGER"),
                ("pending_filename", "TEXT"),
                ("last_error", "TEXT"),
                ("cleanup_version", "INTEGER"),
                ("cleanup_status", "TEXT"),
                ("cleanup_error", "TEXT"),
                ("pending_operation_id", "TEXT"),
                ("pending_action", "TEXT"),
            )
            for name, declaration in additions:
                if name not in existing:
                    self.db.execute(
                        f"ALTER TABLE documents ADD COLUMN {name} {declaration}"
                    )
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS lifecycle_operations(
                    operation_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    content_hash TEXT,
                    target_version INTEGER,
                    previous_version INTEGER,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_lifecycle_operations_document "
                "ON lifecycle_operations(document_id,updated_at)"
            )
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS lifecycle_metadata(
                    key TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                )"""
            )
            self.db.execute(
                """INSERT OR IGNORE INTO lifecycle_metadata(key,value)
                    VALUES('visibility_epoch',0)"""
            )

    @contextmanager
    def _transaction(self):
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.db.rollback()
                raise
            else:
                self.db.commit()

    @staticmethod
    def content_hash(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("document content must be a string")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_unlocked(self, document_id: str):
        row = self.db.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE document_id=?",
            (str(document_id),),
        ).fetchone()
        return dict(row) if row else None

    def _bump_visibility_epoch_unlocked(self) -> None:
        cursor = self.db.execute(
            """UPDATE lifecycle_metadata SET value=value+1
                WHERE key='visibility_epoch'"""
        )
        if cursor.rowcount != 1:  # defensive repair for a damaged old DB
            self.db.execute(
                """INSERT INTO lifecycle_metadata(key,value)
                    VALUES('visibility_epoch',1)
                    ON CONFLICT(key) DO UPDATE SET value=value+1"""
            )

    def get(self, document_id):
        with self._lock:
            return self._get_unlocked(str(document_id))

    def _get_operation_unlocked(self, operation_id: str):
        row = self.db.execute(
            f"SELECT {_OPERATION_COLUMNS} FROM lifecycle_operations "
            "WHERE operation_id=?",
            (str(operation_id),),
        ).fetchone()
        return dict(row) if row else None

    def get_operation(self, operation_id: str):
        with self._lock:
            return self._get_operation_unlocked(str(operation_id))

    @staticmethod
    def _operation_id(value: str | None) -> str:
        if value is None:
            return uuid.uuid4().hex
        operation_id = str(value)
        if not operation_id:
            raise LifecycleError("operation_id cannot be empty")
        return operation_id

    @staticmethod
    def _operation_result(
        row: dict,
        *,
        operation_id: str,
        action: str,
        unchanged: bool,
        replayed: bool = False,
    ) -> dict:
        return {
            **row,
            "operation_id": operation_id,
            "action": action,
            "unchanged": unchanged,
            "replayed": replayed,
        }

    def _validate_replay(
        self,
        operation: dict,
        *,
        document_id: str,
        content_hash: str | None,
        requested_action: str,
    ) -> None:
        if operation["document_id"] != document_id:
            raise LifecycleError("operation_id is already bound to another document")
        if content_hash is not None and operation["content_hash"] != content_hash:
            raise LifecycleError("operation_id is already bound to different content")
        if requested_action == "delete" and operation["action"] != "delete":
            raise LifecycleError("operation_id is already bound to a non-delete action")
        if requested_action == "reindex" and operation["action"] != "reindex":
            raise LifecycleError("operation_id is already bound to a non-reindex action")

    def register(
        self,
        document_id,
        filename,
        text,
        version=None,
        status="active",
    ):
        """Legacy immediate registry write used by small component tests."""

        document_id, filename = str(document_id), str(filename)
        digest = self.content_hash(text)
        now = time.time()
        with self._transaction():
            old = self._get_unlocked(document_id)
            if old and old["hash"] == digest and old["status"] == status:
                return old
            next_version = (
                int(version)
                if version is not None
                else (int(old["version"]) + 1 if old else 1)
            )
            self.db.execute(
                """INSERT INTO documents(
                        document_id,filename,hash,version,status,created_at,
                        updated_at,pending_hash,pending_version,pending_filename,
                        last_error,cleanup_version,cleanup_status,cleanup_error,
                        pending_operation_id,pending_action)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        filename=excluded.filename,hash=excluded.hash,
                        version=excluded.version,status=excluded.status,
                        updated_at=excluded.updated_at,pending_hash=NULL,
                        pending_version=NULL,pending_filename=NULL,last_error=NULL,
                        cleanup_version=NULL,cleanup_status=NULL,
                        cleanup_error=NULL,pending_operation_id=NULL,
                        pending_action=NULL""",
                (
                    document_id,
                    filename,
                    digest,
                    next_version,
                    status,
                    now,
                    now,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            self._bump_visibility_epoch_unlocked()
            return self._get_unlocked(document_id)

    def plan(self, document_id, filename, text, *, force: bool = False):
        """Read-only hint; ``begin_ingest`` is the authoritative allocation."""

        document_id = str(document_id)
        digest = self.content_hash(text)
        existing = self.get(document_id)
        if (
            existing
            and existing["status"] == "active"
            and existing["hash"] == digest
            and not force
        ):
            return {
                "unchanged": True,
                "version": existing["version"],
                "hash": digest,
                "previous": existing,
            }
        resumable = (
            existing
            and existing["status"] == "indexing"
            and existing["pending_hash"] == digest
        )
        version = (
            existing["pending_version"]
            if resumable
            else (int(existing["version"]) + 1 if existing else 1)
        )
        return {
            "unchanged": False,
            "version": version,
            "hash": digest,
            "previous": existing,
        }

    def begin_ingest(
        self,
        document_id,
        filename,
        text,
        *,
        force: bool = False,
        operation_id: str | None = None,
    ):
        document_id, filename = str(document_id), str(filename)
        digest = self.content_hash(text)
        requested_action = "reindex" if force else "ingest"
        operation_id = self._operation_id(operation_id)
        now = time.time()
        with self._transaction():
            existing = self._get_unlocked(document_id)
            operation = self._get_operation_unlocked(operation_id)
            if operation:
                self._validate_replay(
                    operation,
                    document_id=document_id,
                    content_hash=digest,
                    requested_action=requested_action,
                )
                if operation["status"] in {"completed", "unchanged"}:
                    current = self._get_unlocked(document_id)
                    return self._operation_result(
                        current,
                        operation_id=operation_id,
                        action=operation["action"],
                        unchanged=True,
                        replayed=True,
                    )
                if (
                    existing
                    and existing["pending_operation_id"] == operation_id
                    and operation["status"] in _IN_PROGRESS
                ):
                    return self._operation_result(
                        existing,
                        operation_id=operation_id,
                        action=operation["action"],
                        unchanged=False,
                        replayed=True,
                    )

            if existing and existing["status"] in {"indexing", "deleting"}:
                raise ConcurrentUpdateError(
                    f"document {document_id!r} already has operation "
                    f"{existing['pending_operation_id']!r} in progress"
                )

            if (
                existing
                and existing["status"] == "active"
                and existing["hash"] == digest
                and not force
            ):
                action = "unchanged"
                self.db.execute(
                    """INSERT INTO lifecycle_operations(
                            operation_id,document_id,action,content_hash,
                            target_version,previous_version,status,error,
                            created_at,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(operation_id) DO UPDATE SET
                            status='unchanged',error=NULL,updated_at=excluded.updated_at""",
                    (
                        operation_id,
                        document_id,
                        action,
                        digest,
                        existing["version"],
                        existing["version"],
                        "unchanged",
                        None,
                        now,
                        now,
                    ),
                )
                return self._operation_result(
                    existing,
                    operation_id=operation_id,
                    action=action,
                    unchanged=True,
                )

            previous_version = int(existing["version"]) if existing else 0
            version = previous_version + 1 if existing else 1
            action = (
                "reindex"
                if force
                else "update"
                if existing and existing["hash"]
                else "add"
            )
            if existing:
                self.db.execute(
                    """UPDATE documents SET status='indexing',pending_hash=?,
                              pending_version=?,pending_filename=?,last_error=NULL,
                              pending_operation_id=?,pending_action=?,updated_at=?
                        WHERE document_id=?""",
                    (
                        digest,
                        version,
                        filename,
                        operation_id,
                        action,
                        now,
                        document_id,
                    ),
                )
            else:
                self.db.execute(
                    """INSERT INTO documents(
                            document_id,filename,hash,version,status,created_at,
                            updated_at,pending_hash,pending_version,
                            pending_filename,last_error,cleanup_version,
                            cleanup_status,cleanup_error,pending_operation_id,
                            pending_action)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        document_id,
                        filename,
                        "",
                        0,
                        "indexing",
                        now,
                        now,
                        digest,
                        version,
                        filename,
                        None,
                        None,
                        None,
                        None,
                        operation_id,
                        action,
                    ),
                )
            if operation:
                self.db.execute(
                    """UPDATE lifecycle_operations SET action=?,target_version=?,
                              previous_version=?,status='begun',error=NULL,
                              updated_at=? WHERE operation_id=?""",
                    (action, version, previous_version, now, operation_id),
                )
            else:
                self.db.execute(
                    """INSERT INTO lifecycle_operations(
                            operation_id,document_id,action,content_hash,
                            target_version,previous_version,status,error,
                            created_at,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        operation_id,
                        document_id,
                        action,
                        digest,
                        version,
                        previous_version,
                        "begun",
                        None,
                        now,
                        now,
                    ),
                )
            return self._operation_result(
                self._get_unlocked(document_id),
                operation_id=operation_id,
                action=action,
                unchanged=False,
            )

    def record_operation_phase(
        self, operation_id: str, phase: str, error: str | None = None
    ):
        operation_id, phase = str(operation_id), str(phase)
        allowed = _IN_PROGRESS | {
            "committed",
            "cleanup_pending",
            "completed",
            "unchanged",
            "rolled_back",
            "failed",
            "deleted",
        }
        if phase not in allowed:
            raise LifecycleError(f"unsupported lifecycle phase {phase!r}")
        with self._transaction():
            cursor = self.db.execute(
                """UPDATE lifecycle_operations SET status=?,error=?,updated_at=?
                    WHERE operation_id=?""",
                (phase, str(error) if error else None, time.time(), operation_id),
            )
            if cursor.rowcount != 1:
                raise LifecycleError(f"unknown lifecycle operation {operation_id!r}")
            return self._get_operation_unlocked(operation_id)

    def commit(
        self,
        document_id,
        filename,
        text,
        version,
        *,
        operation_id: str | None = None,
    ):
        document_id, version = str(document_id), int(version)
        digest = self.content_hash(text)
        now = time.time()
        with self._transaction():
            previous = self._get_unlocked(document_id)
            if not previous:
                raise LifecycleError(f"unknown pending document {document_id!r}")
            if (
                previous["status"] == "active"
                and int(previous["version"]) == version
                and previous["hash"] == digest
            ):
                if operation_id:
                    operation = self._get_operation_unlocked(operation_id)
                    if operation and operation["status"] in {
                        "committed",
                        "cleanup_pending",
                        "completed",
                    }:
                        return previous
                return previous
            if previous["status"] != "indexing":
                raise ConcurrentUpdateError(
                    f"document {document_id!r} is not in indexing state"
                )
            if int(previous["pending_version"] or 0) != version:
                raise ConcurrentUpdateError("pending version changed before activation")
            if previous["pending_hash"] != digest:
                raise ConcurrentUpdateError("pending content changed before activation")
            expected_operation = previous["pending_operation_id"]
            if operation_id is not None and expected_operation != str(operation_id):
                raise ConcurrentUpdateError("lifecycle operation lost ownership")
            operation_id = str(operation_id or expected_operation or "") or None
            cleanup_version = (
                int(previous["version"])
                if previous["hash"] and int(previous["version"]) != version
                else None
            )
            self.db.execute(
                """UPDATE documents SET filename=?,hash=?,version=?,
                          status='active',updated_at=?,pending_hash=NULL,
                          pending_version=NULL,pending_filename=NULL,
                          last_error=NULL,cleanup_version=?,cleanup_status=?,
                          cleanup_error=NULL,pending_operation_id=NULL,
                          pending_action=NULL
                    WHERE document_id=?""",
                (
                    str(filename),
                    digest,
                    version,
                    now,
                    cleanup_version,
                    "pending" if cleanup_version is not None else None,
                    document_id,
                ),
            )
            if operation_id:
                phase = "cleanup_pending" if cleanup_version is not None else "completed"
                self.db.execute(
                    """UPDATE lifecycle_operations SET status=?,error=NULL,
                              updated_at=? WHERE operation_id=?""",
                    (phase, now, operation_id),
                )
            self._bump_visibility_epoch_unlocked()
            return self._get_unlocked(document_id)

    def abort_ingest(
        self,
        document_id,
        error=None,
        *,
        operation_id: str | None = None,
    ):
        document_id = str(document_id)
        with self._transaction():
            row = self._get_unlocked(document_id)
            if not row:
                return None
            if operation_id is not None and row["pending_operation_id"] not in {
                None,
                str(operation_id),
            }:
                raise ConcurrentUpdateError("cannot abort another lifecycle operation")
            operation_id = str(operation_id or row["pending_operation_id"] or "") or None
            if row["status"] != "indexing":
                return row
            status = "active" if row["hash"] else "failed"
            self.db.execute(
                """UPDATE documents SET status=?,pending_hash=NULL,
                          pending_version=NULL,pending_filename=NULL,last_error=?,
                          pending_operation_id=NULL,pending_action=NULL,
                          updated_at=? WHERE document_id=?""",
                (
                    status,
                    str(error) if error else None,
                    time.time(),
                    document_id,
                ),
            )
            if operation_id:
                self.db.execute(
                    """UPDATE lifecycle_operations SET status='rolled_back',
                              error=?,updated_at=? WHERE operation_id=?""",
                    (
                        str(error) if error else None,
                        time.time(),
                        operation_id,
                    ),
                )
            return self._get_unlocked(document_id)

    def mark_pending_error(self, document_id, error):
        with self._transaction():
            self.db.execute(
                """UPDATE documents SET last_error=?,updated_at=?
                    WHERE document_id=? AND status='indexing'""",
                (str(error), time.time(), str(document_id)),
            )
            row = self._get_unlocked(str(document_id))
            if row and row["pending_operation_id"]:
                self.db.execute(
                    """UPDATE lifecycle_operations SET error=?,updated_at=?
                        WHERE operation_id=?""",
                    (str(error), time.time(), row["pending_operation_id"]),
                )
            return row

    def mark_cleanup_failed(self, document_id, error):
        with self._transaction():
            self.db.execute(
                """UPDATE documents SET cleanup_status='pending',
                          cleanup_error=?,updated_at=? WHERE document_id=?""",
                (str(error), time.time(), str(document_id)),
            )
            operation = self.db.execute(
                """SELECT operation_id FROM lifecycle_operations
                    WHERE document_id=? AND status='cleanup_pending'
                    ORDER BY updated_at DESC LIMIT 1""",
                (str(document_id),),
            ).fetchone()
            if operation:
                self.db.execute(
                    """UPDATE lifecycle_operations SET error=?,updated_at=?
                        WHERE operation_id=?""",
                    (str(error), time.time(), operation[0]),
                )
            return self._get_unlocked(str(document_id))

    def complete_cleanup(self, document_id, version):
        document_id, version = str(document_id), int(version)
        with self._transaction():
            cursor = self.db.execute(
                """UPDATE documents SET cleanup_version=NULL,
                          cleanup_status=NULL,cleanup_error=NULL,updated_at=?
                    WHERE document_id=? AND cleanup_version=?""",
                (time.time(), document_id, version),
            )
            if cursor.rowcount:
                self.db.execute(
                    """UPDATE lifecycle_operations SET status='completed',
                              error=NULL,updated_at=?
                        WHERE operation_id=(
                            SELECT operation_id FROM lifecycle_operations
                            WHERE document_id=? AND status='cleanup_pending'
                            ORDER BY updated_at DESC LIMIT 1)""",
                    (time.time(), document_id),
                )
            return self._get_unlocked(document_id)

    def pending(self):
        with self._lock:
            return [
                dict(row)
                for row in self.db.execute(
                    f"SELECT {_COLUMNS} FROM documents "
                    "WHERE status='indexing' AND pending_version IS NOT NULL"
                )
            ]

    def cleanup_pending(self):
        with self._lock:
            return [
                dict(row)
                for row in self.db.execute(
                    f"SELECT {_COLUMNS} FROM documents "
                    "WHERE cleanup_status='pending' AND cleanup_version IS NOT NULL"
                )
            ]

    def begin_delete(self, document_id, *, operation_id: str | None = None):
        document_id = str(document_id)
        operation_id = self._operation_id(operation_id)
        now = time.time()
        with self._transaction():
            row = self._get_unlocked(document_id)
            operation = self._get_operation_unlocked(operation_id)
            if operation:
                self._validate_replay(
                    operation,
                    document_id=document_id,
                    content_hash=None,
                    requested_action="delete",
                )
                if operation["status"] in {"completed", "deleted"}:
                    return {
                        **(row or {"document_id": document_id, "status": "deleted"}),
                        "operation_id": operation_id,
                        "unchanged": True,
                        "replayed": True,
                        "action": "delete",
                    }
                if (
                    row
                    and row["status"] == "deleting"
                    and row["pending_operation_id"] == operation_id
                ):
                    return self._operation_result(
                        row,
                        operation_id=operation_id,
                        action="delete",
                        unchanged=False,
                        replayed=True,
                    )
            if row and row["status"] in {"indexing", "deleting"}:
                raise ConcurrentUpdateError(
                    f"document {document_id!r} already has an operation in progress"
                )
            if row is None or row["status"] == "deleted":
                self.db.execute(
                    """INSERT INTO lifecycle_operations(
                            operation_id,document_id,action,content_hash,
                            target_version,previous_version,status,error,
                            created_at,updated_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(operation_id) DO UPDATE SET
                            status='deleted',error=NULL,updated_at=excluded.updated_at""",
                    (
                        operation_id,
                        document_id,
                        "delete",
                        row["hash"] if row else None,
                        row["version"] if row else None,
                        row["version"] if row else None,
                        "deleted",
                        None,
                        now,
                        now,
                    ),
                )
                return {
                    **(row or {"document_id": document_id, "status": "deleted"}),
                    "operation_id": operation_id,
                    "unchanged": True,
                    "replayed": False,
                    "action": "delete",
                }
            self.db.execute(
                """UPDATE documents SET status='deleting',pending_operation_id=?,
                          pending_action='delete',last_error=NULL,updated_at=?
                    WHERE document_id=?""",
                (operation_id, now, document_id),
            )
            self.db.execute(
                """INSERT INTO lifecycle_operations(
                        operation_id,document_id,action,content_hash,
                        target_version,previous_version,status,error,
                        created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    operation_id,
                    document_id,
                    "delete",
                    row["hash"],
                    row["version"],
                    row["version"],
                    "delete_pending",
                    None,
                    now,
                    now,
                ),
            )
            self._bump_visibility_epoch_unlocked()
            return self._operation_result(
                self._get_unlocked(document_id),
                operation_id=operation_id,
                action="delete",
                unchanged=False,
            )

    def deleting(self):
        with self._lock:
            return [
                dict(row)
                for row in self.db.execute(
                    f"SELECT {_COLUMNS} FROM documents WHERE status='deleting'"
                )
            ]

    def mark_delete_error(self, document_id, error):
        document_id = str(document_id)
        with self._transaction():
            self.db.execute(
                """UPDATE documents SET last_error=?,updated_at=?
                    WHERE document_id=? AND status='deleting'""",
                (str(error), time.time(), document_id),
            )
            row = self._get_unlocked(document_id)
            if row and row["pending_operation_id"]:
                self.db.execute(
                    """UPDATE lifecycle_operations SET error=?,updated_at=?
                        WHERE operation_id=?""",
                    (str(error), time.time(), row["pending_operation_id"]),
                )
            return row

    def complete_delete(
        self, document_id, *, operation_id: str | None = None
    ):
        document_id = str(document_id)
        with self._transaction():
            row = self._get_unlocked(document_id)
            if not row:
                return None
            if row["status"] == "deleted":
                return row
            if operation_id is not None and row["pending_operation_id"] != str(
                operation_id
            ):
                raise ConcurrentUpdateError("delete operation lost ownership")
            operation_id = str(operation_id or row["pending_operation_id"] or "") or None
            self.db.execute(
                """UPDATE documents SET status='deleted',pending_hash=NULL,
                          pending_version=NULL,pending_filename=NULL,
                          cleanup_version=NULL,cleanup_status=NULL,
                          cleanup_error=NULL,pending_operation_id=NULL,
                          pending_action=NULL,last_error=NULL,updated_at=?
                    WHERE document_id=?""",
                (time.time(), document_id),
            )
            if operation_id:
                self.db.execute(
                    """UPDATE lifecycle_operations SET status='deleted',
                              error=NULL,updated_at=? WHERE operation_id=?""",
                    (time.time(), operation_id),
                )
            return self._get_unlocked(document_id)

    def mark_failed(self, document_id):
        return self.abort_ingest(document_id, "marked failed")

    def mark_deleted(self, document_id):
        """Compatibility tombstone; lifecycle-aware callers use begin/delete."""

        document_id = str(document_id)
        with self._transaction():
            previous = self._get_unlocked(document_id)
            self.db.execute(
                """UPDATE documents SET status='deleted',pending_hash=NULL,
                          pending_version=NULL,pending_filename=NULL,
                          cleanup_version=NULL,cleanup_status=NULL,
                          cleanup_error=NULL,pending_operation_id=NULL,
                          pending_action=NULL,updated_at=? WHERE document_id=?""",
                (time.time(), document_id),
            )
            if (
                previous
                and previous["hash"]
                and previous["status"] in {"active", "indexing"}
            ):
                self._bump_visibility_epoch_unlocked()
            return self._get_unlocked(document_id)

    def is_active_version(self, document_id: str, version: int) -> bool:
        """Return whether a store hit belongs to the registry-visible version.

        During an update, the old committed version remains visible. During a
        delete, the tombstone hides all versions before physical cleanup.
        """

        visible = self.visible_version(document_id)
        return visible is not None and visible == int(version)

    def visible_version(self, document_id: str) -> int | None:
        """Return one consistent logical visibility decision for a document.

        Physical stores may temporarily contain both the old and the prepared
        new version during a hand-off.  Search adapters snapshot this value
        once per document and per request so a concurrent CAS cannot produce a
        mixed-version result within one search call.
        """

        with self._lock:
            row = self._get_unlocked(str(document_id))
        if not row or not row["hash"]:
            return None
        if row["status"] not in {"active", "indexing"}:
            return None
        return int(row["version"])

    def visibility_snapshot(self) -> VisibilitySnapshot:
        """Capture all committed visibility decisions in one SQLite snapshot.

        The epoch and document rows come from one SQL statement, so another
        process cannot place the token and its version map on opposite sides
        of a CAS commit.
        """

        with self._lock:
            rows = self.db.execute(
                """SELECT CAST(m.value AS INTEGER) AS epoch,
                          d.document_id,d.version
                     FROM lifecycle_metadata AS m
                     LEFT JOIN documents AS d
                       ON d.hash<>'' AND d.status IN ('active','indexing')
                    WHERE m.key='visibility_epoch'
                    ORDER BY d.document_id"""
            ).fetchall()
        if not rows:
            raise LifecycleError("visibility epoch metadata is missing")
        return VisibilitySnapshot(
            epoch=int(rows[0]["epoch"]),
            versions={
                str(row["document_id"]): int(row["version"])
                for row in rows
                if row["document_id"] is not None
            },
        )

    def is_visibility_snapshot_current(
        self, snapshot: VisibilitySnapshot
    ) -> bool:
        """Return whether no logical visibility switch followed ``snapshot``."""

        if not isinstance(snapshot, VisibilitySnapshot):
            raise TypeError("snapshot must be a VisibilitySnapshot")
        with self._lock:
            row = self.db.execute(
                """SELECT value FROM lifecycle_metadata
                    WHERE key='visibility_epoch'"""
            ).fetchone()
        return row is not None and int(row[0]) == snapshot.epoch

    def inspect(self, document_id: str | None = None):
        with self._lock:
            if document_id is not None:
                return self._get_unlocked(str(document_id))
            return [
                dict(row)
                for row in self.db.execute(
                    f"SELECT {_COLUMNS} FROM documents ORDER BY document_id"
                )
            ]

    def close(self):
        with self._lock:
            self.db.close()


class LifecycleManager:
    """Thin compatibility facade over :class:`DocumentRegistry`."""

    def __init__(self, registry: DocumentRegistry):
        self.registry = registry

    def plan(self, document_id, filename, text, **kwargs):
        return self.registry.plan(document_id, filename, text, **kwargs)

    def begin(self, document_id, filename, text, **kwargs):
        return self.registry.begin_ingest(document_id, filename, text, **kwargs)

    def commit(self, document_id, filename, text, version, **kwargs):
        return self.registry.commit(
            document_id, filename, text, version, **kwargs
        )

    def fail(self, document_id, error, **kwargs):
        return self.registry.abort_ingest(document_id, error, **kwargs)


__all__ = ["DocumentRegistry", "LifecycleManager", "VisibilitySnapshot"]
