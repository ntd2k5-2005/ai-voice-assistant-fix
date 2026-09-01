"""Persistent SQLite FTS5 sparse store with lifecycle-safe visibility."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
from pathlib import Path
import re
import sqlite3
from threading import RLock
from typing import Any

from ..contracts import MetadataFilter, RetrievalCandidate
from ..sparse_analyzer import SparseAnalyzer, resolve_sparse_analyzer
from .identity import (
    ACTIVE,
    LIFECYCLE_STATES,
    PENDING,
    STALE,
    ChunkManifestEntry,
    manifest_entry,
    prepare_chunks,
)


_MAX_LIST_CHUNKS = 10_000

_STRUCTURED_COLUMNS = {
    "document_id",
    "version",
    "source",
    "file_type",
    "page_number",
    "section",
    "sheet_name",
    "slide_number",
    "status",
    "lifecycle_state",
}


class SQLiteBM25Retriever:
    """Transactional FTS5 representation of canonical chunks.

    ``upsert`` remains an immediate-active compatibility API. Production
    ingestion uses ``stage_chunks`` followed by ``activate_version`` so a
    write cannot become searchable before both representations are verified.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        analyzer: str | SparseAnalyzer = "unicode61",
        tokenizer: str | None = None,
        query_operator: str = "or",
        timeout_seconds: float = 30.0,
    ):
        self.path = str(path)
        if tokenizer is not None:
            requested = analyzer.name if isinstance(analyzer, SparseAnalyzer) else str(analyzer)
            if requested != "unicode61" and requested != str(tokenizer):
                raise ValueError("analyzer and tokenizer compatibility alias conflict")
            analyzer = tokenizer
        self.analyzer = resolve_sparse_analyzer(analyzer)
        self.tokenizer = self.analyzer.sqlite_tokenizer
        self.query_operator = str(query_operator).casefold()
        if self.query_operator not in {"or", "and"}:
            raise ValueError("query_operator must be 'or' or 'and'")
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
            check_same_thread=False,
        )
        self.db.row_factory = sqlite3.Row
        self._lock = RLock()
        self._registry = None
        self._initialize()

    @property
    def index_identity(self) -> dict[str, Any]:
        return {
            "kind": "sqlite-fts5",
            "schema_version": 3,
            "analyzer": self.analyzer.name,
            "sqlite_tokenizer": self.tokenizer,
            "query_operator": self.query_operator,
        }

    def bind_registry(self, registry) -> "SQLiteBM25Retriever":
        """Use the registry as the final cross-store visibility authority."""

        self._registry = registry
        return self

    def _initialize(self) -> None:
        with self._lock, self.db:
            if self.path != ":memory:":
                self.db.execute("PRAGMA journal_mode=WAL")
                self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute(
                f"PRAGMA busy_timeout={max(1, round(self.timeout_seconds * 1000))}"
            )
            self.db.execute("PRAGMA foreign_keys=ON")
            # Tokenizer names cannot be bound parameters. Only this validated
            # built-in tokenizer is currently promised by the adapter.
            if self.tokenizer != "unicode61":
                raise ValueError("SQLiteBM25Retriever currently supports tokenizer='unicode61'")
            self.db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61')"
            )
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS chunk_metadata(
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version > 0),
                    source TEXT,
                    file_type TEXT,
                    page_number INTEGER,
                    section TEXT,
                    sheet_name TEXT,
                    slide_number INTEGER,
                    status TEXT NOT NULL DEFAULT 'active',
                    lifecycle_state TEXT NOT NULL DEFAULT 'active',
                    lifecycle_operation_id TEXT,
                    identity_hash TEXT NOT NULL DEFAULT '',
                    text_hash TEXT NOT NULL DEFAULT '',
                    original_text TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL
                )"""
            )
            existing = {
                row[1]: row for row in self.db.execute("PRAGMA table_info(chunk_metadata)")
            }
            additions = (
                ("lifecycle_state", "TEXT NOT NULL DEFAULT 'active'"),
                ("lifecycle_operation_id", "TEXT"),
                ("identity_hash", "TEXT NOT NULL DEFAULT ''"),
                ("text_hash", "TEXT NOT NULL DEFAULT ''"),
                ("original_text", "TEXT NOT NULL DEFAULT ''"),
            )
            for name, declaration in additions:
                if name not in existing:
                    self.db.execute(
                        f"ALTER TABLE chunk_metadata ADD COLUMN {name} {declaration}"
                    )
            self.db.execute(
                """UPDATE chunk_metadata
                      SET original_text=COALESCE(
                          (SELECT f.text FROM chunks_fts AS f
                            WHERE f.chunk_id=chunk_metadata.chunk_id), '')
                    WHERE original_text=''"""
            )
            for column in _STRUCTURED_COLUMNS - {"status", "lifecycle_state"}:
                self.db.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_chunk_metadata_{column} "
                    f"ON chunk_metadata({column})"
                )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_metadata_document_version_state "
                "ON chunk_metadata(document_id,version,lifecycle_state)"
            )
            self._migrate_legacy_chunks()
            self._migrate_identity_rows()

    def _migrate_legacy_chunks(self) -> None:
        """Preserve searchable legacy rows without evaluating Python repr."""

        legacy = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
        ).fetchone()
        if not legacy:
            return
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(chunks)")}
        required = {"chunk_id", "text", "metadata"}
        if not required.issubset(columns):
            return
        select_document = "document_id" if "document_id" in columns else "''"
        select_version = "version" if "version" in columns else "1"
        rows = self.db.execute(
            f"SELECT chunk_id,{select_document},{select_version},text,metadata FROM chunks"
        ).fetchall()
        migrated = []
        for row in rows:
            raw = row[4] or "{}"
            try:
                metadata = json.loads(raw)
                if not isinstance(metadata, dict):
                    metadata = {
                        "legacy_metadata": raw,
                        "migration_status": "reindex_recommended",
                    }
            except json.JSONDecodeError:
                metadata = {
                    "legacy_metadata": raw,
                    "migration_status": "reindex_recommended",
                }
            document_id = str(row[1] or metadata.get("document_id") or "legacy")
            migrated.append(
                {
                    "chunk_id": str(row[0]),
                    "document_id": document_id,
                    "version": int(row[2] or 1),
                    "text": str(row[3] or ""),
                    "metadata": metadata,
                }
            )
        for chunk in prepare_chunks(migrated, state=ACTIVE):
            self._delete_chunks([chunk["chunk_id"]])
            self._insert_row(chunk)
        self.db.execute("DROP TABLE chunks")

    def _migrate_identity_rows(self) -> None:
        rows = self.db.execute(
            """SELECT f.chunk_id,COALESCE(NULLIF(m.original_text,''),f.text),
                      m.document_id,m.version,m.status,
                      m.lifecycle_state,m.metadata_json,m.identity_hash
                 FROM chunks_fts AS f JOIN chunk_metadata AS m
                   ON f.chunk_id=m.chunk_id
                WHERE m.identity_hash='' OR m.identity_hash IS NULL"""
        ).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row[6])
            except (TypeError, json.JSONDecodeError):
                metadata = {"migration_status": "reindex_recommended"}
            prepared = prepare_chunks(
                [
                    {
                        "chunk_id": row[0],
                        "document_id": row[2],
                        "version": row[3],
                        "text": row[1],
                        "metadata": metadata,
                    }
                ],
                state=row[5] or row[4] or ACTIVE,
            )[0]
            identity = prepared["metadata"]["storage_identity"]
            self.db.execute(
                """UPDATE chunk_metadata
                      SET status=?,lifecycle_state=?,lifecycle_operation_id=?,
                          identity_hash=?,text_hash=?,metadata_json=?
                    WHERE chunk_id=?""",
                (
                    prepared["metadata"]["status"],
                    prepared["metadata"]["lifecycle_state"],
                    prepared["metadata"].get("lifecycle_operation_id"),
                    identity["metadata_hash"],
                    identity["text_hash"],
                    self._json(prepared["metadata"]),
                    prepared["chunk_id"],
                ),
            )

    @staticmethod
    def _json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _integer(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _insert_row(self, chunk: Mapping[str, Any]) -> None:
        metadata = dict(chunk["metadata"])
        identity = metadata["storage_identity"]
        original_text = str(chunk["text"])
        self.db.execute(
            "INSERT INTO chunks_fts(chunk_id,text) VALUES(?,?)",
            (chunk["chunk_id"], self.analyzer.index_text(original_text)),
        )
        self.db.execute(
            """INSERT INTO chunk_metadata(
                    chunk_id,document_id,version,source,file_type,page_number,
                    section,sheet_name,slide_number,status,lifecycle_state,
                    lifecycle_operation_id,identity_hash,text_hash,original_text,
                    metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                chunk["chunk_id"],
                chunk["document_id"],
                chunk["version"],
                metadata.get("source") or metadata.get("source_path"),
                metadata.get("file_type"),
                self._integer(metadata.get("page_number", metadata.get("page"))),
                metadata.get("section"),
                metadata.get("sheet_name"),
                self._integer(metadata.get("slide_number")),
                metadata["status"],
                metadata["lifecycle_state"],
                metadata.get("lifecycle_operation_id"),
                identity["metadata_hash"],
                identity["text_hash"],
                original_text,
                self._json(metadata),
            ),
        )

    def _delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        for chunk_id in chunk_ids:
            self.db.execute(
                "DELETE FROM chunks_fts WHERE chunk_id=?", (str(chunk_id),)
            )
            self.db.execute(
                "DELETE FROM chunk_metadata WHERE chunk_id=?", (str(chunk_id),)
            )

    def _write_prepared(self, chunks: list[dict[str, Any]]) -> None:
        with self._lock, self.db:
            for chunk in chunks:
                self._delete_chunks([chunk["chunk_id"]])
                self._insert_row(chunk)

    def upsert(self, chunks: Iterable[Mapping[str, Any]]) -> None:
        """Compatibility write: rows without an explicit state are active."""

        self._write_prepared(
            prepare_chunks(chunks, index_identity=self.index_identity)
        )

    def stage_chunks(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        operation_id: str,
    ) -> None:
        self._write_prepared(
            prepare_chunks(
                chunks,
                state=PENDING,
                operation_id=operation_id,
                index_identity=self.index_identity,
            )
        )

    def delete(self, chunk_ids: str | Iterable[str]) -> None:
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        with self._lock, self.db:
            self._delete_chunks(chunk_ids)

    def delete_document(
        self,
        document_id: str,
        keep_version: int | None = None,
        only_version: int | None = None,
    ) -> None:
        clauses = ["document_id=?"]
        params: list[Any] = [str(document_id)]
        if only_version is not None:
            clauses.append("version=?")
            params.append(int(only_version))
        elif keep_version is not None:
            clauses.append("version<>?")
            params.append(int(keep_version))
        with self._lock, self.db:
            rows = self.db.execute(
                f"SELECT chunk_id FROM chunk_metadata WHERE {' AND '.join(clauses)}",
                params,
            ).fetchall()
            self._delete_chunks([row[0] for row in rows])

    def _set_state_for_ids(self, chunk_ids: Iterable[str], state: str) -> None:
        for chunk_id in chunk_ids:
            row = self.db.execute(
                "SELECT metadata_json FROM chunk_metadata WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            if not row:
                continue
            metadata = json.loads(row[0])
            metadata["lifecycle_state"] = state
            metadata["status"] = state
            self.db.execute(
                """UPDATE chunk_metadata
                      SET status=?,lifecycle_state=?,metadata_json=?
                    WHERE chunk_id=?""",
                (state, state, self._json(metadata), chunk_id),
            )

    def activate_version(
        self,
        document_id: str,
        version: int,
        *,
        operation_id: str | None = None,
    ) -> None:
        """Make a staged version physically searchable before registry CAS.

        The previous version intentionally remains searchable.  Registry
        visibility is the logical switch; post-commit cleanup retires the old
        rows after the new version is durably selected.
        """

        document_id, version = str(document_id), int(version)
        with self._lock, self.db:
            clauses = ["document_id=?", "version=?"]
            params: list[Any] = [document_id, version]
            if operation_id is not None:
                clauses.append("lifecycle_operation_id=?")
                params.append(str(operation_id))
            targets = [
                row[0]
                for row in self.db.execute(
                    f"SELECT chunk_id FROM chunk_metadata WHERE {' AND '.join(clauses)}",
                    params,
                )
            ]
            if not targets:
                raise RuntimeError(
                    f"sparse version {document_id!r}/v{version} is not staged"
                )
            self._set_state_for_ids(targets, ACTIVE)

    def deactivate_version(
        self, document_id: str, version: int, *, state: str = STALE
    ) -> None:
        """Idempotently remove a version from the searchable state."""

        if state == ACTIVE:
            raise ValueError("deactivate_version state cannot be active")
        with self._lock, self.db:
            rows = self.db.execute(
                "SELECT chunk_id FROM chunk_metadata WHERE document_id=? AND version=?",
                (str(document_id), int(version)),
            ).fetchall()
            self._set_state_for_ids([row[0] for row in rows], state)

    def rollback_version(
        self,
        document_id: str,
        version: int,
        *,
        restore_version: int | None = None,
    ) -> None:
        """Idempotently remove a failed target and restore the old version."""

        with self._lock, self.db:
            self.delete_document(document_id, only_version=version)
            if restore_version is not None:
                rows = self.db.execute(
                    "SELECT chunk_id FROM chunk_metadata WHERE document_id=? AND version=?",
                    (str(document_id), int(restore_version)),
                ).fetchall()
                self._set_state_for_ids([row[0] for row in rows], ACTIVE)

    def verify_chunks(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        state: str | None = None,
        operation_id: str | None = None,
    ) -> bool:
        expected = prepare_chunks(chunks)
        if not expected:
            return True
        with self._lock:
            for chunk in expected:
                identity = chunk["metadata"]["storage_identity"]
                row = self.db.execute(
                    """SELECT document_id,version,lifecycle_state,
                              lifecycle_operation_id,identity_hash,text_hash
                         FROM chunk_metadata WHERE chunk_id=?""",
                    (chunk["chunk_id"],),
                ).fetchone()
                if not row:
                    return False
                if (
                    str(row[0]) != chunk["document_id"]
                    or int(row[1]) != chunk["version"]
                    or str(row[4]) != identity["metadata_hash"]
                    or str(row[5]) != identity["text_hash"]
                    or (state is not None and str(row[2]) != state)
                    or (
                        operation_id is not None
                        and str(row[3] or "") != str(operation_id)
                    )
                ):
                    return False
        return True

    def get_manifest(
        self, document_id: str, version: int | None = None
    ) -> dict[str, ChunkManifestEntry]:
        clauses = ["document_id=?"]
        params: list[Any] = [str(document_id)]
        if version is not None:
            clauses.append("version=?")
            params.append(int(version))
        with self._lock:
            rows = self.db.execute(
                f"SELECT metadata_json FROM chunk_metadata WHERE {' AND '.join(clauses)}",
                params,
            ).fetchall()
        entries = [manifest_entry(json.loads(row[0])) for row in rows]
        return {entry.chunk_id: entry for entry in entries}

    def list_chunks(
        self,
        document_id: str,
        *,
        version: int | None = None,
        state: str = ACTIVE,
        limit: int = 5_000,
    ) -> list[dict[str, Any]]:
        """Return deterministic JSON-safe chunks with original, unanalysed text."""

        document_id = str(document_id).strip()
        if not document_id:
            raise ValueError("document_id must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit <= 0 or limit > _MAX_LIST_CHUNKS:
            raise ValueError(
                f"limit must be between 1 and {_MAX_LIST_CHUNKS}"
            )
        state = str(state).casefold()
        if state not in LIFECYCLE_STATES:
            raise ValueError(f"unsupported lifecycle state {state!r}")
        if version is not None:
            if isinstance(version, bool) or not isinstance(version, int):
                raise TypeError("version must be an integer")
            if version <= 0:
                raise ValueError("version must be positive")

        resolved_version = version
        if state == ACTIVE and self._registry is not None:
            resolver = getattr(self._registry, "visible_version", None)
            visible = resolver(document_id) if callable(resolver) else None
            if visible is None:
                return []
            if resolved_version is not None and resolved_version != int(visible):
                return []
            resolved_version = int(visible)

        clauses = ["m.document_id=?", "m.lifecycle_state=?"]
        params: list[Any] = [document_id, state]
        if resolved_version is not None:
            clauses.append("m.version=?")
            params.append(int(resolved_version))
        params.append(limit)
        with self._lock:
            rows = self.db.execute(
                f"""SELECT m.chunk_id,m.document_id,m.version,
                           m.lifecycle_state,
                           COALESCE(NULLIF(m.original_text,''),f.text),
                           m.metadata_json
                      FROM chunk_metadata AS m JOIN chunks_fts AS f
                        ON f.chunk_id=m.chunk_id
                     WHERE {' AND '.join(clauses)}
                     ORDER BY m.version ASC,m.chunk_id ASC
                     LIMIT ?""",
                params,
            ).fetchall()
        result = []
        for row in rows:
            metadata = json.loads(row[5])
            metadata["document_id"] = str(row[1])
            metadata["version"] = int(row[2])
            metadata["lifecycle_state"] = str(row[3])
            metadata["status"] = str(row[3])
            result.append(
                {
                    "chunk_id": str(row[0]),
                    "document_id": str(row[1]),
                    "version": int(row[2]),
                    "state": str(row[3]),
                    "text": str(row[4] or ""),
                    "metadata": metadata,
                }
            )
        return result

    def verify_version(
        self,
        document_id: str,
        version: int,
        *,
        state: str | None = None,
        operation_id: str | None = None,
    ) -> bool:
        manifest = self.get_manifest(document_id, version)
        if not manifest:
            return False
        return all(
            entry.document_id == str(document_id)
            and entry.version == int(version)
            and bool(entry.identity_hash)
            and (state is None or entry.state == state)
            and (
                operation_id is None
                or entry.operation_id == str(operation_id)
            )
            for entry in manifest.values()
        )

    def count_document(
        self, document_id: str, *, version: int | None = None
    ) -> int:
        clauses = ["document_id=?"]
        params: list[Any] = [str(document_id)]
        if version is not None:
            clauses.append("version=?")
            params.append(int(version))
        with self._lock:
            row = self.db.execute(
                f"SELECT COUNT(*) FROM chunk_metadata WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
        return int(row[0])

    @staticmethod
    def _as_filters(filters) -> list[MetadataFilter]:
        if filters is None:
            return []
        if isinstance(filters, MetadataFilter):
            return [filters]
        result = list(filters)
        if not all(isinstance(item, MetadataFilter) for item in result):
            raise TypeError("filters must be MetadataFilter instances")
        return result

    @staticmethod
    def _json_path(field: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
            raise ValueError(f"unsupported metadata field: {field!r}")
        return f"$.{field}"

    def _filter_sql(self, filters) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for predicate in self._as_filters(filters):
            field = (
                "lifecycle_state" if predicate.field == "status" else predicate.field
            )
            structured = field in _STRUCTURED_COLUMNS
            expression = f"m.{field}" if structured else "json_extract(m.metadata_json, ?)"
            path = None if structured else self._json_path(field)
            operator, value = predicate.operator, predicate.value

            if operator == "in":
                values = list(value) if not isinstance(value, str) else [value]
                if not values:
                    clauses.append("0")
                    continue
                clauses.append(
                    f"{expression} IN ({','.join('?' for _ in values)})"
                )
                if path is not None:
                    params.append(path)
                params.extend(values)
                continue

            if operator == "contains":
                if path is None:
                    clauses.append(
                        f"instr(CAST({expression} AS TEXT),CAST(? AS TEXT)) > 0"
                    )
                    params.append(value)
                else:
                    clauses.append(
                        """(CASE json_type(m.metadata_json, ?)
                            WHEN 'text' THEN instr(
                                CAST(json_extract(m.metadata_json, ?) AS TEXT),
                                CAST(? AS TEXT)) > 0
                            WHEN 'array' THEN EXISTS (
                                SELECT 1 FROM json_each(m.metadata_json, ?) AS item
                                WHERE item.value=?)
                            WHEN 'object' THEN EXISTS (
                                SELECT 1 FROM json_each(m.metadata_json, ?) AS item
                                WHERE item.key=CAST(? AS TEXT))
                            ELSE 0 END)"""
                    )
                    params.extend([path, path, value, path, value, path, value])
                continue

            sql_operator = {
                "=": "=",
                "!=": "!=",
                ">": ">",
                ">=": ">=",
                "<": "<",
                "<=": "<=",
            }[operator]
            if operator == "!=":
                clauses.append(f"COALESCE({expression} != ?,1)")
            else:
                clauses.append(f"{expression} {sql_operator} ?")
            if path is not None:
                params.append(path)
            params.append(value)
        return clauses, params

    def _fts_query(self, query: str) -> str:
        # Quoting every token prevents words such as AND/NOT from becoming FTS
        # syntax. OR is the robust default for natural-language queries whose
        # stopwords need not occur verbatim in every relevant passage; strict
        # all-term matching remains an explicit configuration option.
        tokens = self.analyzer.tokens(query)
        operator = f" {self.query_operator.upper()} "
        return operator.join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)

    def _registry_visible(self, document_id: str, version: int) -> bool:
        if self._registry is None:
            return True
        checker = getattr(self._registry, "is_active_version", None)
        return bool(checker and checker(document_id, version))

    def _registry_snapshot_visible(
        self,
        document_id: str,
        version: int,
        versions: dict[str, int | None],
        visibility_snapshot=None,
    ) -> bool:
        """Apply one registry snapshot per document for this search request."""

        document_id = str(document_id)
        if visibility_snapshot is not None:
            resolver = getattr(visibility_snapshot, "version_for", None)
            expected = (
                resolver(document_id)
                if resolver is not None
                else getattr(visibility_snapshot, "versions", visibility_snapshot).get(
                    document_id
                )
            )
            return expected is not None and int(expected) == int(version)
        if self._registry is None:
            return True
        if document_id not in versions:
            resolver = getattr(self._registry, "visible_version", None)
            if resolver is not None:
                versions[document_id] = resolver(document_id)
            else:  # Compatibility for a custom registry implementation.
                row = self._registry.get(document_id)
                versions[document_id] = (
                    int(row["version"])
                    if row
                    and row.get("hash")
                    and row.get("status") in {"active", "indexing"}
                    else None
                )
        return versions[document_id] == int(version)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters=None,
        *,
        visibility_snapshot=None,
    ) -> list[RetrievalCandidate]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            return []
        match = self._fts_query(query)
        if not match:
            return []
        clauses, params = self._filter_sql(filters)
        # Active state is a non-overridable visibility gate. User filters are
        # additional predicates, never a way to surface pending/stale rows.
        where = "chunks_fts MATCH ? AND m.lifecycle_state='active'"
        if clauses:
            where += " AND " + " AND ".join(clauses)
        with self._lock:
            rows = self.db.execute(
                f"""SELECT f.chunk_id,m.document_id,m.version,m.original_text,
                           m.metadata_json,bm25(chunks_fts) AS rank
                      FROM chunks_fts AS f JOIN chunk_metadata AS m
                        ON f.chunk_id=m.chunk_id
                     WHERE {where}
                     ORDER BY rank ASC,f.chunk_id ASC""",
                [match, *params],
            )
            result = []
            visible_versions: dict[str, int | None] = {}
            for row in rows:
                if not self._registry_snapshot_visible(
                    str(row[1]),
                    int(row[2]),
                    visible_versions,
                    visibility_snapshot,
                ):
                    continue
                metadata = json.loads(row[4])
                metadata["document_id"] = str(row[1])
                metadata["version"] = int(row[2])
                score = float(-row[5])
                result.append(
                    RetrievalCandidate(
                        str(row[0]),
                        str(row[1]),
                        str(row[3]),
                        metadata,
                        score,
                        "sparse",
                        sparse_score=score,
                    )
                )
                if len(result) >= int(top_k):
                    break
        return result

    def close(self) -> None:
        with self._lock:
            self.db.close()


__all__ = ["SQLiteBM25Retriever"]
