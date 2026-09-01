"""Persistent Qdrant dense store with explicit staged visibility."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

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


class QdrantStore:
    """Dense representation of canonical chunks.

    ``upsert_chunks`` is kept as an immediate-active compatibility method.
    Lifecycle-aware ingestion must call ``stage_chunks`` and only activate the
    version after dense and sparse manifests have both been verified.
    """

    def __init__(
        self,
        collection_name: str,
        dim: int,
        persist_dir: str,
        embedding_model: str = "",
        normalization: bool | None = None,
        embedding_revision: str | None = None,
        embedding_profile_hash: str | None = None,
        distance: str = "cosine",
    ):
        if dim <= 0:
            raise ValueError("Qdrant dimension must be positive")
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = str(collection_name)
        if not self.collection_name:
            raise ValueError("Qdrant collection_name cannot be empty")
        self.dim = int(dim)
        self.distance = str(distance).casefold()
        if self.distance != "cosine":
            raise ValueError("QdrantStore currently implements cosine distance only")
        self.embedding_model = str(embedding_model or "")
        self.normalization = normalization
        self.embedding_revision = (
            str(embedding_revision) if embedding_revision is not None else None
        )
        self.embedding_profile_hash = (
            str(embedding_profile_hash)
            if embedding_profile_hash is not None
            else None
        )
        self._identity_path = (
            self.persist_dir / f".{self.collection_name}.embedding.json"
        )
        self._lock = RLock()
        self._registry = None
        self.client = QdrantClient(path=str(self.persist_dir))
        try:
            self._init_collection()
            self._validate_embedding_identity()
            self._migrate_lifecycle_payloads()
        except BaseException:
            self.client.close()
            raise

    @property
    def index_identity(self) -> dict[str, Any]:
        return {
            "kind": "qdrant-dense",
            "collection": self.collection_name,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.dim,
            "normalized": self.normalization,
            "embedding_revision": self.embedding_revision,
            "embedding_profile_hash": self.embedding_profile_hash,
            "distance": self.distance,
        }

    def bind_registry(self, registry) -> "QdrantStore":
        """Use the registry as the final cross-store visibility authority."""

        self._registry = registry
        return self

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 4,
            "dimension": self.dim,
            "model": self.embedding_model,
            "normalized": self.normalization,
            "revision": self.embedding_revision,
            "profile_hash": self.embedding_profile_hash,
            "collection": self.collection_name,
            "distance": self.distance,
        }

    def _write_identity_exclusive(self, identity: Mapping[str, Any]) -> None:
        data = json.dumps(dict(identity), sort_keys=True).encode("utf-8")
        try:
            descriptor = os.open(
                self._identity_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return
        try:
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _replace_identity(self, identity: Mapping[str, Any]) -> None:
        temporary = self._identity_path.with_suffix(
            self._identity_path.suffix + f".{uuid.uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(dict(identity), sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, self._identity_path)

    def _validate_embedding_identity(self) -> None:
        identity = self._identity_payload()
        if not self._identity_path.exists():
            self._write_identity_exclusive(identity)
        try:
            stored = json.loads(self._identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid persisted embedding identity: {self._identity_path}"
            ) from exc
        if int(stored.get("dimension", self.dim)) != self.dim:
            raise ValueError(
                "Qdrant embedding dimension does not match persisted identity"
            )
        stored_collection = str(stored.get("collection") or self.collection_name)
        if stored_collection != self.collection_name:
            raise ValueError("Qdrant identity belongs to a different collection")
        stored_distance = str(stored.get("distance") or "cosine").casefold()
        if stored_distance != self.distance:
            raise ValueError("Qdrant distance does not match persisted identity")
        stored_model = str(stored.get("model") or "")
        if stored_model and self.embedding_model and stored_model != self.embedding_model:
            raise ValueError(
                f"Qdrant collection is bound to embedding model {stored_model!r}, "
                f"not {self.embedding_model!r}"
            )
        stored_normalization = stored.get("normalized")
        if (
            stored_normalization is not None
            and self.normalization is not None
            and bool(stored_normalization) != bool(self.normalization)
        ):
            raise ValueError(
                "Qdrant normalization assumption does not match persisted identity"
            )
        stored_revision = stored.get("revision")
        if (
            stored_revision is not None
            and self.embedding_revision is not None
            and str(stored_revision) != self.embedding_revision
        ):
            raise ValueError(
                "Qdrant embedding checkpoint revision does not match persisted identity"
            )
        stored_profile_hash = stored.get("profile_hash")
        if (
            stored_profile_hash is not None
            and self.embedding_profile_hash is not None
            and str(stored_profile_hash) != self.embedding_profile_hash
        ):
            raise ValueError(
                "Qdrant embedding input profile does not match persisted identity"
            )
        missing_strict_identity = (
            (self.embedding_revision is not None and stored_revision is None)
            or (
                self.embedding_profile_hash is not None
                and stored_profile_hash is None
            )
        )
        if missing_strict_identity:
            existing = int(
                self.client.count(
                    collection_name=self.collection_name,
                    exact=True,
                ).count
            )
            if existing:
                raise ValueError(
                    "existing vectors lack checkpoint/profile identity; reindex is required"
                )
        needs_upgrade = (
            int(stored.get("schema_version", 1)) < 4
            or (self.embedding_model and not stored_model)
            or (self.normalization is not None and stored_normalization is None)
            or missing_strict_identity
            or "collection" not in stored
            or "distance" not in stored
        )
        if needs_upgrade:
            self._replace_identity(identity)

    def ensure_embedding_identity(
        self,
        model_name: str,
        dimension: int,
        normalization: bool | None = None,
        revision: str | None = None,
        profile_hash: str | None = None,
    ) -> None:
        if int(dimension) != self.dim:
            raise ValueError(
                f"embedding dimension {dimension} does not match Qdrant dimension {self.dim}"
            )
        requested = str(model_name or "")
        if requested and self.embedding_model and requested != self.embedding_model:
            raise ValueError(
                f"embedding model mismatch: {self.embedding_model!r} != {requested!r}"
            )
        if (
            normalization is not None
            and self.normalization is not None
            and bool(normalization) != bool(self.normalization)
        ):
            raise ValueError("embedding normalization mismatch")
        requested_revision = str(revision) if revision is not None else None
        requested_profile = str(profile_hash) if profile_hash is not None else None
        if (
            requested_revision is not None
            and self.embedding_revision is not None
            and requested_revision != self.embedding_revision
        ):
            raise ValueError("embedding checkpoint revision mismatch")
        if (
            requested_profile is not None
            and self.embedding_profile_hash is not None
            and requested_profile != self.embedding_profile_hash
        ):
            raise ValueError("embedding input profile mismatch")
        changed = False
        if requested and not self.embedding_model:
            self.embedding_model = requested
            changed = True
        if normalization is not None and self.normalization is None:
            self.normalization = bool(normalization)
            changed = True
        missing_strict_identity = (
            (requested_revision is not None and self.embedding_revision is None)
            or (requested_profile is not None and self.embedding_profile_hash is None)
        )
        if missing_strict_identity:
            existing = int(
                self.client.count(
                    collection_name=self.collection_name,
                    exact=True,
                ).count
            )
            if existing:
                raise ValueError(
                    "existing vectors lack checkpoint/profile identity; reindex is required"
                )
        if requested_revision is not None and self.embedding_revision is None:
            self.embedding_revision = requested_revision
            changed = True
        if requested_profile is not None and self.embedding_profile_hash is None:
            self.embedding_profile_hash = requested_profile
            changed = True
        if changed:
            self._validate_embedding_identity()

    def _init_collection(self) -> None:
        collections = {
            collection.name for collection in self.client.get_collections().collections
        }
        if self.collection_name not in collections:
            self.client.create_collection(
                self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            return
        vectors = self.client.get_collection(
            self.collection_name
        ).config.params.vectors
        existing = getattr(vectors, "size", None)
        if existing != self.dim:
            raise ValueError(
                f"collection {self.collection_name!r} has dimension {existing}, "
                f"expected {self.dim}"
            )

    @staticmethod
    def to_uuid(value: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))

    def _migrate_lifecycle_payloads(self) -> None:
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                identity = payload.get("storage_identity") or {}
                if (
                    payload.get("lifecycle_state") in {ACTIVE, PENDING, STALE}
                    and identity.get("metadata_hash")
                ):
                    continue
                chunk_id = str(payload.get("chunk_id") or point.id)
                document_id = str(payload.get("document_id") or "legacy")
                prepared = prepare_chunks(
                    [
                        {
                            "chunk_id": chunk_id,
                            "document_id": document_id,
                            "version": payload.get("version", 1),
                            "text": payload.get("text", ""),
                            "metadata": payload,
                        }
                    ],
                    state=payload.get("lifecycle_state", payload.get("status", ACTIVE)),
                    index_identity=self.index_identity,
                )[0]
                migrated_payload = {
                    **prepared["metadata"],
                    "text": prepared["text"],
                    "title": payload.get("title", ""),
                }
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=migrated_payload,
                    points=[point.id],
                )
            if offset is None:
                break

    def _prepare_points(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        state: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[PointStruct]]:
        prepared = prepare_chunks(
            chunks,
            state=state,
            operation_id=operation_id,
            index_identity=self.index_identity,
        )
        points = []
        for chunk in prepared:
            vector = np.asarray(chunk.get("vector"), dtype=np.float32)
            if (
                vector.ndim != 1
                or vector.shape[0] != self.dim
                or not np.isfinite(vector).all()
            ):
                raise ValueError(
                    f"chunk {chunk['chunk_id']} has invalid vector; "
                    f"expected finite shape ({self.dim},)"
                )
            metadata = dict(chunk["metadata"])
            payload = {
                **metadata,
                "text": chunk["text"],
                "title": chunk.get("title", ""),
            }
            points.append(
                PointStruct(
                    id=self.to_uuid(chunk["chunk_id"]),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        return prepared, points

    def _upsert_prepared(self, points: list[PointStruct]) -> None:
        if points:
            with self._lock:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )

    def upsert_chunks(self, chunks: Iterable[Mapping[str, Any]]) -> None:
        """Compatibility write: rows without an explicit state are active."""

        _, points = self._prepare_points(chunks)
        self._upsert_prepared(points)

    def stage_chunks(
        self,
        chunks: Iterable[Mapping[str, Any]],
        *,
        operation_id: str,
    ) -> None:
        _, points = self._prepare_points(
            chunks, state=PENDING, operation_id=operation_id
        )
        self._upsert_prepared(points)

    @staticmethod
    def _normalize_filters(filters):
        if filters is None:
            return None
        if isinstance(filters, (list, tuple)):
            result = list(filters)
        elif hasattr(filters, "field"):
            result = [filters]
        else:
            return None
        if not all(hasattr(item, "operator") and hasattr(item, "field") for item in result):
            raise TypeError("dense filters must be MetadataFilter instances")
        return result

    @staticmethod
    def _native_filter(filters):
        if not filters:
            return None
        must, must_not = [], []
        for predicate in filters:
            field = (
                "lifecycle_state" if predicate.field == "status" else predicate.field
            )
            operator, value = predicate.operator, predicate.value
            if operator == "=":
                must.append(FieldCondition(key=field, match=MatchValue(value=value)))
            elif operator == "!=":
                # Qdrant's must_not includes missing fields, matching the
                # shared predicate where ``None != expected`` is true.
                must_not.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
            elif operator == "in":
                values = list(value) if not isinstance(value, str) else [value]
                if not values:
                    # An impossible match without relying on a reserved field.
                    must.append(
                        FieldCondition(
                            key="chunk_id", match=MatchAny(any=["__never_a_chunk_id__"])
                        )
                    )
                else:
                    must.append(
                        FieldCondition(key=field, match=MatchAny(any=values))
                    )
            elif operator in {">", ">=", "<", "<="}:
                bounds = (
                    {"gt": value}
                    if operator == ">"
                    else {"gte": value}
                    if operator == ">="
                    else {"lt": value}
                    if operator == "<"
                    else {"lte": value}
                )
                must.append(FieldCondition(key=field, range=Range(**bounds)))
            else:
                raise ValueError(
                    f"Qdrant does not natively support {operator!r} for {field!r}"
                )
        return Filter(must=must or None, must_not=must_not or None)

    @staticmethod
    def _active_filter(native_filter: Filter | None = None) -> Filter:
        active = FieldCondition(
            key="lifecycle_state", match=MatchValue(value=ACTIVE)
        )
        if native_filter is None:
            return Filter(must=[active])
        return Filter(
            must=[active, *(native_filter.must or [])],
            must_not=native_filter.must_not,
            should=native_filter.should,
            min_should=native_filter.min_should,
        )

    @staticmethod
    def _hit(point, score: float) -> dict[str, Any]:
        payload = dict(point.payload or {})
        return {
            "_id": payload.get("chunk_id", point.id),
            "chunk_id": payload.get("chunk_id", point.id),
            "score": float(score),
            "text": payload.get("text", ""),
            "metadata": payload,
            "document_id": payload.get("document_id", ""),
            "title": payload.get("title", ""),
        }

    def _registry_visible(self, payload: Mapping[str, Any]) -> bool:
        if self._registry is None:
            return True
        checker = getattr(self._registry, "is_active_version", None)
        return bool(
            checker
            and checker(
                str(payload.get("document_id") or ""),
                int(payload.get("version", 0)),
            )
        )

    def _registry_snapshot_visible(
        self,
        payload: Mapping[str, Any],
        versions: dict[str, int | None],
        visibility_snapshot=None,
    ) -> bool:
        """Apply one registry snapshot per document for this search request."""

        document_id = str(payload.get("document_id") or "")
        if visibility_snapshot is not None:
            resolver = getattr(visibility_snapshot, "version_for", None)
            expected = (
                resolver(document_id)
                if resolver is not None
                else getattr(visibility_snapshot, "versions", visibility_snapshot).get(
                    document_id
                )
            )
            return expected is not None and int(expected) == int(
                payload.get("version", 0)
            )
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
        return versions[document_id] == int(payload.get("version", 0))

    def _exact_filter_search(
        self, vector: np.ndarray, top_k: int, filters, visibility_snapshot=None
    ) -> list[dict[str, Any]]:
        """Correctness-first complete scan for predicates such as contains."""

        matches = []
        visible_versions: dict[str, int | None] = {}
        offset = None
        query_norm = float(np.linalg.norm(vector))
        active_filter = self._active_filter()
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=active_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = dict(point.payload or {})
                if not self._registry_snapshot_visible(
                    payload, visible_versions, visibility_snapshot
                ):
                    continue
                if not all(predicate.matches(payload) for predicate in filters):
                    continue
                candidate_vector = np.asarray(point.vector, dtype=np.float32)
                denominator = query_norm * float(np.linalg.norm(candidate_vector))
                if denominator == 0.0:
                    continue
                matches.append(
                    self._hit(
                        point,
                        float(np.dot(vector, candidate_vector) / denominator),
                    )
                )
            if offset is None:
                break
        return sorted(
            matches, key=lambda hit: (-hit["score"], str(hit["chunk_id"]))
        )[: int(top_k)]

    def search(
        self,
        query_vector,
        top_k: int = 10,
        query_filter=None,
        *,
        visibility_snapshot=None,
    ):
        vector = np.asarray(query_vector, dtype=np.float32)
        if (
            vector.ndim != 1
            or vector.shape[0] != self.dim
            or not np.isfinite(vector).all()
        ):
            raise ValueError(
                f"query vector must be finite and have shape ({self.dim},)"
            )
        if top_k <= 0:
            return []
        filters = self._normalize_filters(query_filter)
        if filters and any(predicate.operator == "contains" for predicate in filters):
            return self._exact_filter_search(
                vector,
                int(top_k),
                filters,
                visibility_snapshot=visibility_snapshot,
            )
        if filters is not None:
            native_filter = self._native_filter(filters)
        elif query_filter is None:
            native_filter = None
        elif isinstance(query_filter, Filter):
            native_filter = query_filter
        else:
            raise TypeError("query_filter must be MetadataFilter instances or Qdrant Filter")
        active_filter = self._active_filter(native_filter)
        with self._lock:
            total = int(
                self.client.count(
                    collection_name=self.collection_name,
                    count_filter=active_filter,
                    exact=True,
                ).count
            )
            if total == 0:
                return []
            limit = min(total, max(int(top_k), int(top_k) * 4))
            visible_versions: dict[str, int | None] = {}
            while True:
                points = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=limit,
                    query_filter=active_filter,
                    with_payload=True,
                ).points
                visible = [
                    point
                    for point in points
                    if self._registry_snapshot_visible(
                        point.payload or {}, visible_versions, visibility_snapshot
                    )
                ]
                if len(visible) >= int(top_k) or limit >= total:
                    break
                limit = min(total, limit * 2)
        hits = [self._hit(point, float(point.score)) for point in visible]
        return sorted(
            hits, key=lambda hit: (-hit["score"], str(hit["chunk_id"]))
        )[: int(top_k)]

    @staticmethod
    def _document_filter(
        document_id: str,
        *,
        version: int | None = None,
        operation_id: str | None = None,
        state: str | None = None,
        exclude_version: int | None = None,
    ) -> Filter:
        must = [
            FieldCondition(
                key="document_id", match=MatchValue(value=str(document_id))
            )
        ]
        must_not = []
        if version is not None:
            must.append(
                FieldCondition(key="version", match=MatchValue(value=int(version)))
            )
        if operation_id is not None:
            must.append(
                FieldCondition(
                    key="lifecycle_operation_id",
                    match=MatchValue(value=str(operation_id)),
                )
            )
        if state is not None:
            must.append(
                FieldCondition(
                    key="lifecycle_state", match=MatchValue(value=str(state))
                )
            )
        if exclude_version is not None:
            must_not.append(
                FieldCondition(
                    key="version", match=MatchValue(value=int(exclude_version))
                )
            )
        return Filter(must=must, must_not=must_not or None)

    def _scroll_ids(self, query_filter: Filter) -> list[Any]:
        identifiers = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            identifiers.extend(point.id for point in points)
            if offset is None:
                break
        return identifiers

    def _set_state(self, identifiers: Iterable[Any], state: str) -> None:
        ids = list(identifiers)
        if not ids:
            return
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"lifecycle_state": state, "status": state},
            points=ids,
            wait=True,
        )

    def activate_version(
        self,
        document_id: str,
        version: int,
        *,
        operation_id: str | None = None,
    ) -> None:
        """Make the target searchable without retiring the committed old version."""

        with self._lock:
            target_filter = self._document_filter(
                document_id, version=version, operation_id=operation_id
            )
            targets = self._scroll_ids(target_filter)
            if not targets:
                raise RuntimeError(
                    f"dense version {document_id!r}/v{int(version)} is not staged"
                )
            # This is a physical readiness transition, not the logical
            # version switch.  Keep the previous version searchable until the
            # registry CAS commits the target.  The coordinator retires or
            # deletes old versions only after that commit succeeds.
            self._set_state(targets, ACTIVE)

    def deactivate_version(
        self, document_id: str, version: int, *, state: str = STALE
    ) -> None:
        """Idempotently remove a version from the searchable state."""

        if state == ACTIVE:
            raise ValueError("deactivate_version state cannot be active")
        with self._lock:
            identifiers = self._scroll_ids(
                self._document_filter(document_id, version=int(version))
            )
            self._set_state(identifiers, state)

    def delete_document(
        self,
        document_id: str,
        keep_version: int | None = None,
        only_version: int | None = None,
    ) -> None:
        selector = self._document_filter(
            document_id,
            version=int(only_version) if only_version is not None else None,
            exclude_version=(
                int(keep_version)
                if keep_version is not None and only_version is None
                else None
            ),
        )
        with self._lock:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=selector,
                wait=True,
            )

    def rollback_version(
        self,
        document_id: str,
        version: int,
        *,
        restore_version: int | None = None,
    ) -> None:
        with self._lock:
            self.delete_document(document_id, only_version=version)
            if restore_version is not None:
                restore_ids = self._scroll_ids(
                    self._document_filter(document_id, version=int(restore_version))
                )
                self._set_state(restore_ids, ACTIVE)

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
        identifiers = [self.to_uuid(chunk["chunk_id"]) for chunk in expected]
        with self._lock:
            found = self.client.retrieve(
                collection_name=self.collection_name,
                ids=identifiers,
                with_payload=True,
                with_vectors=True,
            )
        by_id = {
            str((point.payload or {}).get("chunk_id") or point.id): point
            for point in found
        }
        if len(by_id) != len({chunk["chunk_id"] for chunk in expected}):
            return False
        for chunk in expected:
            point = by_id.get(chunk["chunk_id"])
            if point is None:
                return False
            payload = point.payload or {}
            identity = chunk["metadata"]["storage_identity"]
            stored_identity = payload.get("storage_identity") or {}
            vector = np.asarray(point.vector, dtype=np.float32)
            if (
                str(payload.get("document_id") or "") != chunk["document_id"]
                or int(payload.get("version", 0)) != chunk["version"]
                or stored_identity.get("metadata_hash") != identity["metadata_hash"]
                or stored_identity.get("text_hash") != identity["text_hash"]
                or vector.shape != (self.dim,)
                or not np.isfinite(vector).all()
                or (
                    state is not None
                    and str(payload.get("lifecycle_state") or "") != state
                )
                or (
                    operation_id is not None
                    and str(payload.get("lifecycle_operation_id") or "")
                    != str(operation_id)
                )
            ):
                return False
        return True

    def get_manifest(
        self, document_id: str, version: int | None = None
    ) -> dict[str, ChunkManifestEntry]:
        query_filter = self._document_filter(document_id, version=version)
        entries = []
        offset = None
        with self._lock:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                entries.extend(
                    manifest_entry(point.payload or {}) for point in points
                )
                if offset is None:
                    break
        return {entry.chunk_id: entry for entry in entries}

    def list_chunks(
        self,
        document_id: str,
        *,
        version: int | None = None,
        state: str = ACTIVE,
        limit: int = 5_000,
    ) -> list[dict[str, Any]]:
        """Return deterministic JSON-safe chunk data without embedding vectors.

        For the default ``active`` state, a bound registry is authoritative:
        omitting ``version`` resolves the committed version, while requesting
        a different version returns no rows.  Qdrant is always scrolled with
        ``with_vectors=False``.
        """

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

        query_filter = self._document_filter(
            document_id,
            version=resolved_version,
            state=state,
        )
        rows: list[dict[str, Any]] = []
        offset = None
        with self._lock:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = dict(point.payload or {})
                    metadata = {
                        key: value
                        for key, value in payload.items()
                        if key not in {"text", "title"}
                    }
                    if payload.get("title"):
                        metadata.setdefault("title", payload["title"])
                    # Qdrant payloads are JSON-compatible, but round-tripping
                    # defensively also normalizes any custom scalar type.
                    metadata = json.loads(
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        )
                    )
                    chunk_id = str(payload.get("chunk_id") or point.id)
                    item_version = int(payload.get("version", 0))
                    item_state = str(
                        payload.get("lifecycle_state")
                        or payload.get("status")
                        or ""
                    )
                    rows.append(
                        {
                            "chunk_id": chunk_id,
                            "document_id": str(
                                payload.get("document_id") or document_id
                            ),
                            "version": item_version,
                            "state": item_state,
                            "text": str(payload.get("text") or ""),
                            "metadata": metadata,
                        }
                    )
                if offset is None:
                    break
        return sorted(
            rows,
            key=lambda item: (int(item["version"]), str(item["chunk_id"])),
        )[:limit]

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
        with self._lock:
            return int(
                self.client.count(
                    collection_name=self.collection_name,
                    count_filter=self._document_filter(document_id, version=version),
                    exact=True,
                ).count
            )

    def drop_collection(self) -> None:
        with self._lock:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)

    def reset(self) -> None:
        self.drop_collection()
        self._init_collection()

    def close(self) -> None:
        with self._lock:
            self.client.close()


__all__ = ["QdrantStore"]
