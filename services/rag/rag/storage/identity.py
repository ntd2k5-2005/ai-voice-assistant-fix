"""Backend-neutral chunk identity and lifecycle invariants.

Dense and sparse stores are separate physical representations of the same
logical chunk.  This module is deliberately dependency-free so both adapters
canonicalize identity in exactly the same way before they write anything.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from ..errors import StorageConsistencyError


ACTIVE = "active"
PENDING = "pending"
STALE = "stale"
DELETING = "deleting"
FAILED = "failed"
LIFECYCLE_STATES = frozenset({ACTIVE, PENDING, STALE, DELETING, FAILED})

_MUTABLE_IDENTITY_FIELDS = {
    "lifecycle_state",
    "lifecycle_operation_id",
    "status",
    "storage_identity",
    "index_identity",
}


class ChunkIdentityError(StorageConsistencyError, ValueError):
    """Raised before a store write when a chunk identity is ambiguous."""


@dataclass(frozen=True)
class ChunkManifestEntry:
    """Small cross-store manifest used to verify dense/sparse parity."""

    chunk_id: str
    document_id: str
    version: int
    identity_hash: str
    state: str
    operation_id: str | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _positive_version(value: Any) -> int:
    if isinstance(value, bool):
        raise ChunkIdentityError("chunk version must be a positive integer")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ChunkIdentityError("chunk version must be a positive integer") from exc
    if version <= 0:
        raise ChunkIdentityError("chunk version must be a positive integer")
    return version


def prepare_chunk(
    raw: Mapping[str, Any],
    *,
    state: str | None = None,
    operation_id: str | None = None,
    index_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy with one canonical logical identity.

    Conflicting top-level and metadata identifiers are rejected instead of
    silently letting Qdrant and SQLite choose different values.
    """

    if not isinstance(raw, Mapping):
        raise ChunkIdentityError("each chunk must be a mapping")
    chunk = dict(raw)
    metadata_raw = chunk.get("metadata") or {}
    if not isinstance(metadata_raw, Mapping):
        raise ChunkIdentityError("chunk metadata must be a mapping")
    metadata = dict(metadata_raw)

    top_chunk_id = chunk.get("chunk_id") or chunk.get("_id") or chunk.get("id")
    meta_chunk_id = metadata.get("chunk_id")
    if top_chunk_id is None and meta_chunk_id is None:
        raise ChunkIdentityError("each chunk requires deterministic chunk_id")
    chunk_id = str(top_chunk_id if top_chunk_id is not None else meta_chunk_id)
    if not chunk_id:
        raise ChunkIdentityError("each chunk requires deterministic chunk_id")
    if meta_chunk_id is not None and str(meta_chunk_id) != chunk_id:
        raise ChunkIdentityError(f"chunk_id mismatch for {chunk_id!r}")

    top_document_id = chunk.get("document_id")
    meta_document_id = metadata.get("document_id")
    if top_document_id is None and meta_document_id is None:
        raise ChunkIdentityError(f"chunk {chunk_id!r} requires document_id")
    document_id = str(top_document_id if top_document_id is not None else meta_document_id)
    if not document_id:
        raise ChunkIdentityError(f"chunk {chunk_id!r} requires document_id")
    if meta_document_id is not None and str(meta_document_id) != document_id:
        raise ChunkIdentityError(f"document_id mismatch for chunk {chunk_id!r}")

    top_version = chunk.get("version")
    meta_version = metadata.get("version")
    if top_version is not None and meta_version is not None:
        if _positive_version(top_version) != _positive_version(meta_version):
            raise ChunkIdentityError(f"version mismatch for chunk {chunk_id!r}")
    version = _positive_version(
        top_version if top_version is not None else meta_version if meta_version is not None else 1
    )

    resolved_state = str(
        state
        if state is not None
        else metadata.get("lifecycle_state", metadata.get("status", ACTIVE))
    ).lower()
    if resolved_state not in LIFECYCLE_STATES:
        raise ChunkIdentityError(
            f"unsupported lifecycle state {resolved_state!r} for chunk {chunk_id!r}"
        )

    resolved_operation = operation_id
    if resolved_operation is None:
        resolved_operation = metadata.get("lifecycle_operation_id")
    if resolved_operation is not None:
        resolved_operation = str(resolved_operation)
        if not resolved_operation:
            raise ChunkIdentityError("lifecycle operation id cannot be empty")

    text = str(chunk.get("text", ""))
    logical_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in _MUTABLE_IDENTITY_FIELDS
    }
    logical_metadata.update(
        {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "version": version,
        }
    )
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity_hash = hashlib.sha256(
        canonical_json(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "version": version,
                "text_hash": text_hash,
                "metadata": logical_metadata,
            }
        ).encode("utf-8")
    ).hexdigest()

    storage_identity = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "version": version,
        "text_hash": text_hash,
        "metadata_hash": identity_hash,
    }
    metadata = logical_metadata
    metadata["lifecycle_state"] = resolved_state
    # ``status`` is retained as the legacy/public alias, but it can no longer
    # drift from the lifecycle gate used by search.
    metadata["status"] = resolved_state
    metadata["storage_identity"] = storage_identity
    if resolved_operation is not None:
        metadata["lifecycle_operation_id"] = resolved_operation
    if index_identity:
        metadata["index_identity"] = dict(index_identity)

    chunk.update(
        {
            "_id": chunk_id,
            "chunk_id": chunk_id,
            "document_id": document_id,
            "version": version,
            "text": text,
            "metadata": metadata,
        }
    )
    return chunk


def prepare_chunks(
    chunks: Iterable[Mapping[str, Any]],
    *,
    state: str | None = None,
    operation_id: str | None = None,
    index_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    prepared = [
        prepare_chunk(
            chunk,
            state=state,
            operation_id=operation_id,
            index_identity=index_identity,
        )
        for chunk in chunks
    ]
    identities: dict[str, str] = {}
    for chunk in prepared:
        chunk_id = chunk["chunk_id"]
        identity_hash = chunk["metadata"]["storage_identity"]["metadata_hash"]
        previous = identities.setdefault(chunk_id, identity_hash)
        if previous != identity_hash:
            raise ChunkIdentityError(f"conflicting duplicate chunk_id {chunk_id!r}")
    return prepared


def manifest_entry(metadata: Mapping[str, Any]) -> ChunkManifestEntry:
    storage_identity = metadata.get("storage_identity") or {}
    return ChunkManifestEntry(
        chunk_id=str(metadata.get("chunk_id") or storage_identity.get("chunk_id") or ""),
        document_id=str(
            metadata.get("document_id") or storage_identity.get("document_id") or ""
        ),
        version=_positive_version(
            metadata.get("version", storage_identity.get("version", 0))
        ),
        identity_hash=str(storage_identity.get("metadata_hash") or ""),
        state=str(metadata.get("lifecycle_state", metadata.get("status", ""))),
        operation_id=(
            str(metadata["lifecycle_operation_id"])
            if metadata.get("lifecycle_operation_id") is not None
            else None
        ),
    )


__all__ = [
    "ACTIVE",
    "PENDING",
    "STALE",
    "DELETING",
    "FAILED",
    "LIFECYCLE_STATES",
    "ChunkIdentityError",
    "ChunkManifestEntry",
    "canonical_json",
    "manifest_entry",
    "prepare_chunk",
    "prepare_chunks",
]
