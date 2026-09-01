"""Shared parser contracts, diagnostics, source detection, and provenance helpers.

The production parsers intentionally expose one small contract: a source is
converted into ordered ``ParsedDocument`` units. Format-specific libraries are
imported lazily by their parser so importing the registry stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import json
import re
import stat
from typing import Any, Iterable, Mapping
from urllib.parse import unquote
import zipfile

from ...contracts import DocumentSource, ParsedDocument
from ...errors import (
    EmptyDocumentError,
    EncryptedDocumentError,
    OCRError,
    ParserError as RagParserError,
    SourceError,
    SourceTooLargeError,
    UnsupportedSourceError,
)


class ParserErrorCode(str, Enum):
    """Machine-readable parser failure categories."""

    NOT_FOUND = "not_found"
    INVALID_SOURCE = "invalid_source"
    UNSUPPORTED_FORMAT = "unsupported_format"
    TYPE_MISMATCH = "type_mismatch"
    EMPTY_CONTENT = "empty_content"
    CORRUPT_CONTENT = "corrupt_content"
    ENCRYPTED_CONTENT = "encrypted_content"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    EXTRACTION_FAILED = "extraction_failed"


class ParserError(RagParserError, ValueError):
    """Typed, inspectable parser failure.

    ``ValueError`` remains a base class for callers that previously handled
    invalid ingestion input generically. ``code`` and ``diagnostics`` let
    production callers distinguish remediation paths.
    """

    def __init__(
        self,
        code: ParserErrorCode,
        message: str,
        *,
        source: str | Path | None = None,
        stage: str = "parse",
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.source = None if source is None else str(source)
        self.stage = stage
        self.diagnostics = dict(diagnostics or {})
        suffix = f" [{self.source}]" if self.source else ""
        super().__init__(f"{code.value}: {message}{suffix}")


@dataclass(frozen=True)
class FileTypeDetection:
    declared_type: str
    detected_type: str
    confidence: str

    @property
    def mismatch(self) -> bool:
        return bool(
            self.declared_type
            and self.detected_type
            and self.declared_type != self.detected_type
        )


_TYPE_ALIASES = {
    "markdown": "md",
    "mdx": "md",
    "htm": "html",
    "xlsm": "xlsx",
    "ldjson": "jsonl",
}
_TEXT_TYPES = {
    "txt", "md", "html", "csv", "json", "jsonl", "py", "js", "ts",
    "java", "c", "cpp", "h", "php", "go", "sh", "cs", "kt", "sql",
}


class DocumentParser:
    """Parser contract implemented by every ingestion adapter."""

    def parse(self, source: DocumentSource) -> list[ParsedDocument]:
        raise NotImplementedError


def canonical_file_type(value: str) -> str:
    value = value.lower().lstrip(".")
    return _TYPE_ALIASES.get(value, value)


def source_path(source: DocumentSource) -> Path:
    path = Path(str(source.value)).expanduser()
    if not path.exists():
        raise ParserError(ParserErrorCode.NOT_FOUND, "source does not exist", source=path)
    if not path.is_file():
        raise ParserError(ParserErrorCode.INVALID_SOURCE, "source is not a file", source=path)
    return path


def _zip_document_type(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = {name.replace("\\", "/").lower() for name in archive.namelist()}
            if "meta-inf/container.xml" in names:
                # Do not decompress any member during type sniffing. Full ZIP
                # safety preflight runs before the selected parser opens data.
                if "mimetype" in names or any(name.endswith(".opf") for name in names):
                    return "epub"
            if "word/document.xml" in names:
                return "docx"
            if "xl/workbook.xml" in names:
                return "xlsx"
            if "ppt/presentation.xml" in names:
                return "pptx"
    except (OSError, zipfile.BadZipFile):
        return None
    return None


def validate_zip_archive(path: Path, config: Any) -> dict[str, Any]:
    """Reject unsafe or unreasonable OOXML/EPUB containers before extraction."""

    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ParserError(
            ParserErrorCode.CORRUPT_CONTENT,
            f"invalid ZIP container: {type(exc).__name__}: {exc}",
            source=path,
            stage="container_preflight",
        ) from exc
    try:
        members = archive.infolist()
        max_members = int(config.max_archive_members)
        if len(members) > max_members:
            raise ParserError(
                ParserErrorCode.INVALID_SOURCE,
                f"archive has {len(members)} members, exceeding limit {max_members}",
                source=path,
                stage="container_preflight",
                diagnostics={"members": len(members), "max_archive_members": max_members},
            )
        total = 0
        normalized_names: set[str] = set()
        max_member = int(config.max_archive_member_bytes)
        max_total = int(config.max_archive_uncompressed_bytes)
        max_ratio = float(config.max_archive_compression_ratio)
        for member in members:
            raw_name = str(member.filename)
            decoded = unquote(raw_name).replace("\\", "/")
            if "\x00" in decoded:
                reason = "NUL byte"
            else:
                parts = tuple(
                    part for part in decoded.split("/") if part not in {"", "."}
                )
                drive_like = bool(parts and len(parts[0]) >= 2 and parts[0][1] == ":")
                unsafe = (
                    decoded.startswith("/")
                    or drive_like
                    or any(part == ".." for part in parts)
                )
                reason = "path traversal or absolute path" if unsafe else ""
            if reason:
                raise ParserError(
                    ParserErrorCode.CORRUPT_CONTENT,
                    f"unsafe archive member {raw_name!r}: {reason}",
                    source=path,
                    stage="container_preflight",
                )
            canonical_name = "/".join(parts).casefold()
            if canonical_name in normalized_names and not member.is_dir():
                raise ParserError(
                    ParserErrorCode.CORRUPT_CONTENT,
                    f"duplicate archive member after normalization: {raw_name!r}",
                    source=path,
                    stage="container_preflight",
                )
            normalized_names.add(canonical_name)
            unix_mode = (int(member.external_attr) >> 16) & 0o170000
            if unix_mode == stat.S_IFLNK:
                raise ParserError(
                    ParserErrorCode.CORRUPT_CONTENT,
                    f"symbolic-link archive member is not allowed: {raw_name!r}",
                    source=path,
                    stage="container_preflight",
                )
            if member.flag_bits & 0x1:
                raise ParserError(
                    ParserErrorCode.ENCRYPTED_CONTENT,
                    f"encrypted archive member is not supported: {raw_name!r}",
                    source=path,
                    stage="container_preflight",
                )
            size = int(member.file_size)
            if size > max_member:
                raise ParserError(
                    ParserErrorCode.INVALID_SOURCE,
                    f"archive member {raw_name!r} expands to {size} bytes, exceeding {max_member}",
                    source=path,
                    stage="container_preflight",
                )
            total += size
            if total > max_total:
                raise ParserError(
                    ParserErrorCode.INVALID_SOURCE,
                    f"archive expands to more than configured {max_total} bytes",
                    source=path,
                    stage="container_preflight",
                )
            ratio = size / max(1, int(member.compress_size))
            if size and ratio > max_ratio:
                raise ParserError(
                    ParserErrorCode.INVALID_SOURCE,
                    f"archive member {raw_name!r} compression ratio {ratio:.1f} exceeds {max_ratio:.1f}",
                    source=path,
                    stage="container_preflight",
                )
        return {
            "member_count": len(members),
            "uncompressed_bytes": total,
            "max_compression_ratio": max(
                (int(item.file_size) / max(1, int(item.compress_size)) for item in members),
                default=0.0,
            ),
        }
    finally:
        archive.close()


def detect_file_type(path: Path) -> FileTypeDetection:
    """Detect strong binary signatures and retain declared textual types."""

    declared = canonical_file_type(path.suffix)
    try:
        with path.open("rb") as handle:
            head = handle.read(8192)
    except OSError as exc:
        raise ParserError(
            ParserErrorCode.INVALID_SOURCE,
            f"cannot read source: {type(exc).__name__}: {exc}",
            source=path,
            stage="intake",
        ) from exc

    detected = ""
    confidence = "extension"
    if head.startswith(b"%PDF-"):
        detected, confidence = "pdf", "magic"
    elif head.startswith(b"PK\x03\x04"):
        detected = _zip_document_type(path) or "zip"
        confidence = "container"
    elif head.startswith(b"\xd0\xcf\x11\xe0"):
        detected = declared if declared in {"doc", "xls", "ppt"} else "ole"
        confidence = "magic"
    elif declared in _TEXT_TYPES:
        detected = declared
    elif declared:
        detected = declared
    else:
        sample = head.decode("utf-8-sig", errors="ignore").lstrip()
        if re.match(r"(?is)<!doctype\s+html\b|<html\b", sample):
            detected, confidence = "html", "content"
        else:
            try:
                json.loads(sample)
                detected, confidence = "json", "content"
            except (json.JSONDecodeError, TypeError):
                detected = "txt" if sample else ""
                confidence = "content"

    return FileTypeDetection(declared, canonical_file_type(detected), confidence)


def read_text_file(path: Path, encoding_fallbacks: Iterable[str] | None = None) -> tuple[str, str]:
    """Decode a textual source without silently replacing corrupt bytes."""

    raw = path.read_bytes()
    if not raw:
        raise ParserError(ParserErrorCode.EMPTY_CONTENT, "file is empty", source=path)

    candidates: list[str] = []
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        candidates.extend(["utf-32", "utf-8-sig"])
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates.extend(["utf-16", "utf-8-sig"])
    else:
        candidates.append("utf-8-sig")

    configured = tuple(encoding_fallbacks or ("utf-8-sig", "utf-8", "cp1258", "cp1252"))
    candidates.extend(configured)

    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None and best.encoding:
            candidates.append(best.encoding)
    except ImportError:  # pragma: no cover - dependency is optional
        pass

    seen: set[str] = set()
    for encoding in candidates:
        normalized = encoding.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            text = raw.decode(encoding, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
        if _usable_decoded_text(text):
            return text, encoding

    raise ParserError(
        ParserErrorCode.CORRUPT_CONTENT,
        "unable to decode text without replacement characters",
        source=path,
        stage="decode",
    )


def _usable_decoded_text(text: str) -> bool:
    if not text or "\ufffd" in text:
        return False
    nonspace = [char for char in text if not char.isspace()]
    if not nonspace:
        return True
    disallowed = sum(
        1 for char in nonspace
        if ord(char) < 32 and char not in {"\t", "\n", "\r"}
    )
    return disallowed / len(nonspace) < 0.02


def source_metadata(
    source: DocumentSource,
    path: Path,
    method: str,
    detection: FileTypeDetection | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Stable provenance shared by all parser outputs."""

    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    detection = detection or detect_file_type(path)
    return {
        **dict(source.metadata),
        "source_path": str(path.resolve()),
        "source_name": path.name,
        "file_type": detection.detected_type or detection.declared_type,
        "declared_file_type": detection.declared_type,
        "detected_file_type": detection.detected_type,
        "extension_mismatch": detection.mismatch,
        "parser_method": method,
        "source_size_bytes": stat.st_size,
        "source_modified_at": modified,
        **extra,
    }


def parsed_documents(
    source: DocumentSource,
    path: Path,
    documents: Iterable[Mapping[str, Any]],
) -> list[ParsedDocument]:
    """Convert ordered parser units into provenance-preserving records."""

    document_id = str(source.metadata.get("document_id") or path.resolve())
    detection = detect_file_type(path)
    result: list[ParsedDocument] = []
    for segment_index, item in enumerate(documents):
        content = str(item.get("text", "")).strip(" \r\n\ufeff")
        if not content:
            continue
        method = str(item.get("method") or "unknown")
        extra = {
            key: value for key, value in item.items()
            if key not in {"text", "title", "method"}
        }
        extra.setdefault("segment_index", segment_index)
        metadata = source_metadata(source, path, method, detection=detection, **extra)
        result.append(
            ParsedDocument(
                document_id=document_id,
                source=str(path),
                source_type=source.source_type,
                content=content,
                file_name=path.name,
                file_type=(detection.detected_type or canonical_file_type(path.suffix)),
                title=item.get("title"),
                metadata=metadata,
            )
        )
    if not result:
        raise ParserError(
            ParserErrorCode.EMPTY_CONTENT,
            "parser produced no usable content",
            source=path,
            stage="extract",
        )
    return result


def one_file_document(
    source: DocumentSource,
    path: Path,
    text: str,
    method: str,
    **metadata: Any,
) -> list[ParsedDocument]:
    return parsed_documents(
        source,
        path,
        [{"text": text, "page": 1, "method": method, **metadata}],
    )
