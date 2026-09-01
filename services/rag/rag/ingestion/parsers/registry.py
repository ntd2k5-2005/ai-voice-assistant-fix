from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile

from .base import (
    DocumentParser,
    FileTypeDetection,
    ParserError,
    ParserErrorCode,
    canonical_file_type,
    detect_file_type,
    source_path,
)
from .csv import CSVParser
from .docx import DOCXParser
from .epub import EPUBParser
from .html import HTMLParser
from .json_data import JSONParser
from .markdown import MarkdownParser
from .pdf import PDFParser
from .pptx import PPTXParser
from .text import PlainTextFileParser, TextParser
from .xlsx import XLSXParser

from ...config import ParserConfig
from ...contracts import DocumentSource


_CODE_EXTENSIONS = {"py", "js", "ts", "java", "c", "cpp", "h", "php", "go", "sh", "cs", "kt", "sql"}


class FileParser(DocumentParser):
    """Dispatch a file using content signatures plus the declared extension."""

    def __init__(self, config: ParserConfig | None = None):
        self.config = config or ParserConfig()
        encodings = self.config.encoding_fallbacks
        plain_text = PlainTextFileParser(encodings)
        self._parsers: dict[str, DocumentParser] = {
            "txt": plain_text,
            "md": MarkdownParser(encodings),
            "pdf": PDFParser.from_config(self.config),
            "docx": DOCXParser(self.config),
            "csv": CSVParser(encodings),
            "tsv": CSVParser(encodings),
            "xlsx": XLSXParser(self.config),
            "pptx": PPTXParser(self.config),
            "html": HTMLParser(encodings),
            "json": JSONParser(encodings),
            "jsonl": JSONParser(encodings),
            "epub": EPUBParser(encodings, self.config),
            **{extension: plain_text for extension in _CODE_EXTENSIONS},
        }

    @property
    def supported_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._parsers))

    def register(self, file_type: str, parser: DocumentParser) -> None:
        if not isinstance(parser, DocumentParser):
            raise TypeError("parser must implement DocumentParser")
        self._parsers[canonical_file_type(file_type)] = parser

    def parse(self, source):
        path = source_path(source)
        size = path.stat().st_size
        if size > self.config.max_file_bytes:
            raise ParserError(
                ParserErrorCode.INVALID_SOURCE,
                f"source size {size} exceeds configured limit {self.config.max_file_bytes}",
                source=path,
                stage="intake",
                diagnostics={"size_bytes": size, "max_file_bytes": self.config.max_file_bytes},
            )

        declared_type = canonical_file_type(path.suffix)
        detection = (
            detect_file_type(path)
            if self.config.sniff_mime
            else FileTypeDetection(declared_type, declared_type, "extension")
        )
        declared_extension = path.suffix.casefold()
        if declared_extension and declared_extension not in self.config.allowed_extensions:
            declared_type = canonical_file_type(declared_extension)
            strong_supported_signature = (
                not self.config.reject_extension_mismatch
                and detection.confidence in {"magic", "container"}
                and detection.detected_type in self._parsers
            )
            if declared_type not in _CODE_EXTENSIONS and not strong_supported_signature:
                raise ParserError(
                    ParserErrorCode.UNSUPPORTED_FORMAT,
                    f"extension is not allowed: {declared_extension}",
                    source=path,
                    stage="intake",
                )
        if self.config.reject_extension_mismatch and detection.mismatch:
            raise ParserError(
                ParserErrorCode.TYPE_MISMATCH,
                f"declared {detection.declared_type!r}, detected {detection.detected_type!r}",
                source=path,
                stage="intake",
                diagnostics={
                    "declared_type": detection.declared_type,
                    "detected_type": detection.detected_type,
                    "confidence": detection.confidence,
                },
            )

        # A strong PDF/OOXML/EPUB signature wins when permissive mismatch
        # handling is configured. Text grammars retain their declared type.
        selected = detection.declared_type
        if detection.confidence in {"magic", "container"} and detection.detected_type in self._parsers:
            selected = detection.detected_type
        if selected not in self._parsers and detection.detected_type in self._parsers:
            selected = detection.detected_type
        parser = self._parsers.get(selected)
        if parser is None:
            raise ParserError(
                ParserErrorCode.UNSUPPORTED_FORMAT,
                f"unsupported file type: {selected or detection.detected_type or '<none>'}",
                source=path,
                stage="intake",
                diagnostics={"supported_types": self.supported_types},
            )

        try:
            result = parser.parse(source)
        except ParserError:
            raise
        except ImportError as exc:
            raise ParserError(
                ParserErrorCode.DEPENDENCY_UNAVAILABLE,
                f"parser dependency unavailable: {exc}",
                source=path,
                stage="load_dependency",
            ) from exc
        except Exception as exc:
            message = str(exc)
            lowered = message.casefold()
            code = (
                ParserErrorCode.ENCRYPTED_CONTENT
                if any(token in lowered for token in ("password", "encrypted", "decrypt"))
                else ParserErrorCode.CORRUPT_CONTENT
            )
            raise ParserError(
                code,
                f"{type(exc).__name__}: {message}",
                source=path,
                stage="extract",
            ) from exc

        if len(result) > self.config.max_units:
            raise ParserError(
                ParserErrorCode.INVALID_SOURCE,
                f"parser produced {len(result)} units, exceeding configured limit {self.config.max_units}",
                source=path,
                stage="extract",
                diagnostics={"units": len(result), "max_units": self.config.max_units},
            )
        return result


class BytesParser(DocumentParser):
    """Materialize a bounded payload in an isolated, automatically removed directory."""

    def __init__(self, file_parser: FileParser, config: ParserConfig):
        self.file_parser = file_parser
        self.config = config

    def parse(self, source):
        payload = source.value
        if not isinstance(payload, bytes):
            raise ParserError(ParserErrorCode.INVALID_SOURCE, "bytes source requires a bytes payload", stage="intake")
        if len(payload) > self.config.max_file_bytes:
            raise ParserError(
                ParserErrorCode.INVALID_SOURCE,
                f"source size {len(payload)} exceeds configured limit {self.config.max_file_bytes}",
                source=source.file_name,
                stage="intake",
            )
        file_name = source.file_name
        if not file_name:
            raise ParserError(ParserErrorCode.INVALID_SOURCE, "bytes source requires file_name", stage="intake")
        if Path(file_name).name != file_name or "/" in file_name or "\\" in file_name:
            raise ParserError(
                ParserErrorCode.INVALID_SOURCE,
                "bytes source file_name must be a basename without path components",
                source=file_name,
                stage="intake",
            )
        digest = hashlib.sha256(payload).hexdigest()
        metadata = dict(source.metadata)
        metadata.setdefault("document_id", f"bytes:{digest}")
        with tempfile.TemporaryDirectory(prefix="rag-parser-") as directory:
            path = Path(directory) / file_name
            path.write_bytes(payload)
            temporary_source = DocumentSource(
                path,
                source_type="file",
                metadata=metadata,
                mime_type=source.mime_type,
                file_name=file_name,
            )
            try:
                parsed = self.file_parser.parse(temporary_source)
            except ParserError as exc:
                raise ParserError(
                    exc.code,
                    exc.message,
                    source=file_name,
                    stage=exc.stage,
                    diagnostics=exc.diagnostics,
                ) from exc

        result = []
        for document in parsed:
            provenance = dict(document.metadata)
            provenance.update(
                {
                    "source_path": file_name,
                    "source_name": file_name,
                    "source_size_bytes": len(payload),
                    "source_in_memory": True,
                    "content_sha256": digest,
                }
            )
            provenance.pop("source_modified_at", None)
            result.append(
                replace(
                    document,
                    source=file_name,
                    source_type="bytes",
                    file_name=file_name,
                    metadata=provenance,
                )
            )
        return result


class ParserRegistry:
    def __init__(self, config: ParserConfig | None = None):
        self.config = config or ParserConfig()
        self._text = TextParser()
        self._file = FileParser(self.config)
        self._bytes = BytesParser(self._file, self.config)

    @property
    def supported_types(self) -> tuple[str, ...]:
        return self._file.supported_types

    def for_source(self, source):
        if source.source_type == "text":
            return self._text
        if source.source_type == "bytes":
            return self._bytes
        if source.source_type != "file":
            raise ParserError(
                ParserErrorCode.INVALID_SOURCE,
                f"unsupported source_type: {source.source_type!r}",
                source=str(source.value),
                stage="intake",
            )
        return self._file

    def parse(self, source):
        return self.for_source(source).parse(source)


default_registry = ParserRegistry()
