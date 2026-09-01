from __future__ import annotations

from dataclasses import replace
from pathlib import PurePosixPath
from urllib.parse import unquote
from xml.etree import ElementTree
import posixpath
import zipfile

from ...config import ParserConfig
from .base import (
    DocumentParser, ParserError, ParserErrorCode, source_path, validate_zip_archive,
)
from .html import HTMLParser


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_archive_name(value: str) -> str:
    normalized = posixpath.normpath(unquote(value).replace("\\", "/"))
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ParserError(
            ParserErrorCode.CORRUPT_CONTENT,
            f"unsafe EPUB member path: {value}",
            stage="container",
        )
    return normalized.lstrip("./")


class EPUBParser(DocumentParser):
    """Parse XHTML content in EPUB spine (declared reading) order."""

    def __init__(self, encoding_fallbacks=None, config: ParserConfig | None = None):
        self.encoding_fallbacks = tuple(encoding_fallbacks or ("utf-8-sig", "utf-8", "cp1258", "cp1252"))
        self.config = config or ParserConfig()

    def parse(self, source):
        path = source_path(source)
        validate_zip_archive(path, self.config)
        try:
            archive = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ParserError(
                ParserErrorCode.CORRUPT_CONTENT,
                f"invalid EPUB container: {type(exc).__name__}: {exc}",
                source=path,
                stage="container",
            ) from exc

        try:
            encrypted = [item.filename for item in archive.infolist() if item.flag_bits & 0x1]
            if encrypted:
                raise ParserError(
                    ParserErrorCode.ENCRYPTED_CONTENT,
                    "encrypted EPUB members are not supported",
                    source=path,
                    stage="container",
                    diagnostics={"members": encrypted[:10]},
                )
            spine = self._spine_items(archive)
            if not spine:
                raise ParserError(ParserErrorCode.EMPTY_CONTENT, "EPUB has no XHTML content", source=path)

            html_parser = HTMLParser()
            documents = []
            missing: list[str] = []
            for spine_index, member_name in enumerate(spine, 1):
                try:
                    payload = archive.read(member_name)
                except KeyError:
                    missing.append(member_name)
                    continue
                if not payload:
                    continue
                html = None
                encoding = None
                for candidate in self.encoding_fallbacks:
                    try:
                        html = payload.decode(candidate, errors="strict")
                        encoding = candidate
                        break
                    except (LookupError, UnicodeDecodeError):
                        continue
                if html is None:
                    raise ParserError(
                        ParserErrorCode.CORRUPT_CONTENT,
                        f"cannot decode EPUB member {member_name}",
                        source=path,
                        stage="decode",
                    )
                parsed = html_parser.parse_html(source, path, html, encoding=encoding)
                for document in parsed:
                    metadata = {
                        **dict(document.metadata),
                        "parser_method": "epub-html",
                        "epub_item": member_name,
                        "spine_index": spine_index,
                        "missing_spine_items": list(missing),
                    }
                    documents.append(replace(document, metadata=metadata))
            if not documents:
                raise ParserError(
                    ParserErrorCode.EMPTY_CONTENT,
                    "EPUB spine produced no readable content",
                    source=path,
                    diagnostics={"missing_spine_items": missing},
                )
            return documents
        finally:
            archive.close()

    def _spine_items(self, archive: zipfile.ZipFile) -> list[str]:
        names = {item.filename.replace("\\", "/") for item in archive.infolist()}
        try:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            rootfile = next(
                element for element in container.iter()
                if _local_name(element.tag) == "rootfile" and element.get("full-path")
            )
            package_name = _safe_archive_name(rootfile.get("full-path", ""))
            package = ElementTree.fromstring(archive.read(package_name))
            manifest = {
                element.get("id"): (element.get("href", ""), element.get("media-type", ""))
                for element in package.iter()
                if _local_name(element.tag) == "item" and element.get("id")
            }
            base = posixpath.dirname(package_name)
            result = []
            for element in package.iter():
                if _local_name(element.tag) != "itemref":
                    continue
                href, media_type = manifest.get(element.get("idref"), ("", ""))
                if media_type not in {"application/xhtml+xml", "text/html", "text/xml"}:
                    continue
                member = _safe_archive_name(posixpath.join(base, href))
                if member in names:
                    result.append(member)
            if result:
                return result
        except (KeyError, StopIteration, ElementTree.ParseError):
            pass
        return sorted(
            _safe_archive_name(name)
            for name in names
            if name.lower().endswith((".xhtml", ".html", ".htm"))
            and not name.lower().startswith("meta-inf/")
        )
