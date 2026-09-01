"""Structure-preserving, idempotent text normalization."""

from __future__ import annotations

import unicodedata
from typing import Any, Mapping

from ..config import NormalizationConfig

try:
    from ftfy import fix_text as _repair_text
except ImportError:  # optional quality improvement
    def _repair_text(value: str) -> str:
        return value


def _default_config():
    return NormalizationConfig()


def clean_text(text: str, config=None) -> str:
    """Normalize encoding artifacts without joining or flattening structures.

    In particular, Markdown tables, lists, code indentation, page separators,
    and paragraph boundaries retain their line layout. The transformation is
    idempotent for already-normalized input.
    """

    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text:
        return ""
    config = config or _default_config()

    value = _repair_text(text)
    if config.normalize_line_endings:
        value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize(config.unicode_form, value)
    value = value.replace("\ufeff", "").replace("\u00a0", " ")

    if config.strip_control_characters:
        value = "".join(
            character
            for character in value
            if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
        )

    lines = value.split("\n")
    if config.trim_trailing_whitespace:
        lines = [line.rstrip(" \t") for line in lines]

    maximum_blank = int(config.collapse_blank_lines)
    output: list[str] = []
    blank_count = 0
    for line in lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= maximum_blank:
                output.append("")
        else:
            blank_count = 0
            output.append(line)
    return "\n".join(output).strip("\n")


def normalize_document(document: Mapping[str, Any], config=None) -> dict[str, Any]:
    """Return a normalized copy while preserving provenance byte-for-byte."""

    result = dict(document)
    result["metadata"] = dict(document.get("metadata") or {})
    result["text"] = clean_text(str(document.get("text") or ""), config=config)
    return result
