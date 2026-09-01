"""Shared, dependency-free sparse text analysis.

The analyzer is deliberately explicit: both in-memory BM25 and persistent
SQLite FTS receive the same normalized query terms. Original chunk text is
never mutated and remains the value returned to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Protocol, runtime_checkable


_TOKEN = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class SparseAnalyzer(Protocol):
    name: str
    sqlite_tokenizer: str

    def tokens(self, text: str) -> tuple[str, ...]: ...
    def index_text(self, text: str) -> str: ...


@dataclass(frozen=True)
class Unicode61Analyzer:
    """Deterministic counterpart of SQLite FTS5's ``unicode61`` analyzer."""

    name: str = "unicode61"
    sqlite_tokenizer: str = "unicode61"

    @staticmethod
    def _fold(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("sparse analyzer input must be a string")
        decomposed = unicodedata.normalize("NFKD", text.casefold())
        return "".join(
            character
            for character in decomposed
            if unicodedata.category(character) != "Mn"
        )

    def tokens(self, text: str) -> tuple[str, ...]:
        return tuple(_TOKEN.findall(self._fold(text)))

    def index_text(self, text: str) -> str:
        """Return index-only text while keeping caller-owned text untouched."""

        return " ".join(self.tokens(text))


def resolve_sparse_analyzer(value: str | SparseAnalyzer = "unicode61") -> SparseAnalyzer:
    if isinstance(value, str):
        name = value.strip().casefold()
        if name == "unicode61":
            return Unicode61Analyzer()
        raise ValueError(f"unsupported sparse analyzer: {value!r}")
    if not isinstance(value, SparseAnalyzer):
        raise TypeError("analyzer must be a registered name or SparseAnalyzer")
    return value


__all__ = [
    "SparseAnalyzer",
    "Unicode61Analyzer",
    "resolve_sparse_analyzer",
]
