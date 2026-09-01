from __future__ import annotations

import json

from .base import (
    DocumentParser,
    ParserError,
    ParserErrorCode,
    parsed_documents,
    read_text_file,
    source_path,
)


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class JSONParser(DocumentParser):
    """Parse JSON/JSONL into deterministic top-level structural units."""

    def __init__(self, encoding_fallbacks=None):
        self.encoding_fallbacks = encoding_fallbacks

    def parse(self, source):
        path = source_path(source)
        text, encoding = read_text_file(path, self.encoding_fallbacks)
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ldjson"}:
            segments = self._parse_jsonl(text, path, encoding)
        else:
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                nonempty = [line for line in text.splitlines() if line.strip()]
                if nonempty and all(self._valid_json(line) for line in nonempty):
                    segments = self._parse_jsonl(text, path, encoding)
                else:
                    raise ParserError(
                        ParserErrorCode.CORRUPT_CONTENT,
                        f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
                        source=path,
                        stage="decode",
                        diagnostics={"line": exc.lineno, "column": exc.colno},
                    ) from exc
            else:
                segments = self._segments(value, encoding)
        if not segments:
            raise ParserError(ParserErrorCode.EMPTY_CONTENT, "JSON contains no records", source=path)
        return parsed_documents(source, path, segments)

    @staticmethod
    def _valid_json(value: str) -> bool:
        try:
            json.loads(value)
            return True
        except json.JSONDecodeError:
            return False

    def _segments(self, value, encoding: str) -> list[dict]:
        common = {"method": "json", "encoding": encoding, "block_type": "json_record"}
        if isinstance(value, dict):
            return [
                {
                    **common,
                    "text": _dump({key: child}),
                    "section": str(key),
                    "json_path": f"$.{key}",
                    "json_type": type(child).__name__,
                    "record_index": index,
                }
                for index, (key, child) in enumerate(value.items())
            ]
        if isinstance(value, list):
            return [
                {
                    **common,
                    "text": _dump(child),
                    "section": f"record {index}",
                    "json_path": f"$[{index}]",
                    "json_type": type(child).__name__,
                    "record_index": index,
                }
                for index, child in enumerate(value)
            ]
        if value is None:
            return []
        return [{
            **common,
            "text": _dump(value),
            "section": "value",
            "json_path": "$",
            "json_type": type(value).__name__,
            "record_index": 0,
        }]

    def _parse_jsonl(self, text: str, path, encoding: str) -> list[dict]:
        segments = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParserError(
                    ParserErrorCode.CORRUPT_CONTENT,
                    f"invalid JSONL record at line {line_number}, column {exc.colno}: {exc.msg}",
                    source=path,
                    stage="decode",
                    diagnostics={"line": line_number, "column": exc.colno},
                ) from exc
            if value is None:
                continue
            segments.append(
                {
                    "text": _dump(value),
                    "method": "jsonl",
                    "encoding": encoding,
                    "block_type": "json_record",
                    "section": f"line {line_number}",
                    "json_path": f"$line[{line_number}]",
                    "json_type": type(value).__name__,
                    "line_number": line_number,
                    "record_index": len(segments),
                }
            )
        return segments
