from __future__ import annotations

import csv
from io import StringIO

from .base import (
    DocumentParser,
    ParserError,
    ParserErrorCode,
    parsed_documents,
    read_text_file,
    source_path,
)


def _unique_headers(row: list[str]) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(row, 1):
        base = value.strip() or f"column_{index}"
        counts[base] = counts.get(base, 0) + 1
        headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return headers


class CSVParser(DocumentParser):
    """Parse one logical record per segment with dialect/row provenance."""

    def __init__(self, encoding_fallbacks=None):
        self.encoding_fallbacks = encoding_fallbacks

    def parse(self, source):
        path = source_path(source)
        text, encoding = read_text_file(path, self.encoding_fallbacks)
        sample = text[:65536]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel

        try:
            rows = list(csv.reader(StringIO(text), dialect=dialect, strict=True))
        except csv.Error as exc:
            raise ParserError(
                ParserErrorCode.CORRUPT_CONTENT,
                f"malformed CSV: {exc}",
                source=path,
                stage="decode",
            ) from exc
        if not rows or not any(any(str(cell).strip() for cell in row) for row in rows):
            raise ParserError(ParserErrorCode.EMPTY_CONTENT, "CSV file is empty", source=path)

        headers = _unique_headers([str(cell) for cell in rows[0]])
        common = {
            "method": "csv",
            "encoding": encoding,
            "delimiter": dialect.delimiter,
            "column_names": headers,
            "table_id": path.stem,
        }
        segments = []
        for row_number, row in enumerate(rows[1:], 2):
            values = [str(cell).strip() for cell in row]
            if not any(values):
                continue
            width = max(len(headers), len(values))
            if width > len(headers):
                headers.extend(f"column_{index}" for index in range(len(headers) + 1, width + 1))
            values.extend([""] * (len(headers) - len(values)))
            pairs = [f"{header}: {values[index]}" for index, header in enumerate(headers)]
            segments.append(
                {
                    **common,
                    "text": "\n".join(pairs),
                    "section": f"row {row_number}",
                    "row_number": row_number,
                    "row_start": row_number,
                    "row_end": row_number,
                    "block_type": "table_row",
                    "column_names": list(headers),
                }
            )

        if not segments:
            segments.append(
                {
                    **common,
                    "text": " | ".join(headers),
                    "section": "header",
                    "row_number": 1,
                    "row_start": 1,
                    "row_end": 1,
                    "block_type": "table_header",
                }
            )
        return parsed_documents(source, path, segments)
