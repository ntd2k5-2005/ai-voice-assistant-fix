from __future__ import annotations

import re

from .base import DocumentParser, parsed_documents, read_text_file, source_path


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")


def _table_headers(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


class MarkdownParser(DocumentParser):
    """Split Markdown at structural boundaries without flattening syntax."""

    def __init__(self, encoding_fallbacks=None):
        self.encoding_fallbacks = encoding_fallbacks

    def parse(self, source):
        path = source_path(source)
        text, encoding = read_text_file(path, self.encoding_fallbacks)
        method = "markdown" if path.suffix.lower() != ".txt" else "text-file"
        if method == "text-file":
            return parsed_documents(
                source,
                path,
                [{
                    "text": text,
                    "method": method,
                    "section": "text",
                    "block_type": "text",
                    "encoding": encoding,
                    "line_start": 1,
                    "line_end": max(1, len(text.splitlines())),
                }],
            )

        lines = text.splitlines()
        segments: list[dict] = []
        heading_stack: dict[int, str] = {}
        buffer: list[str] = []
        buffer_start = 1
        buffer_type = "prose"
        table_index = 0

        def heading_path() -> list[str]:
            return [heading_stack[level] for level in sorted(heading_stack)]

        def flush(end_line: int) -> None:
            nonlocal buffer, buffer_start, buffer_type
            value = "\n".join(buffer).strip("\n")
            if value.strip():
                path_values = heading_path()
                segments.append(
                    {
                        "text": value,
                        "method": method,
                        "heading": path_values[-1] if path_values else None,
                        "heading_path": path_values,
                        "heading_level": max(heading_stack) if heading_stack else None,
                        "section": " > ".join(path_values) if path_values else "preamble",
                        "block_type": buffer_type,
                        "line_start": buffer_start,
                        "line_end": max(buffer_start, end_line),
                        "encoding": encoding,
                        "links": _LINK.findall(value),
                    }
                )
            buffer = []
            buffer_type = "prose"

        index = 0
        while index < len(lines):
            line = lines[index]
            line_number = index + 1

            fence = _FENCE.match(line)
            if fence:
                flush(line_number - 1)
                marker = fence.group(1)
                fence_char, fence_len = marker[0], len(marker)
                code_lines = [line]
                start = line_number
                index += 1
                while index < len(lines):
                    code_lines.append(lines[index])
                    if re.match(rf"^\s{{0,3}}{re.escape(fence_char)}{{{fence_len},}}\s*$", lines[index]):
                        break
                    index += 1
                buffer = code_lines
                buffer_start = start
                buffer_type = "code"
                flush(index + 1)
                index += 1
                continue

            heading = _HEADING.match(line)
            if heading:
                flush(line_number - 1)
                level = len(heading.group(1))
                for stale_level in [value for value in heading_stack if value >= level]:
                    del heading_stack[stale_level]
                heading_stack[level] = heading.group(2).strip()
                buffer = [line]
                buffer_start = line_number
                buffer_type = "section"
                index += 1
                continue

            is_table = (
                index + 1 < len(lines)
                and "|" in line
                and _TABLE_SEPARATOR.match(lines[index + 1]) is not None
            )
            if is_table:
                flush(line_number - 1)
                start = line_number
                table_lines = [line, lines[index + 1]]
                index += 2
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    table_lines.append(lines[index])
                    index += 1
                table_index += 1
                current_path = heading_path()
                segments.append(
                    {
                        "text": "\n".join(table_lines),
                        "method": method,
                        "heading": current_path[-1] if current_path else None,
                        "heading_path": current_path,
                        "heading_level": max(heading_stack) if heading_stack else None,
                        "section": " > ".join(current_path) if current_path else "preamble",
                        "block_type": "table",
                        "table_id": f"markdown-table-{table_index}",
                        "table_index": table_index,
                        "column_names": _table_headers(line),
                        "row_start": start,
                        "row_end": start + len(table_lines) - 1,
                        "line_start": start,
                        "line_end": start + len(table_lines) - 1,
                        "encoding": encoding,
                    }
                )
                buffer_start = index + 1
                continue

            if _LIST_ITEM.match(line) and buffer and buffer_type != "list":
                flush(line_number - 1)
                buffer_start = line_number
                buffer_type = "list"
            elif buffer_type == "list" and line.strip() and not (_LIST_ITEM.match(line) or line.startswith((" ", "\t"))):
                flush(line_number - 1)
                buffer_start = line_number

            if not buffer:
                buffer_start = line_number
                buffer_type = "list" if _LIST_ITEM.match(line) else buffer_type
            buffer.append(line)
            index += 1

        flush(len(lines))
        return parsed_documents(source, path, segments)
