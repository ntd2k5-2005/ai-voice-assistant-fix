from __future__ import annotations

import re
from statistics import median

from ...config import ParserConfig
from .base import DocumentParser, parsed_documents, source_path, validate_zip_archive


_HEADING_STYLE = re.compile(r"(?:heading|title)\s*(\d+)?", re.IGNORECASE)


def _paragraph_heading_level(paragraph, value: str, style_name: str) -> int | None:
    """Infer a heading without requiring authors to use Word Heading styles.

    Real internal documents frequently format section labels as short bold
    ``Normal`` paragraphs.  We combine explicit Word structure with conservative
    visual/text cues and intentionally reject sentence-like long paragraphs.
    """

    explicit = _HEADING_STYLE.search(style_name or "")
    if explicit:
        return int(explicit.group(1) or 1)

    text = value.strip()
    if (
        not text
        or not any(char.isalnum() for char in text)
        or len(text) > 180
        or len(text.split()) > 24
    ):
        return None

    # OOXML outline level can carry semantic heading information even when the
    # visible style name is customized.
    outline = paragraph._p.xpath("./w:pPr/w:outlineLvl/@w:val")
    if outline:
        try:
            return max(1, min(9, int(outline[0]) + 1))
        except (TypeError, ValueError):
            pass

    visible_runs = [run for run in paragraph.runs if run.text]
    visible_chars = sum(len(run.text) for run in visible_runs)
    bold_chars = sum(len(run.text) for run in visible_runs if run.bold is True)
    bold_ratio = bold_chars / max(1, visible_chars)

    sizes = [run.font.size.pt for run in visible_runs if run.font.size is not None]
    large_run = bool(sizes and median(sizes) >= 13.0)
    centered = getattr(paragraph, "alignment", None) is not None and int(paragraph.alignment) == 1
    ends_with_colon = text.endswith(":")
    all_caps = any(char.isalpha() for char in text) and text.upper() == text

    # A short bold standalone label is a strong signal.  A colon is common in
    # Vietnamese reports ("Mục tiêu của đề tài:", "Kết quả đạt được:").
    if bold_ratio >= 0.75 and (ends_with_colon or len(text.split()) <= 14):
        return 1
    if ends_with_colon and len(text.split()) <= 14 and bold_ratio >= 0.35:
        return 1
    if all_caps and len(text.split()) <= 14 and (bold_ratio >= 0.5 or centered or large_run):
        return 1
    if centered and bold_ratio >= 0.75 and len(text.split()) <= 14:
        return 1
    return None


class DOCXParser(DocumentParser):
    """Parse paragraphs and tables in Word body order with heading context."""

    def __init__(self, config: ParserConfig | None = None):
        self.config = config or ParserConfig()

    def parse(self, source):
        path = source_path(source)
        validate_zip_archive(path, self.config)
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = Document(str(path))
        segments: list[dict] = []
        headings: dict[int, str] = {}
        paragraph_buffer: list[str] = []
        buffer_type = "paragraph"
        buffer_start = 0
        block_index = 0
        table_index = 0
        page_number = 1
        pending_caption: str | None = None

        def current_path() -> list[str]:
            return [headings[level] for level in sorted(headings)]

        def common(block_type: str) -> dict:
            values = current_path()
            return {
                "method": "docx",
                "heading": values[-1] if values else path.stem,
                "heading_path": values,
                "heading_level": max(headings) if headings else None,
                "section": " > ".join(values) if values else path.stem,
                "block_type": block_type,
                "block_index": block_index,
                "page": page_number,
                "page_number": page_number,
            }

        def flush(end_index: int) -> None:
            nonlocal paragraph_buffer, buffer_type
            if paragraph_buffer:
                segments.append(
                    {
                        "text": "\n".join(paragraph_buffer),
                        **common(buffer_type),
                        "paragraph_start": buffer_start,
                        "paragraph_end": end_index,
                    }
                )
                paragraph_buffer = []
                buffer_type = "paragraph"

        def paragraph_text(paragraph: Paragraph) -> str:
            value = paragraph.text.strip()
            style_name = paragraph.style.name if paragraph.style is not None else ""
            if style_name.lower().startswith("list bullet"):
                value = f"- {value}"
            elif style_name.lower().startswith("list number"):
                value = f"1. {value}"
            drawings = paragraph._p.xpath(".//w:drawing")
            for drawing in drawings:
                labels = drawing.xpath(".//*[local-name()='docPr']/@descr | .//*[local-name()='docPr']/@title")
                label = next((str(item).strip() for item in labels if str(item).strip()), "")
                if label:
                    value = f"{value}\n[Image: {label}]".strip()
            return value

        for body_position, child in enumerate(document.element.body, 1):
            block_index = body_position
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
                style_name = paragraph.style.name if paragraph.style is not None else ""
                value = paragraph_text(paragraph)
                heading_level = _paragraph_heading_level(paragraph, value, style_name) if value else None
                if value and heading_level is not None:
                    flush(body_position - 1)
                    level = int(heading_level)
                    for stale in [candidate for candidate in headings if candidate >= level]:
                        del headings[stale]
                    headings[level] = value
                    buffer_start = body_position
                    buffer_type = "section"
                    paragraph_buffer = [value]
                elif value and style_name.lower().startswith("caption"):
                    flush(body_position - 1)
                    pending_caption = value
                    segments.append({"text": value, **common("caption"), "paragraph_start": body_position, "paragraph_end": body_position})
                elif value:
                    if not paragraph_buffer:
                        buffer_start = body_position
                        buffer_type = "paragraph"
                    paragraph_buffer.append(value)

                page_breaks = len(paragraph._p.xpath(".//w:br[@w:type='page'] | .//w:lastRenderedPageBreak"))
                page_number += page_breaks
                continue

            if child.tag.endswith("}tbl"):
                flush(body_position - 1)
                table_index += 1
                table = Table(child, document)
                rendered_rows: list[list[str]] = []
                merge_spans: list[dict] = []
                for row_number, row in enumerate(table.rows, 1):
                    values: list[str] = []
                    seen_cells: set[int] = set()
                    for column_number, cell in enumerate(row.cells, 1):
                        identity = id(cell._tc)
                        if identity in seen_cells:
                            values.append("")
                            continue
                        seen_cells.add(identity)
                        value = " ".join(part.strip() for part in cell.text.splitlines() if part.strip())
                        values.append(value)
                        grid_span = cell._tc.xpath("./w:tcPr/w:gridSpan/@w:val")
                        vertical_merge = cell._tc.xpath("./w:tcPr/w:vMerge/@w:val")
                        if grid_span or vertical_merge:
                            merge_spans.append(
                                {
                                    "row": row_number,
                                    "column": column_number,
                                    "column_span": int(grid_span[0]) if grid_span else 1,
                                    "vertical_merge": vertical_merge[0] if vertical_merge else "continue",
                                }
                            )
                    rendered_rows.append(values)
                if rendered_rows:
                    width = max(len(row) for row in rendered_rows)
                    rendered_rows = [row + [""] * (width - len(row)) for row in rendered_rows]
                    header = rendered_rows[0]
                    lines = [
                        "| " + " | ".join(header) + " |",
                        "| " + " | ".join(["---"] * width) + " |",
                    ]
                    lines.extend("| " + " | ".join(row) + " |" for row in rendered_rows[1:])
                    segments.append(
                        {
                            "text": "\n".join(lines),
                            **common("table"),
                            "method": "docx-table",
                            "table_id": f"docx-table-{table_index}",
                            "table_index": table_index,
                            "column_names": header,
                            "row_start": 1,
                            "row_end": len(rendered_rows),
                            "merged_cells": merge_spans,
                            "caption": pending_caption,
                        }
                    )
                pending_caption = None

        flush(len(document.element.body))
        return parsed_documents(source, path, segments)
