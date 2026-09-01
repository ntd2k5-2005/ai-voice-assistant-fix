"""Per-page PDF extraction with native text, layout, tables, and OCR fallback."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from statistics import mean
from typing import Any

from .base import DocumentParser, ParserError, ParserErrorCode, parsed_documents, source_path


_CID = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
_REPLACEMENT = re.compile(r"[\ufffd\u0000]")


def _normalize_repeated_line(value: str) -> str:
    value = re.sub(r"\d+", "#", value.casefold())
    return re.sub(r"\s+", " ", value).strip(" -–—|\t")


def _markdown_table(rows: list[list[Any]]) -> tuple[str, list[str]]:
    cleaned = [
        ["" if cell is None else re.sub(r"\s+", " ", str(cell)).strip() for cell in row]
        for row in rows if row
    ]
    if not cleaned:
        return "", []
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    headers = [value or f"column_{index}" for index, value in enumerate(cleaned[0], 1)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in cleaned[1:])
    return "\n".join(lines), headers


@dataclass
class PDFParser(DocumentParser):
    ocr_strategy: str = "auto"
    ocr_backend: str = "auto"
    ocr_languages: tuple[str, ...] = ("vie", "eng")
    ocr_dpi: int = 200
    min_native_characters: int = 24
    ocr_timeout_seconds: float = 120.0
    max_pages: int = 2_000

    @classmethod
    def from_config(cls, config):
        ocr = getattr(config, "ocr", config)
        return cls(
            ocr_strategy=getattr(ocr, "strategy", "auto"),
            ocr_backend=getattr(ocr, "backend", "auto"),
            ocr_languages=tuple(getattr(ocr, "languages", ("vie", "eng"))),
            ocr_dpi=int(getattr(ocr, "dpi", 200)),
            min_native_characters=int(getattr(ocr, "min_text_characters", 24)),
            ocr_timeout_seconds=float(getattr(ocr, "timeout_seconds", 120.0)),
            max_pages=int(getattr(config, "max_pages", 2_000)),
        )

    def parse(self, source):
        path = source_path(source)
        try:
            import pdfplumber

            pdf = pdfplumber.open(str(path))
        except Exception as exc:
            lowered = str(exc).casefold()
            code = (
                ParserErrorCode.ENCRYPTED_CONTENT
                if any(token in lowered for token in ("password", "encrypted", "decrypt"))
                or self._has_encryption_dictionary(path)
                else ParserErrorCode.CORRUPT_CONTENT
            )
            raise ParserError(code, f"cannot open PDF: {type(exc).__name__}: {exc}", source=path, stage="open") from exc

        try:
            if len(pdf.pages) > self.max_pages:
                raise ParserError(
                    ParserErrorCode.INVALID_SOURCE,
                    f"PDF has {len(pdf.pages)} pages, exceeding configured limit {self.max_pages}",
                    source=path,
                    stage="intake",
                )
            page_results: list[dict] = []
            table_results: list[dict] = []
            warnings: list[str] = []
            empty_pages: list[int] = []

            for page_number, page in enumerate(pdf.pages, 1):
                tables, table_boxes, table_warnings = self._tables(page)
                warnings.extend(f"page {page_number}: {warning}" for warning in table_warnings)
                native_lines, column_count, native_warning = self._native_lines(page, table_boxes)
                if native_warning:
                    warnings.append(f"page {page_number}: {native_warning}")
                native_text = "\n".join(line["text"] for line in native_lines if line["text"].strip())
                native_character_count = len("".join(native_text.split()))
                native_usable = self._usable_native_text(native_text)
                should_ocr = self.ocr_strategy == "always" or (
                    self.ocr_strategy == "auto" and not native_usable
                )
                ocr_used = False
                confidence = 1.0 if native_usable else None
                lines = native_lines
                method = "pdf-text"

                if should_ocr and self.ocr_strategy != "never":
                    try:
                        ocr_lines, ocr_confidence, backend = self._ocr_page(page)
                        ocr_text = "\n".join(line["text"] for line in ocr_lines if line["text"].strip())
                        if ocr_text.strip():
                            lines = ocr_lines
                            native_text = ocr_text
                            confidence = ocr_confidence
                            ocr_used = True
                            method = f"pdf-ocr-{backend}"
                            column_count = 1
                    except Exception as exc:
                        warnings.append(f"page {page_number}: OCR failed: {type(exc).__name__}: {exc}")

                page_results.append(
                    {
                        "page_number": page_number,
                        "page": page_number,
                        "width": float(page.width),
                        "height": float(page.height),
                        "text": native_text.strip(),
                        "lines": lines,
                        "method": method,
                        "ocr_used": ocr_used,
                        "confidence": confidence,
                        "native_character_count": native_character_count,
                        "column_count": column_count,
                        "image_count": len(getattr(page, "images", []) or []),
                    }
                )
                if not native_text.strip() and not tables:
                    empty_pages.append(page_number)

                for table_index, table in enumerate(tables, 1):
                    rendered, headers = _markdown_table(table["rows"])
                    if not rendered:
                        continue
                    table_results.append(
                        {
                            "text": rendered,
                            "method": "pdf-table",
                            "page": page_number,
                            "page_number": page_number,
                            "section": f"page {page_number} table {table_index}",
                            "heading": f"Page {page_number}",
                            "block_type": "table",
                            "table_id": f"page-{page_number}-table-{table_index}",
                            "table_index": table_index,
                            "column_names": headers,
                            "row_start": 1,
                            "row_end": len(table["rows"]),
                            "bounding_box": table.get("bbox"),
                        }
                    )

            removed = self._remove_repeated_margins(page_results)
            records: list[dict] = []
            table_by_page: dict[int, list[dict]] = {}
            for table in table_results:
                table_by_page.setdefault(int(table["page_number"]), []).append(table)
            for page in page_results:
                page_number = int(page["page_number"])
                value = "\n".join(line["text"] for line in page["lines"] if line["text"].strip()).strip()
                if value:
                    records.append(
                        {
                            **{key: item for key, item in page.items() if key != "lines"},
                            "text": value,
                            "section": f"page {page_number}",
                            "heading": f"Page {page_number}",
                            "block_type": "page_text",
                            "layout_lines": page["lines"],
                            "removed_repeated_lines": removed.get(page_number, []),
                            "document_page_count": len(page_results),
                            "empty_pages": empty_pages,
                            "parser_warnings": warnings,
                        }
                    )
                records.extend(table_by_page.get(page_number, []))

            if not records:
                raise ParserError(
                    ParserErrorCode.EXTRACTION_FAILED,
                    "PDF produced no usable text or tables",
                    source=path,
                    stage="extract",
                    diagnostics={"empty_pages": empty_pages, "warnings": warnings},
                )
            return parsed_documents(source, path, records)
        finally:
            pdf.close()

    @staticmethod
    def _has_encryption_dictionary(path) -> bool:
        """Inspect bounded head/tail regions for the standard PDF Encrypt key."""

        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                head = handle.read(min(size, 1024 * 1024))
                tail = b""
                if size > len(head):
                    handle.seek(max(0, size - 1024 * 1024))
                    tail = handle.read()
            return b"/Encrypt" in head or b"/Encrypt" in tail
        except OSError:
            return False

    def _usable_native_text(self, text: str) -> bool:
        compact = "".join(text.split())
        if len(compact) < self.min_native_characters:
            return False
        if _CID.search(text) or _REPLACEMENT.search(text):
            return False
        printable = sum(char.isprintable() for char in compact)
        alphanumeric = sum(char.isalnum() for char in compact)
        return printable / max(1, len(compact)) >= 0.95 and alphanumeric >= max(3, len(compact) // 10)

    def _tables(self, page) -> tuple[list[dict], list[tuple[float, float, float, float]], list[str]]:
        tables: list[dict] = []
        boxes: list[tuple[float, float, float, float]] = []
        warnings: list[str] = []
        try:
            found = page.find_tables() or []
            for table in found:
                rows = table.extract() or []
                if rows:
                    bbox = tuple(float(value) for value in table.bbox)
                    tables.append({"rows": rows, "bbox": bbox})
                    boxes.append(bbox)
        except Exception as exc:
            warnings.append(f"table detection failed: {type(exc).__name__}")
        if not tables:
            try:
                for rows in page.extract_tables() or []:
                    if rows:
                        tables.append({"rows": rows, "bbox": None})
            except Exception as exc:
                warnings.append(f"table extraction failed: {type(exc).__name__}")
        return tables, boxes, warnings

    def _native_lines(self, page, table_boxes):
        warning = None
        try:
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=False,
            ) or []
        except Exception as exc:
            words = []
            warning = f"word extraction failed: {type(exc).__name__}"
        if table_boxes:
            words = [word for word in words if not self._inside_any_table(word, table_boxes)]
        if not words:
            try:
                text = page.extract_text(layout=True) or ""
            except Exception as exc:
                return [], 1, warning or f"text extraction failed: {type(exc).__name__}"
            return [
                {"text": line.strip(), "bbox": None, "top": None, "bottom": None}
                for line in text.splitlines() if line.strip()
            ], 1, warning

        split = self._column_split(words, float(page.width))
        if split is None:
            return self._words_to_lines(words), 1, warning
        left = [word for word in words if (float(word["x0"]) + float(word["x1"])) / 2 < split]
        right = [word for word in words if (float(word["x0"]) + float(word["x1"])) / 2 >= split]
        return self._words_to_lines(left) + self._words_to_lines(right), 2, warning

    @staticmethod
    def _inside_any_table(word, boxes) -> bool:
        x = (float(word["x0"]) + float(word["x1"])) / 2
        y = (float(word["top"]) + float(word["bottom"])) / 2
        return any(x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in boxes)

    @staticmethod
    def _column_split(words, width: float) -> float | None:
        if len(words) < 20 or width <= 0:
            return None
        bins = 100
        coverage = [0] * bins
        for word in words:
            x0, x1 = float(word["x0"]), float(word["x1"])
            if x1 - x0 > width * 0.7:
                continue
            start = max(0, min(bins - 1, int(x0 / width * bins)))
            end = max(start, min(bins - 1, int(math.ceil(x1 / width * bins))))
            for index in range(start, end + 1):
                coverage[index] += 1
        maximum = max(coverage) if coverage else 0
        if not maximum:
            return None
        threshold = max(0, int(maximum * 0.04))
        runs: list[tuple[int, int]] = []
        start = None
        for index in range(30, 71):
            if coverage[index] <= threshold and start is None:
                start = index
            elif coverage[index] > threshold and start is not None:
                runs.append((start, index - 1))
                start = None
        if start is not None:
            runs.append((start, 70))
        runs = [run for run in runs if run[1] - run[0] + 1 >= 3]
        if not runs:
            return None
        run = max(runs, key=lambda value: value[1] - value[0])
        split = ((run[0] + run[1] + 1) / 2) / bins * width
        left = sum((float(word["x0"]) + float(word["x1"])) / 2 < split for word in words)
        right = len(words) - left
        return split if min(left, right) >= 8 else None

    @staticmethod
    def _words_to_lines(words) -> list[dict]:
        if not words:
            return []
        ordered = sorted(words, key=lambda word: (round(float(word["top"]) / 3), float(word["x0"])))
        groups: list[list[dict]] = []
        for word in ordered:
            if not groups or abs(float(word["top"]) - mean(float(item["top"]) for item in groups[-1])) > 3:
                groups.append([word])
            else:
                groups[-1].append(word)
        lines = []
        for group in groups:
            group.sort(key=lambda word: float(word["x0"]))
            text = " ".join(str(word.get("text", "")).strip() for word in group if str(word.get("text", "")).strip())
            if not text:
                continue
            bbox = [
                min(float(word["x0"]) for word in group),
                min(float(word["top"]) for word in group),
                max(float(word["x1"]) for word in group),
                max(float(word["bottom"]) for word in group),
            ]
            lines.append({"text": text, "bbox": bbox, "top": bbox[1], "bottom": bbox[3]})
        return lines

    def _ocr_page(self, page):
        backend = self.ocr_backend
        if backend == "tesseract":
            return self._ocr_tesseract(page)
        if backend == "rapidocr":
            return self._ocr_rapid(page)
        if backend == "auto":
            failures = []
            for name, callback in (
                ("tesseract", self._ocr_tesseract),
                ("rapidocr", self._ocr_rapid),
            ):
                try:
                    return callback(page)
                except Exception as exc:
                    failures.append(f"{name}={type(exc).__name__}: {exc}")
            raise RuntimeError("all configured OCR backends failed: " + "; ".join(failures))
        raise ValueError(f"unsupported OCR backend: {backend}")

    def _ocr_tesseract(self, page):
        import pytesseract
        from pytesseract import Output

        available = set(pytesseract.get_languages(config=""))
        languages = [language for language in self.ocr_languages if language in available]
        if not languages and "eng" in available:
            languages = ["eng"]
        if not languages:
            raise RuntimeError(f"none of the configured OCR languages are installed: {self.ocr_languages}")
        image = page.to_image(resolution=self.ocr_dpi, antialias=True).original
        try:
            data = pytesseract.image_to_data(
                image,
                lang="+".join(languages),
                config="--oem 3 --psm 3",
                output_type=Output.DICT,
                timeout=self.ocr_timeout_seconds,
            )
        finally:
            image.close()
        scale = 72.0 / self.ocr_dpi
        groups: dict[tuple[int, int, int], list[dict]] = {}
        confidences = []
        for index, value in enumerate(data.get("text", [])):
            value = str(value).strip()
            try:
                confidence = float(data["conf"][index])
            except (KeyError, TypeError, ValueError):
                confidence = -1
            if not value or confidence < 0:
                continue
            key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
            left = float(data["left"][index]) * scale
            top = float(data["top"][index]) * scale
            width = float(data["width"][index]) * scale
            height = float(data["height"][index]) * scale
            groups.setdefault(key, []).append({"text": value, "x0": left, "x1": left + width, "top": top, "bottom": top + height})
            confidences.append(confidence / 100.0)
        lines = []
        for group in groups.values():
            lines.extend(self._words_to_lines(group))
        lines.sort(key=lambda line: (line["top"], line["bbox"][0]))
        return lines, (mean(confidences) if confidences else None), "tesseract"

    def _ocr_rapid(self, page):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError("rapidocr_onnxruntime is not installed") from exc
        image = page.to_image(resolution=self.ocr_dpi, antialias=True).original
        try:
            result, _ = RapidOCR()(image)
        finally:
            image.close()
        scale = 72.0 / self.ocr_dpi
        lines = []
        confidences = []
        for box, text, confidence in result or []:
            xs = [float(point[0]) * scale for point in box]
            ys = [float(point[1]) * scale for point in box]
            lines.append({"text": str(text), "bbox": [min(xs), min(ys), max(xs), max(ys)], "top": min(ys), "bottom": max(ys)})
            confidences.append(float(confidence))
        lines.sort(key=lambda line: (line["top"], line["bbox"][0]))
        return lines, (mean(confidences) if confidences else None), "rapidocr"

    @staticmethod
    def _remove_repeated_margins(pages: list[dict]) -> dict[int, list[str]]:
        candidates: Counter[str] = Counter()
        by_page: dict[int, list[tuple[str, str]]] = {}
        for page in pages:
            height = float(page["height"])
            page_candidates = []
            for line in page["lines"]:
                top, bottom = line.get("top"), line.get("bottom")
                if top is None or bottom is None:
                    continue
                if float(top) <= height * 0.12 or float(bottom) >= height * 0.88:
                    normalized = _normalize_repeated_line(line["text"])
                    if normalized:
                        page_candidates.append((normalized, line["text"]))
            for normalized in {item[0] for item in page_candidates}:
                candidates[normalized] += 1
            by_page[int(page["page_number"])] = page_candidates

        threshold = max(2, math.ceil(len(pages) * 0.6))
        repeated = {value for value, count in candidates.items() if count >= threshold}
        removed: dict[int, list[str]] = {}
        if not repeated:
            return removed
        for page in pages:
            page_number = int(page["page_number"])
            kept = []
            removed_values = []
            for line in page["lines"]:
                normalized = _normalize_repeated_line(line["text"])
                if normalized in repeated:
                    removed_values.append(line["text"])
                else:
                    kept.append(line)
            page["lines"] = kept
            if removed_values:
                removed[page_number] = removed_values
        return removed
