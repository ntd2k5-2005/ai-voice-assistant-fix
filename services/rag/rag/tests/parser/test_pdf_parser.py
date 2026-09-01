from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from rag.config import OCRConfig, ParserConfig
from rag.contracts import DocumentSource
from rag.ingestion.parsers import PDFParser, ParserError, ParserErrorCode, ParserRegistry


def _unicode_font_path() -> Path:
    candidates: list[Path] = []
    if sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates.extend(fonts / name for name in ("arial.ttf", "calibri.ttf", "tahoma.ttf"))
    elif sys.platform == "darwin":
        candidates.extend(
            [
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
                Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
                Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
                Path("/usr/local/share/fonts/DejaVuSans.ttf"),
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("no host Unicode TrueType font is available for the Vietnamese PDF fixture")


def _font_name():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    path = _unicode_font_path()
    name = f"FixtureUnicode{abs(hash(path.resolve()))}"
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    return name


def _image_with_text(path: Path, text: str) -> None:
    import reportlab
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1800, 900), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf"
    assert font_path.is_file(), f"ReportLab OCR fixture font is missing: {font_path}"
    font = ImageFont.truetype(str(font_path), 64)
    draw.text((100, 350), text, fill="black", font=font)
    image.save(path)


def _mixed_pdf(path: Path, image_path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setFont("Helvetica", 14)
    pdf.drawString(72, 700, "Native page contains searchable text and remains native extraction.")
    pdf.showPage()
    pdf.drawImage(str(image_path), 50, 220, width=510, height=255, preserveAspectRatio=True)
    pdf.showPage()
    pdf.save()


def test_normal_unicode_pdf_page_provenance_and_repeated_margin_removal(tmp_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "normal.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    font = _font_name()
    for page in range(1, 4):
        pdf.setFont(font, 10)
        pdf.drawString(72, 760, "REPEATED REPORT HEADER")
        pdf.drawString(72, 20, f"Page {page}")
        pdf.setFont(font, 13)
        pdf.drawString(72, 680, f"Nội dung tiếng Việt của trang {page} có provenance chính xác.")
        pdf.showPage()
    pdf.save()

    parsed = ParserRegistry().parse(DocumentSource(str(path)))
    pages = [item for item in parsed if item.metadata["block_type"] == "page_text"]
    assert [item.metadata["page_number"] for item in pages] == [1, 2, 3]
    assert all("REPEATED REPORT HEADER" not in item.content for item in pages)
    assert all(item.metadata["removed_repeated_lines"] for item in pages)
    assert "tiếng Việt" in pages[0].content
    assert all(item.metadata["ocr_used"] is False for item in pages)


def test_scanned_pdf_uses_real_tesseract_ocr(tmp_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    image_path = tmp_path / "scan.png"
    _image_with_text(image_path, "SCANNED KNOWLEDGE PAGE")
    path = tmp_path / "scan.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawImage(str(image_path), 50, 220, width=510, height=255, preserveAspectRatio=True)
    pdf.showPage()
    pdf.save()

    config = ParserConfig(ocr=OCRConfig(strategy="auto", backend="tesseract", languages=("eng",), dpi=200))
    parsed = ParserRegistry(config).parse(DocumentSource(str(path)))
    page = next(item for item in parsed if item.metadata["block_type"] == "page_text")
    assert page.metadata["ocr_used"] is True
    assert page.metadata["parser_method"] == "pdf-ocr-tesseract"
    assert "SCANNED" in page.content.upper()
    assert page.metadata["confidence"] is not None


def test_mixed_pdf_ocr_is_applied_per_page_only(tmp_path):
    image_path = tmp_path / "scan.png"
    _image_with_text(image_path, "SECOND PAGE OCR CONTENT")
    path = tmp_path / "mixed.pdf"
    _mixed_pdf(path, image_path)

    config = ParserConfig(ocr=OCRConfig(strategy="auto", backend="tesseract", languages=("eng",), dpi=200))
    parsed = ParserRegistry(config).parse(DocumentSource(str(path)))
    pages = [item for item in parsed if item.metadata["block_type"] == "page_text"]
    assert [item.metadata["page_number"] for item in pages] == [1, 2]
    assert pages[0].metadata["ocr_used"] is False
    assert pages[1].metadata["ocr_used"] is True
    assert "SECOND PAGE" in pages[1].content.upper()


def test_auto_ocr_falls_back_and_reports_every_backend_failure(monkeypatch):
    parser = PDFParser(ocr_backend="auto")
    calls = []

    def unavailable_tesseract(page):
        calls.append(("tesseract", page))
        raise RuntimeError("tesseract executable is unavailable")

    def working_rapidocr(page):
        calls.append(("rapidocr", page))
        return ([{"text": "fallback text", "bbox": [0, 0, 1, 1]}], 0.91, "rapidocr")

    monkeypatch.setattr(parser, "_ocr_tesseract", unavailable_tesseract)
    monkeypatch.setattr(parser, "_ocr_rapid", working_rapidocr)
    page = object()
    lines, confidence, backend = parser._ocr_page(page)
    assert calls == [("tesseract", page), ("rapidocr", page)]
    assert lines[0]["text"] == "fallback text"
    assert confidence == pytest.approx(0.91)
    assert backend == "rapidocr"

    failing = PDFParser(ocr_backend="auto")

    def broken_tesseract(_page):
        raise ValueError("missing language data")

    def broken_rapidocr(_page):
        raise LookupError("model initialization failed")

    monkeypatch.setattr(failing, "_ocr_tesseract", broken_tesseract)
    monkeypatch.setattr(failing, "_ocr_rapid", broken_rapidocr)
    with pytest.raises(RuntimeError, match="all configured OCR backends failed") as error:
        failing._ocr_page(object())
    message = str(error.value)
    assert "tesseract=ValueError: missing language data" in message
    assert "rapidocr=LookupError: model initialization failed" in message


def test_multicolumn_pdf_reads_left_column_before_right_column(tmp_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "columns.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.setFont("Helvetica", 10)
    for row in range(1, 13):
        y = 740 - row * 28
        pdf.drawString(45, y, f"LEFT COLUMN LINE {row:02d} alpha beta gamma")
        pdf.drawString(335, y, f"RIGHT COLUMN LINE {row:02d} delta epsilon zeta")
    pdf.showPage()
    pdf.save()

    parsed = ParserRegistry().parse(DocumentSource(str(path)))
    page = next(item for item in parsed if item.metadata["block_type"] == "page_text")
    assert page.metadata["column_count"] == 2
    assert page.content.index("LEFT COLUMN LINE 12") < page.content.index("RIGHT COLUMN LINE 01")
    assert all(line["bbox"] is not None for line in page.metadata["layout_lines"])


def test_pdf_table_is_a_separate_structured_unit(tmp_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = tmp_path / "table.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    x_positions = [72, 220, 380]
    y_positions = [700, 660, 620]
    for x in x_positions:
        pdf.line(x, y_positions[-1], x, y_positions[0])
    for y in y_positions:
        pdf.line(x_positions[0], y, x_positions[-1], y)
    pdf.drawString(82, 675, "Name")
    pdf.drawString(230, 675, "Role")
    pdf.drawString(82, 635, "Lan")
    pdf.drawString(230, 635, "Engineer")
    pdf.showPage()
    pdf.save()

    parsed = ParserRegistry().parse(DocumentSource(str(path)))
    table = next(item for item in parsed if item.metadata["block_type"] == "table")
    assert table.metadata["page_number"] == 1
    assert table.metadata["column_names"] == ["Name", "Role"]
    assert "| Lan | Engineer |" in table.content
    assert table.metadata["bounding_box"] is not None


def test_empty_encrypted_corrupt_and_page_limit_fail_explicitly(tmp_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.pdfencrypt import StandardEncryption
    from reportlab.pdfgen import canvas

    empty_path = tmp_path / "empty.pdf"
    empty_pdf = canvas.Canvas(str(empty_path), pagesize=letter)
    empty_pdf.showPage()
    empty_pdf.save()
    with pytest.raises(ParserError) as empty_error:
        ParserRegistry(ParserConfig(ocr=OCRConfig(strategy="never"))).parse(DocumentSource(str(empty_path)))
    assert empty_error.value.code is ParserErrorCode.EXTRACTION_FAILED
    assert empty_error.value.diagnostics["empty_pages"] == [1]

    encrypted_path = tmp_path / "encrypted.pdf"
    encryption = StandardEncryption("secret", canPrint=0)
    encrypted_pdf = canvas.Canvas(str(encrypted_path), pagesize=letter, encrypt=encryption)
    encrypted_pdf.drawString(72, 700, "Protected content")
    encrypted_pdf.showPage()
    encrypted_pdf.save()
    with pytest.raises(ParserError) as encrypted_error:
        ParserRegistry().parse(DocumentSource(str(encrypted_path)))
    assert encrypted_error.value.code is ParserErrorCode.ENCRYPTED_CONTENT

    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.7\ninvalid")
    with pytest.raises(ParserError) as corrupt_error:
        ParserRegistry().parse(DocumentSource(str(corrupt_path)))
    assert corrupt_error.value.code is ParserErrorCode.CORRUPT_CONTENT

    image_path = tmp_path / "scan.png"
    _image_with_text(image_path, "PAGE TWO")
    mixed_path = tmp_path / "two-pages.pdf"
    _mixed_pdf(mixed_path, image_path)
    limited = ParserRegistry(ParserConfig(max_pages=1))
    with pytest.raises(ParserError) as page_error:
        limited.parse(DocumentSource(str(mixed_path)))
    assert page_error.value.code is ParserErrorCode.INVALID_SOURCE
