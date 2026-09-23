from __future__ import annotations

import io
from pathlib import Path
from typing import Any

SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def extract_text_from_file(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise ValueError(f"不支持的文件类型：{suffix or 'unknown'}，当前支持 {supported}")
    if suffix in {".txt", ".md"}:
        return _decode_text(data)
    if suffix == ".pdf":
        text, _ = extract_pdf_document(data)
        return text
    if suffix == ".docx":
        return _extract_docx_text(data)
    return ""


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_pdf_text_pypdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n[Page {index}]\n{text.strip()}")
    return "\n".join(pages).strip()


def _markdown_table(rows: list[list[Any]]) -> str:
    normalized = [
        [str(cell or "").replace("|", "\\|").replace("\n", " ").strip() for cell in row]
        for row in rows
        if row
    ]
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def extract_pdf_document(data: bytes) -> tuple[str, dict[str, Any]]:
    """Extract PDF tables/images before body text and OCR low-text pages on demand."""

    try:
        import fitz
    except ImportError:
        return _extract_pdf_text_pypdf(data), {
            "loader": "pypdf",
            "layout_analysis": False,
            "ocr_available": False,
        }

    document = fitz.open(stream=data, filetype="pdf")
    blocks: list[str] = []
    page_metadata: list[dict[str, Any]] = []
    for page_index, page in enumerate(document, start=1):
        page_blocks: list[str] = []
        table_count = 0
        try:
            finder = page.find_tables()
            for table_index, table in enumerate(getattr(finder, "tables", []), start=1):
                markdown = _markdown_table(table.extract())
                if markdown:
                    page_blocks.append(f"[Table {page_index}-{table_index}]\n{markdown}")
                    table_count += 1
        except Exception:
            table_count = 0

        body_text = page.get_text("text", sort=True).strip()
        image_count = len(page.get_images(full=True))
        needs_ocr = len(body_text) < 40 or (image_count > 0 and len(body_text) < 160)
        ocr_applied = False
        ocr_text = ""
        ocr_confidence: float | None = None
        if needs_ocr:
            try:
                import pytesseract
                from PIL import Image

                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                ocr_text = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
                ocr_data = pytesseract.image_to_data(
                    image,
                    lang="chi_sim+eng",
                    output_type=pytesseract.Output.DICT,
                )
                confidences = [
                    float(value)
                    for value in ocr_data.get("conf", [])
                    if str(value).replace(".", "", 1).lstrip("-").isdigit() and float(value) >= 0
                ]
                if confidences:
                    ocr_confidence = sum(confidences) / len(confidences) / 100
                ocr_applied = bool(ocr_text)
            except Exception:
                ocr_applied = False
        if ocr_text and ocr_text not in body_text:
            page_blocks.append(f"[OCR page={page_index}]\n{ocr_text}")
        if body_text:
            page_blocks.append(body_text)
        if page_blocks:
            blocks.append(f"[Page {page_index}]\n" + "\n\n".join(page_blocks))
        page_metadata.append(
            {
                "page_number": page_index,
                "table_count": table_count,
                "image_count": image_count,
                "ocr_required": needs_ocr,
                "ocr_applied": ocr_applied,
                "ocr_confidence": ocr_confidence,
                "needs_review": (
                    needs_ocr
                    and (
                        not ocr_applied
                        or (ocr_confidence is not None and ocr_confidence < 0.6)
                    )
                ),
            }
        )
    document.close()
    return "\n\n".join(blocks).strip(), {
        "loader": "pymupdf-layout",
        "layout_analysis": True,
        "page_count": len(page_metadata),
        "pages": page_metadata,
        "ocr_available": any(item["ocr_applied"] for item in page_metadata),
    }


def _extract_docx_text(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    blocks: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))
    return "\n".join(blocks).strip()
