from __future__ import annotations

import fitz

from app.services.document_parser import extract_pdf_document


def test_pdf_parser_preserves_page_metadata_and_body_text() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Business travel hotel limit is 800 CNY. Approval is required above the limit.",
    )
    payload = document.tobytes()
    document.close()

    text, metadata = extract_pdf_document(payload)

    assert "[Page 1]" in text
    assert "800 CNY" in text
    assert metadata["loader"] == "pymupdf-layout"
    assert metadata["layout_analysis"] is True
    assert metadata["pages"][0]["page_number"] == 1
    assert metadata["pages"][0]["ocr_required"] is False
