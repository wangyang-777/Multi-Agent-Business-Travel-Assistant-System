from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.document_parser import extract_text_from_file


SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
}

UNSTRUCTURED_FIRST_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".md", ".html", ".htm"}


class DocumentLoadError(ValueError):
    pass


@dataclass(slots=True)
class LoadedDocument:
    text: str
    loader: str
    element_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def load_document_from_file(filename: str, data: bytes) -> LoadedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise DocumentLoadError(f"不支持的文件类型：{suffix or 'unknown'}，当前支持 {supported}")

    if suffix in UNSTRUCTURED_FIRST_EXTENSIONS:
        try:
            loaded = _load_with_unstructured(filename, data)
            if loaded.text.strip():
                return loaded
        except ImportError as exc:
            if suffix not in {".txt", ".md", ".pdf", ".docx"}:
                raise DocumentLoadError(f"{suffix} 文件需要安装本地 Unstructured 解析依赖") from exc
        except Exception:
            if suffix not in {".txt", ".md", ".pdf", ".docx"}:
                raise

    try:
        text = extract_text_from_file(filename, data)
    except ValueError as exc:
        raise DocumentLoadError(str(exc)) from exc
    return LoadedDocument(
        text=text,
        loader="simple",
        element_count=1 if text.strip() else 0,
        metadata={"filename": filename, "fallback": True},
    )


def _load_with_unstructured(filename: str, data: bytes) -> LoadedDocument:
    from unstructured.partition.auto import partition

    file_obj = io.BytesIO(data)
    file_obj.name = filename
    suffix = Path(filename).suffix.lower()
    kwargs: dict[str, Any] = {
        "file": file_obj,
        "metadata_filename": filename,
        "include_page_breaks": True,
    }
    elements = partition(**kwargs)
    blocks: list[str] = []
    for element in elements:
        text = str(element).strip()
        if not text:
            continue
        category = getattr(element, "category", None) or element.__class__.__name__
        page_number = getattr(getattr(element, "metadata", None), "page_number", None)
        prefix = f"[Page {page_number}] " if page_number else ""
        if category in {"Title", "Header", "Footer"}:
            blocks.append(f"\n{prefix}{text}")
        else:
            blocks.append(f"{prefix}{text}")

    metadata = {
        "filename": filename,
        "loader": "unstructured",
        "element_types": sorted({element.__class__.__name__ for element in elements}),
    }
    return LoadedDocument(
        text="\n\n".join(blocks).strip(),
        loader="unstructured",
        element_count=len(elements),
        metadata=metadata,
    )
