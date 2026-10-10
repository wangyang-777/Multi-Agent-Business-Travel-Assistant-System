"""Structure-first PDF chunking with embedding-guided sentence boundaries."""

from __future__ import annotations

import math
import re
from typing import Any

from app.etl.pipeline import DocumentChunk, _split_long_unit, _split_units


_ARTICLE_HEADING = re.compile(r"(?m)^\s*(第[一二三四五六七八九十百零〇0-9]+条)")
_APPENDIX_HEADING = re.compile(r"(?m)^\s*附件\s*$")


def group_pdf_articles(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Join article continuations across pages, keeping extracted tables separate."""

    paragraphs = [
        chunk for chunk in chunks
        if "table" not in chunk.metadata.get("block_types", [])
        and "image_ocr" not in chunk.metadata.get("block_types", [])
    ]
    protected = [
        chunk for chunk in chunks
        if "table" in chunk.metadata.get("block_types", [])
        or "image_ocr" in chunk.metadata.get("block_types", [])
    ]
    output: list[DocumentChunk] = []
    pending: list[str] = []
    pending_pages: list[int] = []
    pending_article: str | None = None
    pending_metadata: dict[str, Any] = {}

    def flush() -> None:
        if not pending:
            return
        metadata = dict(pending_metadata)
        metadata["article_no"] = pending_article
        metadata["page_start"] = min(pending_pages)
        metadata["page_end"] = max(pending_pages)
        metadata["page_numbers"] = sorted(set(pending_pages))
        metadata["block_types"] = ["paragraph"]
        content = "\n\n".join(part for part in pending if part.strip()).strip()
        if content:
            output.append(DocumentChunk(content=content, metadata=metadata))
        pending.clear()
        pending_pages.clear()

    for chunk in paragraphs:
        page = int(chunk.metadata.get("page_number") or 0)
        content = chunk.content
        article_matches = list(_ARTICLE_HEADING.finditer(content))
        appendix_matches = list(_APPENDIX_HEADING.finditer(content))
        boundaries = sorted(
            [(match.start(), match.group(1), match.end()) for match in article_matches]
            + [(match.start(), "附件", match.end()) for match in appendix_matches]
        )
        if not boundaries:
            if pending and pending_article is None and page != pending_pages[-1]:
                flush()
            if not pending:
                pending_metadata = dict(chunk.metadata)
            pending.append(content)
            pending_pages.append(page)
            continue

        prefix = content[:boundaries[0][0]].strip()
        if prefix:
            if not pending:
                pending_metadata = dict(chunk.metadata)
            pending.append(prefix)
            pending_pages.append(page)
        for index, (start, article_no, _end) in enumerate(boundaries):
            flush()
            end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(content)
            pending_metadata = dict(chunk.metadata)
            pending_article = article_no
            pending.append(content[start:end].strip())
            pending_pages.append(page)
    flush()

    for chunk in protected:
        metadata = dict(chunk.metadata)
        page = int(metadata.get("page_number") or 0)
        metadata.update({"page_start": page, "page_end": page, "page_numbers": [page]})
        output.append(DocumentChunk(content=chunk.content, metadata=metadata))
    return sorted(output, key=lambda item: (
        int(item.metadata.get("page_start") or 0),
        int(item.metadata.get("page_end") or 0),
        0 if "table" not in item.metadata.get("block_types", []) else 1,
    ))


def expand_table_rows(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Index each table row with its headers while retaining the whole table.

    A whole-table vector can hide a short city/grade row among unrelated rows.
    The row copies give retrieval a focused match; the original table remains
    available for questions that compare multiple rows.
    """

    output: list[DocumentChunk] = []
    for chunk in chunks:
        output.append(chunk)
        if "table" not in chunk.metadata.get("block_types", []):
            continue
        lines = [line.strip() for line in chunk.content.splitlines() if line.strip()]
        if len(lines) < 4 or not lines[1].startswith("|"):
            continue
        table_label, header = lines[:2]
        if not re.fullmatch(r"\[[^\]]+\]", table_label):
            continue
        context_line = next(
            (line for line in lines[2:] if line.startswith("[Context] ")), ""
        )
        row_index = 0
        carried_values: list[str] = []
        for line in lines[2:]:
            if not line.startswith("|") or re.fullmatch(r"[|:\-\s]+", line):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not any(cells):
                continue
            if not carried_values:
                carried_values = [""] * len(cells)
            for index, cell in enumerate(cells):
                if index < len(carried_values) and cell:
                    carried_values[index] = cell
            row_index += 1
            metadata = dict(chunk.metadata)
            metadata.update({"table_row_index": row_index, "table_parent": table_label})
            # Merged cells in PDF tables are blank in continuation rows. Carry
            # their heading into the row text without changing the source row.
            context = " | ".join(
                carried_values[index]
                for index, cell in enumerate(cells[:-2])
                if not cell and carried_values[index]
            )
            prefix = f"{table_label}\n{header}"
            if context_line:
                prefix += f"\n{context_line}"
            if context:
                prefix += f"\n上级分组：{context}"
            output.append(DocumentChunk(content=f"{prefix}\n{line}", metadata=metadata))
    return output


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _semantic_boundary(
    units: list[str], similarities: list[float], *, start: int, min_chars: int,
    target_chars: int, max_chars: int,
) -> int:
    """Choose the lowest-similarity feasible boundary near the desired size."""

    candidates: list[tuple[float, int]] = []
    length = 0
    for end in range(start + 1, len(units)):
        length += len(units[end - 1]) + (1 if end > start + 1 else 0)
        if length > max_chars:
            break
        if length < min_chars:
            continue
        distance_penalty = 0.06 * abs(length - target_chars) / max(target_chars, 1)
        candidates.append((similarities[end - 1] + distance_penalty, end))
    if candidates:
        return min(candidates)[1]
    return min(start + 1, len(units))


async def split_on_embedding_similarity(
    chunks: list[DocumentChunk],
    embedder: Any,
    *,
    min_chars: int = 100,
    target_chars: int = 180,
    max_chars: int = 300,
) -> list[DocumentChunk]:
    """Split long article bodies at low adjacent-sentence cosine similarity."""

    if not 0 < min_chars <= target_chars <= max_chars:
        raise ValueError("semantic chunk sizes must satisfy 0 < min <= target <= max")
    output: list[DocumentChunk] = []
    for chunk in chunks:
        if set(chunk.metadata.get("block_types") or []) & {"table", "image_ocr", "faq", "code"}:
            output.append(chunk)
            continue
        units = [
            piece
            for unit in _split_units(chunk.content)
            for piece in _split_long_unit(unit, chunk_size=max_chars)
        ]
        if len(chunk.content) <= max_chars or len(units) < 2:
            output.append(chunk)
            continue
        vectors = await embedder.embed_texts(units)
        if len(vectors) != len(units):
            raise ValueError("embedding response did not match semantic units")
        similarities = [
            _cosine(vectors[index], vectors[index + 1])
            for index in range(len(vectors) - 1)
        ]
        start = 0
        segment_index = 1
        while start < len(units):
            remaining = "\n".join(units[start:])
            if len(remaining) <= max_chars:
                end = len(units)
            else:
                end = _semantic_boundary(
                    units, similarities, start=start, min_chars=min_chars,
                    target_chars=target_chars, max_chars=max_chars,
                )
            content = "\n".join(units[start:end]).strip()
            metadata = dict(chunk.metadata)
            metadata.update({
                "semantic_split": True,
                "semantic_segment_index": segment_index,
                "semantic_boundary_similarity": (
                    round(similarities[end - 1], 4) if end < len(units) else None
                ),
            })
            if segment_index > 1 and metadata.get("article_no"):
                content = f"{metadata['article_no']}（续）\n{content}"
            output.append(DocumentChunk(content=content, metadata=metadata))
            segment_index += 1
            start = end
    return output
