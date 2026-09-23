"""Document ingestion: parse → chunk → embed → Milvus."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.infrastructure.vector.milvus_client import MilvusVectorClient
from app.utils.tokenization import count_text_tokens, truncate_text_by_tokens


@dataclass(slots=True)
class IngestionConfig:
    chunk_size: int = 3000
    chunk_overlap: int = 450
    embedding_chunk_max_tokens: int = 7000
    embedding_model: str = "text-embedding-3-small"
    default_intent: str = "general"


@dataclass(slots=True)
class DocumentChunk:
    content: str
    metadata: dict[str, Any]


def parse_plain_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _split_units(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    units: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            piece = "\n".join(buffer).strip()
            if piece:
                units.append(piece)
            buffer.clear()

    heading_re = re.compile(
        r"^(第[一二三四五六七八九十百千万0-9]+[章节条部分]|[一二三四五六七八九十]+[、.．]|[0-9]+[.．、])"
    )
    for line in lines:
        if not line:
            flush()
            continue
        if heading_re.match(line):
            flush()
            units.append(line)
            continue
        buffer.append(line)
    flush()

    out: list[str] = []
    sentence_re = re.compile(r"(?<=[。！？!?；;])")
    for unit in units:
        pieces = [piece.strip() for piece in sentence_re.split(unit) if piece.strip()]
        out.extend(pieces or [unit])
    return out


def _split_long_unit(unit: str, *, chunk_size: int) -> list[str]:
    if len(unit) <= chunk_size:
        return [unit]
    separators = ("。", "；", ";", "，", ",", "、", " ")
    chunks: list[str] = []
    remaining = unit.strip()
    while len(remaining) > chunk_size:
        cut = -1
        window = remaining[:chunk_size]
        for sep in separators:
            pos = window.rfind(sep)
            if pos > chunk_size * 0.45:
                cut = pos + len(sep)
                break
        if cut <= 0:
            cut = chunk_size
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [chunk for chunk in chunks if chunk]


def _semantic_overlap_tail(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    selected: list[str] = []
    total = 0
    for unit in reversed(_split_units(text)):
        if not unit:
            continue
        projected = total + len(unit) + (1 if selected else 0)
        if projected > max_chars and selected:
            break
        if projected > max_chars and not selected:
            return ""
        selected.append(unit)
        total = projected
    return "\n".join(reversed(selected)).strip()


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    units: list[str] = []
    for unit in _split_units(text):
        units.extend(_split_long_unit(unit, chunk_size=chunk_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        current = unit
    if current:
        chunks.append(current.strip())

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    with_overlap: list[str] = [chunks[0]]
    for previous, current_chunk in zip(chunks, chunks[1:]):
        tail = _semantic_overlap_tail(previous, max_chars=overlap)
        if tail and not current_chunk.startswith(tail):
            with_overlap.append(f"{tail}\n{current_chunk}".strip())
        else:
            with_overlap.append(current_chunk)
    return with_overlap


def _assemble_semantic_units(
    units: list[str],
    *,
    max_chars: int,
    keep_oversized_units: bool = False,
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for raw_unit in units:
        unit = raw_unit.strip()
        if not unit:
            continue
        parts = (
            [unit]
            if keep_oversized_units or len(unit) <= max_chars
            else _split_long_unit(unit, chunk_size=max_chars)
        )
        for part in parts:
            candidate = f"{current}\n\n{part}".strip() if current else part
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _faq_units(text: str) -> list[str]:
    marker = re.compile(r"(?im)^(?:Q(?:uestion)?\s*[:：]|问题\s*[:：])")
    starts = [match.start() for match in marker.finditer(text)]
    if not starts:
        return []
    starts.append(len(text))
    pairs: list[str] = []
    for start, end in zip(starts, starts[1:]):
        pair = text[start:end].strip()
        if re.search(r"(?im)^(?:A(?:nswer)?\s*[:：]|答案\s*[:：])", pair):
            pairs.append(pair)
    return pairs


def _markdown_sections(text: str) -> list[tuple[str, dict[str, Any]]]:
    heading_stack: list[str] = []
    sections: list[tuple[str, dict[str, Any]]] = []
    current: list[str] = []
    current_level = 0

    def flush() -> None:
        if not current:
            return
        content = "\n".join(current).strip()
        if content:
            block_types: list[str] = []
            if "```" in content or "~~~" in content:
                block_types.append("code")
            if re.search(r"(?m)^\s*\|.+\|\s*$", content):
                block_types.append("table")
            if re.search(r"(?m)^\s*(?:[-*+] |\d+[.)] )", content):
                block_types.append("list")
            sections.append(
                (
                    content,
                    {
                        "heading_path": list(heading_stack),
                        "heading_level": current_level,
                        "block_types": block_types or ["paragraph"],
                    },
                )
            )
        current.clear()

    in_fence = False
    fence_marker = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            current.append(line)
            continue
        heading = None if in_fence else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack[level - 1 :] = [title]
            current_level = level
            current.append(line)
            continue
        current.append(line)
    flush()
    return sections


def _pdf_sections(text: str) -> list[tuple[str, dict[str, Any]]]:
    page_marker = re.compile(r"(?m)^\[Page\s+(\d+)\]\s*$")
    matches = list(page_marker.finditer(text))
    if not matches:
        return [(unit, {"block_types": ["paragraph"]}) for unit in _split_units(text)]
    sections: list[tuple[str, dict[str, Any]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page = int(match.group(1))
        page_text = text[match.end() : end].strip()
        for unit in re.split(r"\n\s*\n", page_text):
            content = unit.strip()
            if not content:
                continue
            if content.startswith("[Table "):
                block_type = "table"
            elif content.startswith("[OCR "):
                block_type = "image_ocr"
            else:
                block_type = "paragraph"
            sections.append((content, {"page_number": page, "block_types": [block_type]}))
    return sections


def _apply_overlap(chunks: list[DocumentChunk], *, ratio: float) -> list[DocumentChunk]:
    if ratio <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        previous_types = set(previous.metadata.get("block_types") or [])
        current_types = set(current.metadata.get("block_types") or [])
        protected = {"faq", "code", "table", "image_ocr"}
        if previous_types & protected or current_types & protected:
            result.append(current)
            continue
        overlap_chars = max(1, round(len(previous.content) * ratio))
        tail = _semantic_overlap_tail(previous.content, max_chars=overlap_chars)
        if not tail:
            previous_units = _split_units(previous.content)
            if previous_units:
                last_unit = previous_units[-1]
                if len(last_unit) <= max(overlap_chars * 2, 120):
                    tail = last_unit
        if not tail or current.content.startswith(tail):
            result.append(current)
            continue
        metadata = dict(current.metadata)
        metadata["overlap_ratio"] = ratio
        metadata["overlap_chars"] = len(tail)
        result.append(DocumentChunk(content=f"{tail}\n\n{current.content}", metadata=metadata))
    return result


def chunk_document(
    text: str,
    *,
    source_format: str = ".txt",
    max_chars: int = 3000,
    overlap_ratio: float = 0.15,
    source_metadata: dict[str, Any] | None = None,
) -> list[DocumentChunk]:
    """Format-aware semantic chunking used by online document ingestion."""

    normalized = parse_plain_text(text)
    if not normalized:
        return []
    suffix = source_format.lower()
    if suffix and not suffix.startswith("."):
        suffix = Path(suffix).suffix or f".{suffix}"
    common = dict(source_metadata or {})
    common["source_format"] = suffix or ".txt"

    faq_pairs = _faq_units(normalized)
    if faq_pairs:
        chunks = [
            DocumentChunk(
                content=pair,
                metadata={**common, "block_types": ["faq"], "faq_index": index},
            )
            for index, pair in enumerate(faq_pairs, start=1)
        ]
        return chunks

    if suffix == ".md":
        sections = _markdown_sections(normalized)
    elif suffix == ".pdf":
        sections = _pdf_sections(normalized)
        page_details = {
            int(item["page_number"]): item
            for item in common.get("pages", [])
            if isinstance(item, dict) and item.get("page_number") is not None
        }
        for _, metadata in sections:
            page_detail = page_details.get(int(metadata.get("page_number") or 0))
            if page_detail:
                metadata.update(
                    {
                        "ocr_applied": bool(page_detail.get("ocr_applied")),
                        "ocr_confidence": page_detail.get("ocr_confidence"),
                        "needs_review": bool(page_detail.get("needs_review")),
                        "table_count": int(page_detail.get("table_count") or 0),
                        "image_count": int(page_detail.get("image_count") or 0),
                    }
                )
    else:
        sections = [(unit, {"block_types": ["paragraph"]}) for unit in _split_units(normalized)]

    chunks: list[DocumentChunk] = []
    for content, metadata in sections:
        block_types = set(metadata.get("block_types") or [])
        keep_whole = bool(block_types & {"code", "table", "faq", "image_ocr"})
        units = [content] if keep_whole else _split_units(content)
        for piece in _assemble_semantic_units(
            units,
            max_chars=max_chars,
            keep_oversized_units=keep_whole,
        ):
            chunks.append(DocumentChunk(content=piece, metadata={**common, **metadata}))
    return _apply_overlap(chunks, ratio=overlap_ratio)


def enforce_document_chunk_token_limit(
    chunks: list[DocumentChunk], *, max_tokens: int
) -> list[DocumentChunk]:
    limited: list[DocumentChunk] = []
    protected = {"faq", "code", "table", "image_ocr"}
    for chunk in chunks:
        if count_text_tokens(chunk.content) <= max_tokens:
            limited.append(chunk)
            continue
        metadata = dict(chunk.metadata)
        if set(metadata.get("block_types") or []) & protected:
            metadata["needs_review"] = True
            metadata["oversized_semantic_unit"] = True
            limited.append(DocumentChunk(content=chunk.content, metadata=metadata))
            continue
        parts = enforce_chunk_token_limit([chunk.content], max_tokens=max_tokens)
        for part_index, part in enumerate(parts, start=1):
            limited.append(
                DocumentChunk(
                    content=part,
                    metadata={**metadata, "token_split_part": part_index},
                )
            )
    return limited


def _split_long_unit_by_tokens(unit: str, *, max_tokens: int) -> list[str]:
    if count_text_tokens(unit) <= max_tokens:
        return [unit]

    chunks: list[str] = []
    remaining = unit.strip()
    while remaining and count_text_tokens(remaining) > max_tokens:
        piece = truncate_text_by_tokens(remaining, max_tokens)
        if not piece:
            break
        chunks.append(piece)
        remaining = remaining[len(piece) :].strip()
    if remaining:
        chunks.append(remaining)
    return [chunk for chunk in chunks if chunk]


def enforce_chunk_token_limit(chunks: list[str], *, max_tokens: int) -> list[str]:
    """Split oversized chunks before sending them to an embedding model."""

    if max_tokens <= 0:
        return chunks

    limited: list[str] = []
    for chunk in chunks:
        if count_text_tokens(chunk) <= max_tokens:
            limited.append(chunk)
            continue

        current = ""
        token_units: list[str] = []
        for unit in _split_units(chunk):
            token_units.extend(_split_long_unit_by_tokens(unit, max_tokens=max_tokens))

        for unit in token_units:
            candidate = f"{current}\n{unit}".strip() if current else unit
            if count_text_tokens(candidate) <= max_tokens:
                current = candidate
                continue
            if current:
                limited.append(current.strip())
            current = unit
        if current:
            limited.append(current.strip())
    return limited


def stable_chunk_id(doc_id: str, index: int, content: str) -> str:
    h = hashlib.sha256(f"{doc_id}:{index}:{content}".encode()).hexdigest()[:20]
    return f"chk_{doc_id}_{h}"


async def embed_openai(client: AsyncOpenAI, texts: list[str], *, model: str) -> list[list[float]]:
    resp = await client.embeddings.create(model=model, input=texts)
    return [list(d.embedding) for d in resp.data]


class DocumentIngestionPipeline:
    """End-to-end ingestion into Milvus."""

    def __init__(
        self,
        milvus: MilvusVectorClient,
        embedder: AsyncOpenAI,
        *,
        config: IngestionConfig | None = None,
    ) -> None:
        self._milvus = milvus
        self._embedder = embedder
        self._cfg = config or IngestionConfig()

    async def run(
        self,
        doc_id: str,
        raw_text: str,
        *,
        intent: str | None = None,
        source_format: str = ".txt",
    ) -> int:
        text = parse_plain_text(raw_text)
        document_chunks = chunk_document(
            text,
            source_format=source_format,
            max_chars=self._cfg.chunk_size,
            overlap_ratio=0.15,
        )
        document_chunks = enforce_document_chunk_token_limit(
            document_chunks,
            max_tokens=self._cfg.embedding_chunk_max_tokens,
        )
        pieces = [chunk.content for chunk in document_chunks]
        if not document_chunks:
            return 0
        intent_val = intent or self._cfg.default_intent
        embeddings = await embed_openai(self._embedder, pieces, model=self._cfg.embedding_model)
        chunk_ids = [stable_chunk_id(doc_id, i, p) for i, p in enumerate(pieces)]
        await self._milvus.insert_vectors(
            embeddings=embeddings,
            texts=pieces,
            chunk_ids=chunk_ids,
            intents=[intent_val] * len(pieces),
        )
        return len(pieces)
