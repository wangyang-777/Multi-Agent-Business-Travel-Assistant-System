"""Document ingestion: parse → chunk → embed → Milvus."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.infrastructure.vector.milvus_client import MilvusVectorClient
from app.utils.tokenization import count_text_tokens, truncate_text_by_tokens


@dataclass(slots=True)
class IngestionConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_chunk_max_tokens: int = 7000
    embedding_model: str = "text-embedding-3-small"
    default_intent: str = "general"


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

    async def run(self, doc_id: str, raw_text: str, *, intent: str | None = None) -> int:
        text = parse_plain_text(raw_text)
        pieces = chunk_text(text, chunk_size=self._cfg.chunk_size, overlap=self._cfg.chunk_overlap)
        pieces = enforce_chunk_token_limit(pieces, max_tokens=self._cfg.embedding_chunk_max_tokens)
        if not pieces:
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
