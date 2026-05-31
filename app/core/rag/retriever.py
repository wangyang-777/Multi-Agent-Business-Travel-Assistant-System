"""Multi-channel RAG retrieval with parallel search and merged results."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from app.infrastructure.vector.milvus_client import MilvusVectorClient


class KeywordSearchBackend(Protocol):
    async def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]: ...


class IntentIndexBackend(Protocol):
    async def search_by_intent(
        self, query: str, intent: str | None, *, top_k: int
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Retrieved chunk with metadata for scoring and deduplication."""

    chunk_id: str
    text: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        payload = f"{self.chunk_id}|{self.text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:32]


class MultiChannelRetriever:
    """Runs vector, keyword, and intent-directed retrieval in parallel, then merges."""

    def __init__(
        self,
        milvus: MilvusVectorClient,
        keyword_backend: KeywordSearchBackend,
        intent_backend: IntentIndexBackend,
        *,
        vector_weight: float = 1.0,
        keyword_weight: float = 0.85,
        intent_weight: float = 0.95,
    ) -> None:
        self._milvus = milvus
        self._keyword = keyword_backend
        self._intent = intent_backend
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._intent_weight = intent_weight

    async def retrieve(
        self,
        query: str,
        query_embedding: Sequence[float],
        *,
        intent: str | None = None,
        top_k: int = 8,
        per_channel_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = per_channel_k or max(top_k, 4)

        async def vector_channel() -> list[RetrievedChunk]:
            rows = await self._milvus.search_vectors(
                list(query_embedding), top_k=k, intent_filter=intent
            )
            out: list[RetrievedChunk] = []
            for row in rows:
                cid = str(row.get("id", row.get("chunk_id", "")))
                text = str(row.get("text", row.get("content", "")))
                score = float(row.get("score", row.get("distance", 0.0)))
                meta = {x: y for x, y in row.items() if x not in {"text", "content"}}
                out.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        text=text,
                        score=score * self._vector_weight,
                        source="vector",
                        metadata=meta,
                    )
                )
            return out

        async def keyword_channel() -> list[RetrievedChunk]:
            rows = await self._keyword.search(query, top_k=k)
            return [
                RetrievedChunk(
                    chunk_id=str(r.get("chunk_id", r.get("id", ""))),
                    text=str(r.get("text", "")),
                    score=float(r.get("score", 0.0)) * self._keyword_weight,
                    source="keyword",
                    metadata=dict(r.get("metadata", {})),
                )
                for r in rows
            ]

        async def intent_channel() -> list[RetrievedChunk]:
            rows = await self._intent.search_by_intent(query, intent, top_k=k)
            return [
                RetrievedChunk(
                    chunk_id=str(r.get("chunk_id", r.get("id", ""))),
                    text=str(r.get("text", "")),
                    score=float(r.get("score", 0.0)) * self._intent_weight,
                    source="intent",
                    metadata=dict(r.get("metadata", {})),
                )
                for r in rows
            ]

        vec_hits, kw_hits, int_hits = await asyncio.gather(
            vector_channel(), keyword_channel(), intent_channel()
        )
        return self._merge_and_dedup(vec_hits + kw_hits + int_hits, top_k=top_k)

    @staticmethod
    def _content_key(text: str) -> str:
        """Stable key for cross-channel deduplication of identical passage text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _merge_and_dedup(self, chunks: list[RetrievedChunk], *, top_k: int) -> list[RetrievedChunk]:
        best: dict[str, RetrievedChunk] = {}
        for ch in chunks:
            key = self._content_key(ch.text)
            prev = best.get(key)
            if prev is None or ch.score > prev.score:
                best[key] = ch
        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return ranked[:top_k]
