"""Unit tests for multi-channel RAG retriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.rag.retriever import MultiChannelRetriever, RetrievedChunk
from app.infrastructure.vector.milvus_client import MilvusVectorClient


@pytest.mark.asyncio
async def test_retriever_merges_and_dedupes() -> None:
    mc = MagicMock(spec=MilvusVectorClient)
    mc.search_vectors = AsyncMock(
        return_value=[{"id": "a", "text": "dup text", "score": 0.9}]
    )

    class KwBackend:
        async def search(self, q: str, *, top_k: int) -> list[dict]:
            return [{"chunk_id": "b", "text": "dup text", "score": 0.4}]

    class IntentBackend:
        async def search_by_intent(
            self, q: str, intent: str | None, *, top_k: int
        ) -> list[dict]:
            return [{"chunk_id": "c", "text": "unique", "score": 0.7}]

    r = MultiChannelRetriever(
        milvus=mc,
        keyword_backend=KwBackend(),
        intent_backend=IntentBackend(),
    )
    out = await r.retrieve("query", [0.1, 0.2], intent=None, top_k=5)
    texts = {c.text for c in out}
    assert "dup text" in texts
    assert "unique" in texts
    dup_chunks = [c for c in out if c.text == "dup text"]
    assert len(dup_chunks) == 1
    assert dup_chunks[0].score == max(c.score for c in out if c.text == "dup text")


@pytest.mark.asyncio
async def test_retrieved_chunk_fingerprint_stable() -> None:
    a = RetrievedChunk(
        chunk_id="1",
        text="hello",
        score=1.0,
        source="vector",
    )
    b = RetrievedChunk(
        chunk_id="1",
        text="hello",
        score=0.5,
        source="keyword",
    )
    assert a.fingerprint == b.fingerprint
