from __future__ import annotations

import json

import httpx
import pytest

from app.core.rag.reranker import ApiReranker
from app.core.rag.retriever import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk("a", "航班信息", 0.4, "hybrid", {"title": "航班"}),
        RetrievedChunk("b", "上海酒店标准为每晚800元", 0.3, "hybrid", {"title": "酒店"}),
    ]


@pytest.mark.asyncio
async def test_api_reranker_maps_indices_and_preserves_metadata() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/services/rerank/text-rerank/text-rerank"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload == {
            "model": "qwen3.7-text-rerank",
            "input": {"query": "上海住宿标准", "documents": ["航班信息", "上海酒店标准为每晚800元"]},
            "parameters": {"top_n": 2},
        }
        return httpx.Response(
            200,
            json={"output": {"results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.04},
            ]}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        reranker = ApiReranker(
            api_key="test-key",
            url="https://example.com/api/v1/services/rerank/text-rerank/text-rerank",
            client=client,
        )
        results = await reranker.rerank_as_chunks("上海住宿标准", _chunks(), top_k=2)

    assert [chunk.chunk_id for chunk in results] == ["b", "a"]
    assert [chunk.score for chunk in results] == [0.91, 0.04]
    assert results[0].metadata == {"title": "酒店"}


@pytest.mark.asyncio
async def test_api_reranker_rejects_missing_key_and_incomplete_response() -> None:
    reranker = ApiReranker(api_key="", url="https://example.com/rerank")
    with pytest.raises(RuntimeError, match="RAG_RERANKER_API_KEY"):
        await reranker.rerank_as_chunks("上海", _chunks())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"output": {"results": []}}))
    ) as client:
        reranker = ApiReranker(api_key="test-key", url="https://example.com/rerank", client=client)
        with pytest.raises(ValueError, match="incomplete result list"):
            await reranker.rerank_as_chunks("上海", _chunks())
