"""Rerank retrieved text with the Alibaba Cloud Model Studio API."""

from __future__ import annotations

import math
from typing import Any

import httpx

from app.core.rag.retriever import RetrievedChunk
from app.utils.openai_client import create_async_http_client


class ApiReranker:
    """Call the native DashScope text-rerank endpoint and retain chunk metadata."""

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        model_name: str = "qwen3.7-text-rerank",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._url = url
        self._model_name = model_name
        self._timeout = timeout
        self._client = client

    async def rerank_as_chunks(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if not self._api_key:
            raise RuntimeError("RAG_RERANKER_API_KEY is not configured")
        if not self._url:
            raise RuntimeError("RAG_RERANKER_URL is not configured")

        limit = min(top_k or len(chunks), len(chunks))
        payload = {
            "model": self._model_name,
            "input": {"query": query, "documents": [chunk.text for chunk in chunks]},
            "parameters": {"top_n": limit},
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._client is None:
            async with create_async_http_client(url=self._url, timeout=self._timeout) as client:
                response = await client.post(self._url, headers=headers, json=payload)
        else:
            response = await self._client.post(self._url, headers=headers, json=payload)

        if response.is_error:
            try:
                error_code = response.json().get("code", "unknown")
            except ValueError:
                error_code = "unknown"
            raise RuntimeError(f"Rerank API HTTP {response.status_code}: {error_code}")

        body: Any = response.json()
        results = body.get("output", {}).get("results") if isinstance(body, dict) else None
        if not isinstance(results, list) or len(results) != limit:
            raise ValueError("Rerank API returned an incomplete result list")

        ranked: list[RetrievedChunk] = []
        seen: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("Rerank API returned an invalid result")
            index = result.get("index")
            score = result.get("relevance_score")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(chunks)
                or index in seen
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise ValueError("Rerank API returned an invalid index or score")
            seen.add(index)
            chunk = chunks[index]
            ranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    source=chunk.source,
                    metadata=dict(chunk.metadata),
                )
            )
        return ranked
