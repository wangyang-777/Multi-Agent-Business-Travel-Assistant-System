"""Online hybrid retrieval: keyword + dense recall, RRF and API rerank."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from app.config import settings
from app.core.logging import get_logger
from app.core.rag.reranker import ApiReranker
from app.core.rag.retriever import RetrievedChunk

logger = get_logger(__name__)


class HybridRAGRetriever:
    def __init__(
        self,
        vector_store: Any,
        keyword_index: Any,
        *,
        reranker: ApiReranker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._reranker = reranker

    @property
    def connected(self) -> bool:
        return bool(
            self._vector_store is not None
            and getattr(self._vector_store, "connected", False)
            and self._keyword_index is not None
            and getattr(self._keyword_index, "connected", False)
        )

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        *,
        keyword_top_k: int | None = None,
        vector_top_k: int | None = None,
        rrf_k: int | None = None,
        candidate_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.connected:
            raise RuntimeError("hybrid retrieval requires both vector and keyword indexes")
        keyword_k = keyword_top_k or settings.rag_keyword_top_k
        vector_k = vector_top_k or settings.rag_vector_top_k
        fusion_k = rrf_k or settings.rag_rrf_k
        candidate_k = candidate_top_k or settings.rag_fused_top_k
        final_k = final_top_k or settings.rag_final_top_k

        vector_task = asyncio.to_thread(self._vector_store.search, query_embedding, vector_k)
        keyword_task = self._keyword_index.search(query, top_k=keyword_k)
        vector_hits, keyword_hits = await asyncio.gather(vector_task, keyword_task)
        fused = self._rrf_fuse(vector_hits, keyword_hits, rrf_k=fusion_k, top_k=candidate_k)
        stored_chunks = await self._keyword_index.get_many(
            str(item["chunk_id"]) for item in fused
        )
        for item in fused:
            stored = stored_chunks.get(str(item["chunk_id"]))
            if not stored:
                continue
            item["title"] = stored.get("title") or item.get("title")
            item["doc_type"] = stored.get("doc_type") or item.get("doc_type")
            item["content"] = stored.get("text") or item.get("content")
            item["metadata"].update(dict(stored.get("metadata") or {}))
        fused = [item for item in fused if not item.get("metadata", {}).get("needs_review")]
        if not fused:
            return []

        rerank_status = "disabled"
        ranked = fused[:final_k]
        if self._reranker is not None and settings.rag_reranker_enabled:
            chunks = [
                RetrievedChunk(
                    chunk_id=str(item["chunk_id"]),
                    text=str(item["content"]),
                    score=float(item["rrf_score"]),
                    source="hybrid",
                    metadata={key: value for key, value in item.items() if key != "content"},
                )
                for item in fused
            ]
            try:
                reranked = await self._reranker.rerank_as_chunks(query, chunks, top_k=final_k)
                ranked = []
                for result in reranked:
                    item = dict(result.metadata)
                    item["chunk_id"] = result.chunk_id
                    item["content"] = result.text
                    item["rerank_score"] = result.score
                    item["rerank_status"] = "api"
                    ranked.append(item)
                rerank_status = "api"
            except Exception as exc:  # noqa: BLE001
                rerank_status = "rrf_fallback"
                logger.warning("rag.reranker_unavailable", error=str(exc))

        for item in ranked:
            item.setdefault("rerank_score", None)
            item.setdefault("rerank_status", rerank_status)
        return ranked[:final_k]

    @staticmethod
    def _chunk_key(item: dict[str, Any]) -> str:
        raw_id = item.get("chunk_id") or item.get("id")
        if raw_id:
            return str(raw_id)
        content = str(item.get("content") or item.get("text") or "")
        return "content:" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def _rrf_fuse(
        cls,
        vector_hits: list[dict[str, Any]],
        keyword_hits: list[dict[str, Any]],
        *,
        rrf_k: int,
        top_k: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for channel, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
            for rank, hit in enumerate(hits, start=1):
                key = cls._chunk_key(hit)
                current = fused.setdefault(
                    key,
                    {
                        "chunk_id": key,
                        "title": hit.get("title"),
                        "doc_type": hit.get("doc_type"),
                        "content": str(hit.get("content") or hit.get("text") or ""),
                        "metadata": dict(hit.get("metadata") or {}),
                        "rrf_score": 0.0,
                        "vector_rank": None,
                        "keyword_rank": None,
                        "vector_score": None,
                        "keyword_score": None,
                    },
                )
                current["rrf_score"] += 1.0 / (rrf_k + rank)
                current[f"{channel}_rank"] = rank
                current[f"{channel}_score"] = float(hit.get("score") or 0.0)
                if not current.get("content"):
                    current["content"] = str(hit.get("content") or hit.get("text") or "")
                if not current.get("title"):
                    current["title"] = hit.get("title")
                if not current.get("doc_type"):
                    current["doc_type"] = hit.get("doc_type")
                if hit.get("metadata"):
                    current["metadata"].update(dict(hit["metadata"]))
        ranked = sorted(
            fused.values(),
            key=lambda item: (-float(item["rrf_score"]), str(item["chunk_id"])),
        )
        return ranked[:top_k]
