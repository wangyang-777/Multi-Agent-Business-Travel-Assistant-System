from __future__ import annotations

from app.core.rag.hybrid import HybridRAGRetriever
from app.core.rag.retriever import RetrievedChunk
from app.services.keyword_index import RedisKeywordIndex


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, *, mapping: dict[str, str]) -> None:
        self.hashes.setdefault(key, {}).update(mapping)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def hdel(self, key: str, *fields: str) -> int:
        target = self.hashes.setdefault(key, {})
        deleted = 0
        for field in fields:
            deleted += int(target.pop(field, None) is not None)
        return deleted

    async def hlen(self, key: str) -> int:
        return len(self.hashes.get(key, {}))


class _VectorStore:
    connected = True

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        return [
            {"id": "vector-only", "content": "差旅需要事前申请", "score": 0.91},
            {"id": "shared", "content": "上海酒店标准为 800 CNY", "score": 0.82},
        ][:top_k]


class _StubReranker:
    def rerank_as_chunks(
        self, query: str, chunks: list[RetrievedChunk], *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        ranked = sorted(chunks, key=lambda chunk: chunk.chunk_id == "shared", reverse=True)
        out = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=1.0 - index * 0.1,
                source=chunk.source,
                metadata={**chunk.metadata, "cross_encoder_score": 1.0 - index * 0.1},
            )
            for index, chunk in enumerate(ranked)
        ]
        return out if top_k is None else out[:top_k]


async def test_keyword_index_and_hybrid_rrf_rerank() -> None:
    keyword = RedisKeywordIndex(_FakeRedis(), key="test:index")
    await keyword.upsert_many(
        [
            {
                "chunk_id": "shared",
                "text": "staff 员工在上海住宿的酒店标准为 800 CNY",
                "title": "酒店制度",
                "doc_type": "policy",
            },
            {
                "chunk_id": "keyword-only",
                "text": "上海出差报销需要提供住宿发票",
                "title": "报销制度",
                "doc_type": "policy",
            },
        ]
    )

    retriever = HybridRAGRetriever(
        _VectorStore(),
        keyword,
        reranker=_StubReranker(),  # type: ignore[arg-type]
    )
    results = await retriever.retrieve(
        "staff 上海酒店标准",
        [0.1],
        keyword_top_k=20,
        vector_top_k=20,
        rrf_k=60,
        candidate_top_k=20,
        final_top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "shared"
    assert results[0]["vector_rank"] == 2
    assert results[0]["keyword_rank"] == 1
    assert results[0]["rerank_status"] == "cross_encoder"
    assert results[0]["rerank_score"] == 1.0


async def test_keyword_delete_by_parent_document() -> None:
    redis = _FakeRedis()
    keyword = RedisKeywordIndex(redis, key="test:index")
    await keyword.upsert_many(
        [
            {"chunk_id": "c1", "parent_doc_id": "doc-1", "text": "第一段"},
            {"chunk_id": "c2", "parent_doc_id": "doc-1", "text": "第二段"},
        ]
    )

    assert await keyword.delete(doc_id="doc-1") == 2
    assert await keyword.search("第一段", top_k=5) == []
