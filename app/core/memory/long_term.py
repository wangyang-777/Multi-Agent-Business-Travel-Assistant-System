"""Long-term memory backed by vector storage for durable facts and preferences."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from app.infrastructure.vector.milvus_client import MilvusVectorClient

EmbedFn = Callable[[str], Sequence[float]]


@dataclass(slots=True)
class MemoryRecord:
    memory_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class LongTermMemory:
    """Persists and recalls user or org knowledge slices through Milvus."""

    def __init__(
        self,
        milvus: MilvusVectorClient,
        embed: EmbedFn,
        *,
        namespace: str = "default",
    ) -> None:
        self._milvus = milvus
        self._embed = embed
        self._namespace = namespace

    def _namespaced_id(self, raw: str) -> str:
        digest = hashlib.sha256(f"{self._namespace}:{raw}".encode()).hexdigest()[:24]
        return f"ltm_{self._namespace}_{digest}"

    async def remember(self, text: str, *, intent: str = "") -> str:
        vec = list(self._embed(text))
        mid = self._namespaced_id(text)
        await self._milvus.insert_vectors(
            embeddings=[vec],
            texts=[text],
            chunk_ids=[mid],
            intents=[intent],
        )
        return mid

    async def recall(self, query: str, *, top_k: int = 5) -> list[MemoryRecord]:
        qv = list(self._embed(query))
        rows = await self._milvus.search_vectors(qv, top_k=top_k, intent_filter=None)
        out: list[MemoryRecord] = []
        for row in rows:
            mid = str(row.get("id", ""))
            content = str(row.get("text", ""))
            score = float(row.get("score", 0.0))
            meta = {k: v for k, v in row.items() if k not in {"id", "text", "score"}}
            out.append(MemoryRecord(memory_id=mid, text=content, score=score, metadata=meta))
        return out
