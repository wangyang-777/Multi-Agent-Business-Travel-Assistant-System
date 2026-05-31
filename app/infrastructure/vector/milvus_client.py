"""Milvus client for dense vector search and metadata filters."""

from __future__ import annotations

import asyncio
from typing import Any

from pymilvus import Collection, connections, utility

from app.infrastructure.observability.metrics import MetricsCollector


class MilvusVectorClient:
    """Async-friendly wrapper around pymilvus for embedding search and insert."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 19530,
        alias: str = "default",
        collection_name: str = "travel_docs",
        vector_field: str = "embedding",
        text_field: str = "text",
        id_field: str = "chunk_id",
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._alias = alias
        self._collection_name = collection_name
        self._vector_field = vector_field
        self._text_field = text_field
        self._id_field = id_field
        self._metrics = metrics
        self._collection: Collection | None = None
        self._connected = False

    def _ensure_connection(self) -> None:
        if self._connected:
            return
        connections.connect(alias=self._alias, host=self._host, port=self._port)
        self._connected = True

    def _collection_or_raise(self) -> Collection:
        self._ensure_connection()
        if self._collection is None:
            if not utility.has_collection(self._collection_name, using=self._alias):
                raise RuntimeError(
                    f"Milvus collection '{self._collection_name}' missing; run ETL first."
                )
            col = Collection(self._collection_name, using=self._alias)
            col.load()
            self._collection = col
        return self._collection

    async def search_vectors(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        intent_filter: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        def _search() -> list[dict[str, Any]]:
            col = self._collection_or_raise()
            fields = output_fields or [self._id_field, self._text_field, "intent"]
            params = {"metric_type": "IP", "params": {"nprobe": 16}}
            expr = None
            if intent_filter:
                safe = intent_filter.replace('"', '\\"')
                expr = f'intent == "{safe}"'
            res = col.search(
                data=[query_vector],
                anns_field=self._vector_field,
                param=params,
                limit=top_k,
                expr=expr,
                output_fields=fields,
            )
            rows: list[dict[str, Any]] = []
            for hit in res[0]:
                ent = hit.entity
                row: dict[str, Any] = {
                    "id": ent.get(self._id_field),
                    "text": ent.get(self._text_field, ""),
                    "score": float(hit.score),
                }
                if "intent" in fields:
                    row["intent"] = ent.get("intent")
                rows.append(row)
            return rows

        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(None, _search)
        if self._metrics:
            self._metrics.increment_counter("milvus_search_total")
        return rows

    async def insert_vectors(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        chunk_ids: list[str],
        intents: list[str] | None = None,
    ) -> None:
        if not (len(embeddings) == len(texts) == len(chunk_ids)):
            raise ValueError("embeddings, texts, chunk_ids length mismatch")
        if intents is None:
            intents = [""] * len(texts)

        def _insert() -> None:
            col = self._collection_or_raise()
            col.insert([chunk_ids, texts, intents, embeddings])
            col.flush()

        await asyncio.get_running_loop().run_in_executor(None, _insert)
