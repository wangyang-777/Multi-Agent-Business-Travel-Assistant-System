"""Redis-backed keyword index with deterministic BM25 ranking."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from app.config import settings


def tokenize_for_keyword_search(text: str) -> list[str]:
    """Tokenize mixed Chinese/ASCII text without requiring a segmentation service."""

    normalized = text.lower()
    ascii_tokens = re.findall(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese_tokens: list[str] = []
    for run in chinese_runs:
        chinese_tokens.extend(run)
        chinese_tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return ascii_tokens + chinese_tokens


class RedisKeywordIndex:
    """Stores complete chunks in a Redis hash and ranks them with BM25 at query time."""

    def __init__(self, redis_client: Any | None, *, key: str | None = None) -> None:
        self._redis = redis_client
        self._key = key or settings.keyword_index_redis_key

    @property
    def connected(self) -> bool:
        return self._redis is not None

    async def ping(self) -> bool:
        if self._redis is None:
            return False
        await self._redis.hlen(self._key)
        return True

    async def upsert_many(self, documents: Iterable[dict[str, Any]]) -> int:
        if self._redis is None:
            raise RuntimeError("keyword index unavailable")
        mapping: dict[str, str] = {}
        for document in documents:
            chunk_id = str(document.get("chunk_id") or document.get("id") or "").strip()
            text = str(document.get("text") or document.get("content") or "").strip()
            if not chunk_id or not text:
                continue
            payload = dict(document)
            payload["chunk_id"] = chunk_id
            payload["text"] = text
            mapping[chunk_id] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if mapping:
            await self._redis.hset(self._key, mapping=mapping)
        return len(mapping)

    async def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if self._redis is None:
            raise RuntimeError("keyword index unavailable")
        raw_documents = await self._redis.hgetall(self._key)
        documents: list[dict[str, Any]] = []
        for raw in raw_documents.values():
            try:
                item = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(item, dict) and item.get("text"):
                documents.append(item)
        if not documents:
            return []

        query_terms = tokenize_for_keyword_search(query)
        if not query_terms:
            return []
        tokenized = [tokenize_for_keyword_search(str(item["text"])) for item in documents]
        avg_len = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
        doc_freq = Counter(term for terms in tokenized for term in set(terms))
        n_docs = len(documents)
        k1 = 1.5
        b = 0.75
        ranked: list[dict[str, Any]] = []
        for item, terms in zip(documents, tokenized, strict=True):
            frequencies = Counter(terms)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5)
                )
                norm = frequency + k1 * (1 - b + b * len(terms) / max(avg_len, 1.0))
                score += inverse_frequency * frequency * (k1 + 1) / norm
            if score <= 0:
                continue
            ranked.append(
                {
                    "chunk_id": str(item.get("chunk_id") or ""),
                    "text": str(item.get("text") or ""),
                    "title": item.get("title"),
                    "doc_type": item.get("doc_type"),
                    "score": score,
                    "metadata": dict(item.get("metadata") or {}),
                    "parent_doc_id": item.get("parent_doc_id"),
                }
            )
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["chunk_id"])))
        return ranked[:top_k]

    async def get_many(self, chunk_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        if self._redis is None:
            raise RuntimeError("keyword index unavailable")
        wanted = {str(chunk_id) for chunk_id in chunk_ids if chunk_id}
        if not wanted:
            return {}
        raw_documents = await self._redis.hgetall(self._key)
        found: dict[str, dict[str, Any]] = {}
        for chunk_id in wanted:
            raw = raw_documents.get(chunk_id)
            if raw is None:
                continue
            try:
                item = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(item, dict):
                found[chunk_id] = item
        return found

    async def delete(self, *, doc_id: str | None = None, title: str | None = None) -> int:
        if self._redis is None:
            raise RuntimeError("keyword index unavailable")
        if not doc_id and not title:
            raise ValueError("doc_id or title is required")
        raw_documents = await self._redis.hgetall(self._key)
        matches: list[str] = []
        for field, raw in raw_documents.items():
            try:
                item = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or field)
            parent_doc_id = str(item.get("parent_doc_id") or "")
            if doc_id and (chunk_id == doc_id or parent_doc_id == doc_id):
                matches.append(str(field))
            elif title and str(item.get("title") or "") == title:
                matches.append(str(field))
        if matches:
            await self._redis.hdel(self._key, *matches)
        return len(matches)
