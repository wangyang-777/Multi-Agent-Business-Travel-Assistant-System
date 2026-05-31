"""Async Redis client for caching and ephemeral state."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis


class RedisCache:
    """Async Redis wrapper with JSON helpers and connection lifecycle."""

    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        encoding: str = "utf-8",
    ) -> None:
        self._client = redis.from_url(url, encoding=encoding, decode_responses=True)

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        await self._client.set(key, value, ex=ex)

    async def delete(self, key: str) -> int:
        return int(await self._client.delete(key))

    async def get_json(self, key: str) -> Any | None:
        raw = await self.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_json(self, key: str, obj: Any, *, ex: int | None = None) -> None:
        await self.set(key, json.dumps(obj, ensure_ascii=False), ex=ex)

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
