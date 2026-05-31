from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from app.agent.orchestrator import SESSION_KEY_PREFIX
from app.domain.schemas import ChatMessage

router = APIRouter(tags=["sessions"])


def _redis_key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}:{session_id}"


def _session_id_from_key(key: str) -> str:
    prefix = f"{SESSION_KEY_PREFIX}:"
    return key.removeprefix(prefix)


def _parse_messages(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    messages: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            messages.append(ChatMessage.model_validate(item).model_dump(mode="json"))
        except Exception:
            continue
    return messages


@router.get("/sessions")
async def list_sessions(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis 未连接，无法查看会话历史")

    pattern = f"{SESSION_KEY_PREFIX}:*"
    rows: list[dict[str, Any]] = []
    async for key in redis.scan_iter(match=pattern, count=100):
        ttl = await redis.ttl(key)
        raw = await redis.get(key)
        messages = _parse_messages(raw)
        last = messages[-1] if messages else {}
        rows.append(
            {
                "session_id": _session_id_from_key(str(key)),
                "message_count": len(messages),
                "ttl_seconds": ttl,
                "last_role": last.get("role"),
                "last_content": str(last.get("content") or "")[:160],
            }
        )
        if len(rows) >= limit:
            break

    rows.sort(key=lambda item: item["session_id"])
    return {"sessions": rows}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request) -> dict[str, Any]:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis 未连接，无法查看会话历史")

    key = _redis_key(session_id)
    raw = await redis.get(key)
    if not raw:
        raise HTTPException(status_code=404, detail="未找到该 session_id 的历史")
    return {
        "session_id": session_id,
        "ttl_seconds": await redis.ttl(key),
        "messages": _parse_messages(raw),
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict[str, Any]:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis 未连接，无法删除会话历史")
    deleted = await redis.delete(_redis_key(session_id))
    return {"session_id": session_id, "deleted": int(deleted)}
