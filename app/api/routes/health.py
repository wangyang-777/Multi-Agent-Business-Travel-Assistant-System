from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.config import settings
from app.domain.schemas import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health(request: Request) -> HealthStatus:
    checks: dict[str, bool] = {}
    detail_parts: list[str] = []

    r = getattr(request.app.state, "redis", None)
    if r is None:
        checks["redis"] = False
    else:
        try:
            await r.ping()
            checks["redis"] = True
        except Exception as exc:
            checks["redis"] = False
            detail_parts.append(f"redis:{exc!s}")

    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None:
        checks["milvus"] = False
    else:
        checks["milvus"] = milvus.connected

    keyword_index = getattr(request.app.state, "keyword_index", None)
    if keyword_index is None:
        checks["keyword_index"] = False
    else:
        try:
            checks["keyword_index"] = bool(await keyword_index.ping())
        except Exception as exc:
            checks["keyword_index"] = False
            detail_parts.append(f"keyword_index:{exc!s}")

    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        checks["database"] = False
    else:
        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import AsyncEngine

            assert isinstance(engine, AsyncEngine)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["database"] = True
        except Exception as exc:
            checks["database"] = False
            detail_parts.append(f"db:{exc!s}")

    checks["chat_model_configured"] = bool(settings.openai_api_key)
    checks["embedding_model_configured"] = bool(
        settings.embedding_api_key or settings.openai_api_key
    )
    if settings.rag_reranker_enabled:
        checks["rerank_model_configured"] = bool(
            settings.rag_reranker_api_key and settings.rag_reranker_url
        )
    status: Any = "ok" if all(checks.values()) else "degraded"

    return HealthStatus(
        status=status,
        checks=checks,
        detail="; ".join(detail_parts) if detail_parts else None,
    )
