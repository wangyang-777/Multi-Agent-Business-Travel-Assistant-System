from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.domain.schemas import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health(request: Request) -> HealthStatus:
    checks: dict[str, bool] = {}
    detail_parts: list[str] = []

    r = getattr(request.app.state, "redis", None)
    if r is None:
        checks["redis"] = True
    else:
        try:
            await r.ping()
            checks["redis"] = True
        except Exception as exc:
            checks["redis"] = False
            detail_parts.append(f"redis:{exc!s}")

    milvus = getattr(request.app.state, "milvus", None)
    if milvus is None:
        checks["milvus"] = True
    else:
        checks["milvus"] = milvus.connected

    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        checks["database"] = True
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

    status: Any = "ok"
    if not checks.get("redis", True) or not checks.get("database", True):
        status = "degraded"
    elif milvus is not None and not checks.get("milvus", True):
        status = "degraded"

    return HealthStatus(
        status=status,
        checks=checks,
        detail="; ".join(detail_parts) if detail_parts else None,
    )
