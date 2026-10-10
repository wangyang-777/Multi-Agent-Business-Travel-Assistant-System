from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api.routes.health import health


@pytest.mark.asyncio
async def test_health_marks_missing_dependencies_unavailable() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    result = await health(request)  # type: ignore[arg-type]

    assert result.status == "degraded"
    assert result.checks["redis"] is False
    assert result.checks["database"] is False
    assert result.checks["milvus"] is False
    assert result.checks["keyword_index"] is False
