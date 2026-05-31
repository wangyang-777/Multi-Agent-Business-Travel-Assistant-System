from __future__ import annotations

import pytest

from app.api.routes.mcp import _dispatch


@pytest.mark.asyncio
async def test_mcp_tools_list() -> None:
    data = await _dispatch("tools/list", {}, None)  # type: ignore[arg-type]
    names = [tool["name"] for tool in data["tools"]]
    assert "plan_travel_itinerary" in names
    assert "check_travel_policy" in names
    assert "search_flights" in names
    assert "search_hotels" in names
    assert "search_trains" in names
    assert "recommend_travel_options" in names
