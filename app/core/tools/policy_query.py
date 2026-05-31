"""Travel policy / 差标 query tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PolicyQuery(BaseModel):
    employee_level: str = Field(..., description="职级或员工组，如 M4 / staff")
    city_tier: str | None = Field(None, description="目的地城市级别：一线/新一线/其他")
    trip_type: str = Field("domestic", description="domestic | international")


def _tier_rules(city_tier: str | None) -> dict[str, Any]:
    tier = city_tier or "其他"
    hotel_cap = {"一线": 800, "新一线": 600}.get(tier, 450)
    return {
        "hotel_nightly_cap_cny": hotel_cap,
        "flight_cabin": "economy",
        "train_seat": "二等座",
    }


async def query_travel_policy(q: PolicyQuery) -> dict[str, Any]:
    rules = _tier_rules(q.city_tier)
    per_diem = 180 if q.trip_type == "domestic" else 45
    return {
        "employee_level": q.employee_level,
        "city_tier": q.city_tier,
        "trip_type": q.trip_type,
        "rules": rules,
        "meals_per_diem_cny": per_diem,
        "notes": "超标需事前审批；国际行程以财务最新通告为准。",
    }


async def policy_query(**kwargs: Any) -> dict[str, Any]:
    q = PolicyQuery.model_validate(kwargs)
    return await query_travel_policy(q)
