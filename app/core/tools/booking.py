"""Booking tool — creates reservations for approved itineraries."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BookingType(str, Enum):
    FLIGHT = "flight"
    HOTEL = "hotel"
    TRAIN = "train"


class BookingRequest(BaseModel):
    booking_type: BookingType
    inventory_id: str = Field(..., description="上游搜索结果中的库存标识")
    traveler_name: str
    contact_phone: str
    policy_case_id: str | None = None


class BookingResult(BaseModel):
    booking_id: str
    status: str
    confirmation_code: str
    created_at: datetime


async def create_booking(req: BookingRequest) -> dict[str, Any]:
    bid = str(uuid4())
    code = f"{req.booking_type.value[:2].upper()}-{uuid4().hex[:8].upper()}"
    result = BookingResult(
        booking_id=bid,
        status="confirmed",
        confirmation_code=code,
        created_at=datetime.now(timezone.utc),
    )
    return {
        "ok": True,
        "booking": result.model_dump(mode="json"),
        "inventory_id": req.inventory_id,
    }
