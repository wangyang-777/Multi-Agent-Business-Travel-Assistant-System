from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TravelClass(str, Enum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class TripPurpose(str, Enum):
    CLIENT = "client_visit"
    INTERNAL = "internal_meeting"
    TRAINING = "training"
    CONFERENCE = "conference"
    OTHER = "other"


class EmployeeGrade(str, Enum):
    STAFF = "staff"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class TravelRequest(BaseModel):
    request_id: str
    employee_id: str
    grade: EmployeeGrade
    origin_city: str
    destination_city: str
    departure_date: date
    return_date: Optional[date] = None
    purpose: TripPurpose
    passenger_count: int = Field(1, ge=1, le=9)
    preferred_class: Optional[TravelClass] = None
    budget_ceiling_cny: Optional[Decimal] = None
    needs_approval: bool = False
    extra_notes: Optional[str] = None


class Segment(BaseModel):
    segment_id: str
    from_city: str
    to_city: str
    depart_local: datetime
    arrive_local: datetime
    carrier: Optional[str] = None
    transport: Literal["flight", "train", "hotel", "ground"] = "flight"
    booking_ref: Optional[str] = None


class Itinerary(BaseModel):
    itinerary_id: str
    request_id: str
    title: str
    segments: list[Segment] = Field(default_factory=list)
    total_estimated_cny: Decimal = Decimal("0")
    policy_warnings: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class CabinRule(BaseModel):
    max_class: TravelClass
    international_long_haul_allows_business: bool = False


class HotelRule(BaseModel):
    max_nightly_cny: Decimal
    city_tier: str = "tier1"


class TravelPolicy(BaseModel):
    policy_id: str
    company_id: str
    effective_from: date
    cabin_rules: dict[EmployeeGrade, CabinRule] = Field(default_factory=dict)
    hotel_rules: dict[str, HotelRule] = Field(default_factory=dict)
    per_diem_cny: dict[EmployeeGrade, Decimal] = Field(default_factory=dict)
    advance_booking_days_min: int = 7
    requires_pre_approval_above_cny: Decimal = Decimal("5000")


class BookingStatus(str, Enum):
    HELD = "held"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


class Booking(BaseModel):
    booking_id: str
    request_id: str
    itinerary_id: str
    status: BookingStatus
    total_amount_cny: Decimal
    vendor: str
    booked_at: Optional[datetime] = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
