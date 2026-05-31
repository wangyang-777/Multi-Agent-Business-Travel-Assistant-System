from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from app.domain.travel.models import Itinerary, Segment, TravelRequest


def _city_tz(city: str) -> str:
    mapping = {
        "北京": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "深圳": "Asia/Shanghai",
        "广州": "Asia/Shanghai",
        "成都": "Asia/Shanghai",
        "香港": "Asia/Hong_Kong",
        "新加坡": "Asia/Singapore",
        "东京": "Asia/Tokyo",
    }
    return mapping.get(city, "Asia/Shanghai")


def _estimate_segment_cost_cny(
    origin: str,
    destination: str,
    transport: str,
) -> Decimal:
    base = Decimal("1200")
    if origin != destination:
        base += Decimal("800")
    if transport == "train":
        base *= Decimal("0.55")
    elif transport == "hotel":
        base = Decimal("650")
    return base.quantize(Decimal("0.01"))


def build_draft_itinerary(
    request: TravelRequest,
    *,
    default_carrier: str = "示例航司",
) -> Itinerary:
    """
    根据差旅申请生成草稿行程：含 outbound 航段，可选返程与酒店占位。
    生产环境可替换为 GDS/NDC 或供应商 API 结果。
    """
    tz_name = _city_tz(request.destination_city)
    tz = ZoneInfo(tz_name)
    depart_dt = datetime.combine(request.departure_date, time(9, 0), tzinfo=tz)
    flight_duration = timedelta(hours=2 if request.origin_city == request.destination_city else 3)
    arrive_dt = depart_dt + flight_duration

    outbound = Segment(
        segment_id=str(uuid.uuid4()),
        from_city=request.origin_city,
        to_city=request.destination_city,
        depart_local=depart_dt,
        arrive_local=arrive_dt,
        carrier=default_carrier,
        transport="flight",
    )
    segments: list[Segment] = [outbound]

    if request.return_date and request.return_date > request.departure_date:
        ret_tz = ZoneInfo(_city_tz(request.origin_city))
        ret_depart = datetime.combine(request.return_date, time(18, 0), tzinfo=tz)
        ret_arrive = ret_depart + flight_duration
        ret_arrive = ret_arrive.astimezone(ret_tz)
        segments.append(
            Segment(
                segment_id=str(uuid.uuid4()),
                from_city=request.destination_city,
                to_city=request.origin_city,
                depart_local=ret_depart,
                arrive_local=ret_arrive,
                carrier=default_carrier,
                transport="flight",
            )
        )

    if request.return_date:
        hotel_nights = (request.return_date - request.departure_date).days
        if hotel_nights > 0:
            check_in = datetime.combine(request.departure_date, time(15, 0), tzinfo=tz)
            check_out = check_in + timedelta(days=hotel_nights)
            segments.append(
                Segment(
                    segment_id=str(uuid.uuid4()),
                    from_city=request.destination_city,
                    to_city=request.destination_city,
                    depart_local=check_in,
                    arrive_local=check_out,
                    carrier=None,
                    transport="hotel",
                )
            )

    total = sum(
        _estimate_segment_cost_cny(s.from_city, s.to_city, s.transport) for s in segments
    )

    title = f"{request.origin_city} → {request.destination_city} 商务行程"
    return Itinerary(
        itinerary_id=str(uuid.uuid4()),
        request_id=request.request_id,
        title=title,
        segments=segments,
        total_estimated_cny=total,
        created_at=datetime.now(tz),
    )


def merge_segments_by_time(segments: list[Segment]) -> list[Segment]:
    return sorted(segments, key=lambda s: s.depart_local)


def summarize_itinerary_text(itinerary: Itinerary) -> str:
    lines = [f"【{itinerary.title}】", f"预估总额：{itinerary.total_estimated_cny} CNY"]
    for i, s in enumerate(itinerary.segments, start=1):
        mode = "航班" if s.transport == "flight" else "火车" if s.transport == "train" else "酒店"
        lines.append(
            f"{i}. {mode} {s.from_city}→{s.to_city} "
            f"{s.depart_local.isoformat()} — {s.arrive_local.isoformat()}"
        )
    if itinerary.policy_warnings:
        lines.append("策略提示：" + "；".join(itinerary.policy_warnings))
    return "\n".join(lines)
