from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from app.domain.travel.models import (
    CabinRule,
    EmployeeGrade,
    HotelRule,
    Itinerary,
    TravelClass,
    TravelPolicy,
    TravelRequest,
)


_CLASS_ORDER: list[TravelClass] = [
    TravelClass.ECONOMY,
    TravelClass.PREMIUM_ECONOMY,
    TravelClass.BUSINESS,
    TravelClass.FIRST,
]


def _class_rank(c: TravelClass) -> int:
    return _CLASS_ORDER.index(c)


def default_corporate_policy(company_id: str = "default") -> TravelPolicy:
    return TravelPolicy(
        policy_id=f"POL-{company_id}",
        company_id=company_id,
        effective_from=date.today(),
        cabin_rules={
            EmployeeGrade.STAFF: CabinRule(max_class=TravelClass.ECONOMY),
            EmployeeGrade.MANAGER: CabinRule(max_class=TravelClass.PREMIUM_ECONOMY),
            EmployeeGrade.DIRECTOR: CabinRule(
                max_class=TravelClass.BUSINESS,
                international_long_haul_allows_business=True,
            ),
            EmployeeGrade.EXECUTIVE: CabinRule(
                max_class=TravelClass.BUSINESS,
                international_long_haul_allows_business=True,
            ),
        },
        hotel_rules={
            "tier1": HotelRule(max_nightly_cny=Decimal("800"), city_tier="tier1"),
            "tier2": HotelRule(max_nightly_cny=Decimal("500"), city_tier="tier2"),
        },
        per_diem_cny={
            EmployeeGrade.STAFF: Decimal("300"),
            EmployeeGrade.MANAGER: Decimal("450"),
            EmployeeGrade.DIRECTOR: Decimal("600"),
            EmployeeGrade.EXECUTIVE: Decimal("800"),
        },
        advance_booking_days_min=7,
        requires_pre_approval_above_cny=Decimal("5000"),
    )


def resolve_city_tier(destination_city: str) -> str:
    tier1 = {"北京", "上海", "深圳", "广州", "杭州"}
    return "tier1" if destination_city in tier1 else "tier2"


def evaluate_cabin_compliance(
    policy: TravelPolicy,
    request: TravelRequest,
    preferred: Optional[TravelClass],
) -> list[str]:
    warnings: list[str] = []
    rule = policy.cabin_rules.get(request.grade)
    if not rule:
        return ["未配置该职级的舱位规则，请人工复核。"]
    chosen = preferred or TravelClass.ECONOMY
    if _class_rank(chosen) > _class_rank(rule.max_class):
        warnings.append(
            f"所选舱位 {chosen.value} 超出职级允许上限 {rule.max_class.value}，需特批或降级。"
        )
    return warnings


def evaluate_budget_and_approval(
    policy: TravelPolicy,
    request: TravelRequest,
    itinerary: Itinerary,
) -> list[str]:
    warnings: list[str] = []
    if request.budget_ceiling_cny and itinerary.total_estimated_cny > request.budget_ceiling_cny:
        warnings.append(
            f"行程预估 {itinerary.total_estimated_cny} 超过个人填报上限 {request.budget_ceiling_cny}。"
        )
    if itinerary.total_estimated_cny > policy.requires_pre_approval_above_cny:
        warnings.append(
            f"金额超过公司事前审批线 {policy.requires_pre_approval_above_cny}，需提交 OA 审批。"
        )
    delta = (request.departure_date - date.today()).days
    if delta < policy.advance_booking_days_min:
        warnings.append(
            f"距出发仅 {delta} 天，低于提前 {policy.advance_booking_days_min} 天预订要求，可能产生附加费。"
        )
    return warnings


def apply_policy_to_itinerary(
    policy: TravelPolicy,
    request: TravelRequest,
    itinerary: Itinerary,
    *,
    preferred_class: Optional[TravelClass] = None,
) -> Itinerary:
    merged = list(itinerary.policy_warnings)
    merged.extend(evaluate_cabin_compliance(policy, request, preferred_class))
    merged.extend(evaluate_budget_and_approval(policy, request, itinerary))
    tier = resolve_city_tier(request.destination_city)
    hotel_rule = policy.hotel_rules.get(tier)
    if hotel_rule:
        merged.append(
            f"目的地酒店标准为每晚不超过 {hotel_rule.max_nightly_cny} CNY（{tier}）。"
        )
    per_diem = policy.per_diem_cny.get(request.grade)
    if per_diem:
        merged.append(f"差补参考：{per_diem} CNY/天（以财务制度为准）。")
    return itinerary.model_copy(update={"policy_warnings": merged})
