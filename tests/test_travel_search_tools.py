from __future__ import annotations

import json
from datetime import date

import pytest

from app.agent import orchestrator as orchestrator_module
from app.agent.orchestrator import TravelOrchestrator, _resolve_relative_date
from app.core.tools.travel_search import (
    FlyAITravelInventoryProvider,
    FlightSearchRequest,
    HotelSearchRequest,
    TrainSearchRequest,
    TravelRecommendationRequest,
    _build_12306_skill_command,
    _extract_station_code,
    _normalize_12306_skill_results,
    _rank_train_results,
    recommend_travel_options,
    search_trains,
)
from app.domain.schemas import ChatMessage, MessageRole


@pytest.mark.asyncio
async def test_recommend_travel_options_tool_returns_flight_and_hotel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.tools.travel_search.settings.travel_inventory_provider", "demo")
    orch = TravelOrchestrator()

    raw = await orch._execute_tool(
        "recommend_travel_options",
        json.dumps(
            {
                "employee_id": "u1",
                "grade": "staff",
                "origin_city": "北京",
                "destination_city": "上海",
                "departure_date": "2026-05-20",
                "return_date": "2026-05-21",
                "preferred_class": "economy",
                "hotel_budget_cny": 800,
                "include_trains": False,
            },
            ensure_ascii=False,
        ),
    )

    payload = json.loads(raw)
    assert payload["mode"] == "travel_recommendation"
    assert payload["flights"]
    assert payload["hotels"]
    assert payload["recommendation"]["flight"]["flight_no"]


@pytest.mark.asyncio
async def test_recommend_travel_options_includes_train_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.tools.travel_search._today_china", lambda: date(2026, 5, 14))

    async def fake_search_trains(req: TrainSearchRequest) -> dict:
        return {
            "mode": "train",
            "provider": "12306_skill",
            "query": req.model_dump(mode="json"),
            "results": [
                {
                    "trainCode": "G1",
                    "fromStation": "北京南",
                    "toStation": "上海虹桥",
                    "departTime": "06:30",
                    "arriveTime": "11:24",
                    "duration": "04:54",
                    "zy": "有",
                    "ze": "有",
                    "canBuy": "Y",
                }
            ],
            "disclaimer": "以 12306 官方为准",
        }

    monkeypatch.setattr("app.core.tools.travel_search.search_trains", fake_search_trains)

    payload = await recommend_travel_options(
        TravelRecommendationRequest(
            employee_id="u1",
            grade="staff",
            origin_city="北京",
            destination_city="上海",
            departure_date=date(2026, 5, 20),
            return_date=date(2026, 5, 21),
            preferred_class="economy",
            hotel_budget_cny=800,
            hotel_nearby_poi="陆家嘴",
        )
    )

    assert payload["trains"][0]["trainCode"] == "G1"
    assert payload["query"]["hotel_nearby_poi"] == "陆家嘴"
    assert payload["recommendation"]["train"]["trainCode"] == "G1"
    assert payload["providers"]["trains"] == "12306_skill"
    assert payload["policy_checks"]["advance_booking_ok"] is False
    assert "低于提前 7 天" in payload["policy_warnings"][0]


def test_relative_next_monday_resolves_from_current_week() -> None:
    resolved = _resolve_relative_date("下周一从北京去上海", today=date(2026, 5, 14))

    assert resolved == date(2026, 5, 18)


@pytest.mark.asyncio
async def test_tool_date_args_are_normalized_from_relative_user_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_module, "_today", lambda: date(2026, 5, 14))
    monkeypatch.setattr("app.core.tools.travel_search.settings.travel_inventory_provider", "demo")
    orch = TravelOrchestrator()

    raw = await orch._execute_tool(
        "recommend_travel_options",
        json.dumps(
            {
                "employee_id": "u1",
                "grade": "staff",
                "origin_city": "北京",
                "destination_city": "上海",
                "departure_date": "2026-07-14",
                "return_date": "2026-07-14",
                "preferred_class": "economy",
                "include_trains": False,
            },
            ensure_ascii=False,
        ),
        user_text="下周一从北京去上海当天往返，酒店安排在陆家嘴附近",
    )

    payload = json.loads(raw)
    assert payload["query"]["departure_date"] == "2026-05-18"
    assert payload["query"]["return_date"] == "2026-05-18"
    assert payload["query"]["hotel_nearby_poi"] == "陆家嘴"


def test_enterprise_contract_flags_demo_inventory_and_relative_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_module, "_today", lambda: date(2026, 5, 14))
    answer = "好的，以下是7月14日（下周一）北京→上海当天往返差旅方案。"
    fixed = TravelOrchestrator._enforce_enterprise_answer_contract(
        answer,
        [{"output": json.dumps({"provider": "demo"}, ensure_ascii=False)}],
        [ChatMessage(role=MessageRole.USER, content="下周一从北京去上海当天往返")],
    )

    assert "2026年5月18日（周一）" in fixed
    assert "演示数据" in fixed
    assert "非实时库存" in fixed


def test_extract_station_code_from_mcp_text() -> None:
    assert _extract_station_code("北京: BJP\n北京南: VNP", "北京南") == "VNP"
    assert _extract_station_code("上海虹桥 AOH") == "AOH"


@pytest.mark.asyncio
async def test_search_trains_without_mcp_does_not_fabricate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.tools.travel_search.settings.railway_12306_skill_dir", "")
    monkeypatch.setattr("app.core.tools.travel_search.settings.railway_mcp_url", "")

    payload = await search_trains(
        TrainSearchRequest(origin_station="北京", dest_station="上海", depart_date=date(2026, 5, 18))
    )

    assert payload["provider"] == "railway_unconfigured"
    assert payload["results"] == []
    assert "不会生成演示车票" in payload["disclaimer"]


def test_build_12306_skill_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.tools.travel_search.settings.railway_12306_skill_dir", "/opt/12306")
    monkeypatch.setattr("app.core.tools.travel_search.settings.railway_12306_node_bin", "node")

    cmd = _build_12306_skill_command(
        TrainSearchRequest(origin_station="北京", dest_station="上海", depart_date=date(2026, 5, 18))
    )

    assert cmd == [
        "node",
        "/opt/12306/scripts/query.mjs",
        "北京",
        "上海",
        "-d",
        "2026-05-18",
        "--json",
        "-t",
        "GD",
    ]


def test_normalize_12306_skill_results_limits_rows() -> None:
    rows = _normalize_12306_skill_results(
        {"results": [{"trainNo": "G1"}, {"trainNo": "G2"}]},
        limit=1,
    )

    assert len(rows) == 1
    assert rows[0]["trainNo"] == "G1"
    assert "recommendation_score" in rows[0]


def test_rank_train_results_prefers_bookable_staff_friendly_morning_train() -> None:
    rows = _rank_train_results(
        [
            {
                "trainCode": "G99",
                "departTime": "14:00",
                "arriveTime": "19:30",
                "duration": "05:30",
                "ze": "无",
                "zy": "有",
                "canBuy": "Y",
            },
            {
                "trainCode": "G1",
                "departTime": "08:00",
                "arriveTime": "12:32",
                "duration": "04:32",
                "ze": "有",
                "canBuy": "Y",
            },
            {
                "trainCode": "D1",
                "departTime": "09:00",
                "arriveTime": "15:20",
                "duration": "06:20",
                "ze": "有",
                "canBuy": "N",
            },
        ]
    )

    assert rows[0]["trainCode"] == "G1"
    assert rows[0]["recommendation_score"] > rows[1]["recommendation_score"]
    assert "二等座有票" in rows[0]["reason"]


@pytest.mark.asyncio
async def test_flyai_provider_normalizes_flight_and_hotel(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FlyAITravelInventoryProvider()
    captured_hotel_args: list[str] = []

    async def fake_run(args: list[str]) -> dict:
        if args[0] == "search-flight":
            return {
                "status": 0,
                "systemMessage": "flyai hint",
                "data": {
                    "itemList": [
                        {
                            "ticketPrice": "400.0",
                            "jumpUrl": "https://booking.example/flight",
                            "totalDuration": "140分钟",
                            "journeys": [
                                {
                                    "segments": [
                                        {
                                            "marketingTransportName": "国航",
                                            "marketingTransportNo": "CA1883",
                                            "depCityName": "北京",
                                            "arrCityName": "上海",
                                            "depDateTime": "2026-05-18 21:00:00",
                                            "arrDateTime": "2026-05-18 23:20:00",
                                            "seatClassName": "经济舱",
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                },
            }
        captured_hotel_args.extend(args)
        return {
            "status": 0,
            "data": {
                "itemList": [
                    {
                        "name": "上海望湖宾馆",
                        "price": "¥618",
                        "star": "豪华型",
                        "score": "5.0",
                        "interestsPoi": "近陆家嘴",
                        "mainPic": "https://img.example/hotel.jpg",
                        "detailUrl": "https://booking.example/hotel",
                        "review": "商务出行方便",
                    }
                ]
            },
        }

    monkeypatch.setattr(provider, "_run", fake_run)

    flights = await provider.search_flights(
        FlightSearchRequest(origin="北京", destination="上海", depart_date=date(2026, 5, 18))
    )
    hotels = await provider.search_hotels(
        HotelSearchRequest(
            city="上海",
            check_in=date(2026, 5, 18),
            check_out=date(2026, 5, 19),
            poi_name="陆家嘴",
        )
    )

    assert flights[0]["provider"] == "flyai"
    assert flights[0]["flight_no"] == "CA1883"
    assert flights[0]["price_cny"] == "400.0"
    assert flights[0]["booking_url"] == "https://booking.example/flight"
    assert hotels[0]["provider"] == "flyai"
    assert "--poi-name" in captured_hotel_args
    assert "陆家嘴" in captured_hotel_args
    assert hotels[0]["poi_name"] == "陆家嘴"
    assert hotels[0]["nearby"] == "近陆家嘴"
    assert hotels[0]["nightly_cny"] == "618"
    assert hotels[0]["detail_url"] == "https://booking.example/hotel"


@pytest.mark.asyncio
async def test_flyai_provider_keeps_masked_hotel_price(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FlyAITravelInventoryProvider()

    async def fake_run(args: list[str]) -> dict:
        return {
            "status": 0,
            "data": {
                "itemList": [
                    {
                        "name": "上海体验模式酒店",
                        "price": "¥2x",
                        "detailUrl": "https://booking.example/masked-hotel",
                    }
                ]
            },
        }

    monkeypatch.setattr(provider, "_run", fake_run)

    hotels = await provider.search_hotels(
        HotelSearchRequest(city="上海", check_in=date(2026, 5, 18), check_out=date(2026, 5, 19))
    )

    assert hotels[0]["nightly_cny"] == "¥2x"
    assert hotels[0]["total_cny"] == ""
