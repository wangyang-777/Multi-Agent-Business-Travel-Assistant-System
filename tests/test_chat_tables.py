from __future__ import annotations

from app.api.routes.chat import _build_response_tables


def test_build_response_tables_from_tool_trace() -> None:
    raw = {
        "choices": [{"message": {"content": "文本回复仍然保留"}}],
        "metadata": {
            "tool_trace": [
                {
                    "tool": "plan_travel_itinerary",
                    "output": "\n".join(
                        [
                            "【北京 → 上海 商务行程】",
                            "预估总额：2000.00 CNY",
                            "1. 航班 北京→上海 2026-05-12T09:00:00+08:00 — 2026-05-12T12:00:00+08:00",
                            "策略提示：目的地酒店标准为每晚不超过 800 CNY（tier1）。；差补参考：300 CNY/天（以财务制度为准）。",
                        ]
                    ),
                }
            ]
        },
    }

    tables = _build_response_tables(raw)

    assert [table.title for table in tables] == ["行程概览", "行程明细", "差标与策略提示"]
    assert tables[0].rows[0] == {"项目": "行程标题", "内容": "北京 → 上海 商务行程"}
    assert tables[1].rows[0]["类型"] == "航班"
    assert tables[2].rows[0]["序号"] == "1"


def test_build_response_tables_from_travel_recommendation_json() -> None:
    raw = {
        "choices": [{"message": {"content": "文本回复仍然保留"}}],
        "metadata": {
            "tool_trace": [
                {
                    "tool": "recommend_travel_options",
                    "output": """
                    {
                      "mode": "travel_recommendation",
                      "flights": [
                        {
                          "flight_no": "CA1501",
                          "carrier": "示例航司",
                          "origin": "北京",
                          "destination": "上海",
                          "depart_at": "2026-05-20T08:30:00",
                          "arrive_at": "2026-05-20T10:55:00",
                          "cabin": "economy",
                          "price_cny": "1500.00",
                          "reason": "早到达"
                        }
                      ],
                      "trains": [
                        {
                          "trainCode": "G1",
                          "fromStation": "北京南",
                          "toStation": "上海虹桥",
                          "departTime": "06:30",
                          "arriveTime": "11:24",
                          "duration": "04:54",
                          "zy": "有",
                          "ze": "有",
                          "canBuy": "Y"
                        }
                      ],
                      "hotels": [
                        {
                          "name": "上海商务精选酒店",
                          "city": "上海",
                          "star": 4,
                          "check_in": "2026-05-20",
                          "check_out": "2026-05-21",
                          "nightly_cny": "560",
                          "total_cny": "560",
                          "reason": "符合差标"
                        }
                      ],
                      "recommendation": {
                        "flight": {"flight_no": "CA1501"},
                        "train": {"trainCode": "G1"},
                        "hotel": {"name": "上海商务精选酒店"},
                        "reason": "时间和价格更适合"
                      },
                      "disclaimer": "正式预订前确认库存"
                    }
                    """,
                }
            ]
        },
    }

    tables = _build_response_tables(raw)

    assert [table.title for table in tables] == ["推荐组合", "航班候选", "高铁/火车候选", "酒店候选"]
    assert tables[0].rows[0] == {"项目": "推荐航班", "内容": "CA1501"}
    assert tables[1].rows[0]["航班"] == "CA1501"
    assert tables[2].rows[0]["车次"] == "G1"
    assert tables[3].rows[0]["酒店"] == "上海商务精选酒店"


def test_build_response_tables_from_booking_draft_metadata() -> None:
    raw = {
        "choices": [{"message": {"content": "文本回复仍然保留"}}],
        "metadata": {
            "booking_draft": {
                "draft_id": "bd_test",
                "status": "approval_pending",
                "employee_id": "u1",
                "grade": "staff",
                "origin_city": "北京",
                "destination_city": "上海",
                "departure_date": "2026-05-20",
                "recommended_flight": {"flight_no": "CA1501"},
                "recommended_train": {"train_code": "G1"},
                "recommended_hotel": {"name": "上海商务精选酒店"},
                "price_snapshot": {"flight_cny": "1500"},
                "estimated_total_cny": None,
                "policy_checks": {},
                "policy_warnings": ["低于提前 7 天预订要求"],
                "approval_required": True,
                "confirmation_items": ["确认交通方式"],
                "next_action": "submit_for_approval",
                "created_at": "2026-05-18T00:00:00+00:00",
                "expires_at": "2026-05-18T00:30:00+00:00",
            },
            "tool_trace": [],
        },
    }

    tables = _build_response_tables(raw)

    assert [table.title for table in tables] == ["预订草稿", "预订确认项"]
    assert tables[0].rows[0] == {"项目": "草稿 ID", "内容": "bd_test"}
    assert tables[0].rows[8] == {"项目": "是否需要审批", "内容": "是"}
    assert tables[1].rows[0] == {"序号": "1", "确认项": "确认交通方式"}


def test_build_response_tables_from_planner_policy_metadata() -> None:
    raw = {
        "choices": [{"message": {"content": "文本回复仍然保留"}}],
        "metadata": {
            "execution_plan": {
                "goal": "北京到上海出差",
                "planner": "llm",
                "required_tools": ["recommend_travel_options"],
                "missing_slots": [],
                "needs_clarification": False,
                "rationale": "需要查询交通、酒店并校验制度",
                "steps": [
                    {"order": 1, "tool": "recommend_travel_options", "reason": "查询候选并组合推荐"}
                ],
            },
            "policy_constraints": {
                "source": "rag_extracted",
                "confidence": 0.9,
                "constraints": {
                    "hotel_limit_cny": 800,
                    "advance_booking_days": 7,
                    "approval_threshold_cny": 5000,
                },
                "notes": [],
            },
            "policy_validation": {
                "status": "needs_review",
                "summary": "低于提前预订要求",
                "constraint_source": "rag_extracted",
                "constraint_confidence": 0.9,
                "checks": [
                    {"name": "酒店差标", "status": "passed", "detail": "酒店每晚 560 CNY <= 制度上限 800 CNY"},
                    {"name": "提前预订", "status": "warning", "detail": "低于提前 7 天预订要求"},
                ],
                "warnings": ["低于提前 7 天预订要求"],
                "violations": [],
            },
            "tool_trace": [],
        },
    }

    tables = _build_response_tables(raw)

    assert [table.title for table in tables] == [
        "执行计划",
        "执行步骤",
        "制度约束",
        "合规校验",
        "合规检查项",
        "合规风险与提示",
    ]
    assert tables[0].rows[0] == {"项目": "目标", "内容": "北京到上海出差"}
    assert tables[2].rows[2] == {"项目": "酒店标准上限", "内容": "800"}
    assert tables[4].rows[1]["状态"] == "warning"


def test_build_response_tables_from_train_json() -> None:
    raw = {
        "choices": [{"message": {"content": "文本回复仍然保留"}}],
        "metadata": {
            "tool_trace": [
                {
                    "tool": "search_trains",
                    "output": """
                    {
                      "mode": "train",
                      "provider": "12306_skill",
                      "query": {
                        "origin_station": "北京",
                        "dest_station": "上海",
                        "depart_date": "2026-05-18"
                      },
                      "results": [
                        {
                          "trainCode": "G1",
                          "fromStation": "北京南",
                          "toStation": "上海虹桥",
                          "departTime": "06:30",
                          "arriveTime": "11:24",
                          "duration": "04:54",
                          "swz": "无",
                          "zy": "有",
                          "ze": "有",
                          "canBuy": "Y"
                        }
                      ],
                      "disclaimer": "以 12306 官方为准"
                    }
                    """,
                }
            ]
        },
    }

    tables = _build_response_tables(raw)

    assert [table.title for table in tables] == ["12306 查询摘要", "12306 车票候选"]
    assert tables[0].rows[0] == {"项目": "路线", "内容": "北京 → 上海"}
    assert tables[1].rows[0]["车次"] == "G1"
    assert tables[1].rows[0]["状态"] == "可购"


def test_build_response_tables_from_markdown_content_fallback() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "content": "\n".join(
                        [
                            "## 推荐航班方案",
                            "",
                            "| 航班 | 时间 | 价格 |",
                            "|------|------|------|",
                            "| CA1501 | 08:30-10:55 | ¥1500 |",
                            "| MU5102 | 10:15-12:35 | ¥1340 |",
                        ]
                    )
                }
            }
        ],
        "metadata": {"tool_trace": []},
    }

    tables = _build_response_tables(raw)

    assert [table.title for table in tables] == ["推荐航班方案"]
    assert tables[0].columns == ["航班", "时间", "价格"]
    assert tables[0].rows[0] == {"航班": "CA1501", "时间": "08:30-10:55", "价格": "¥1500"}
