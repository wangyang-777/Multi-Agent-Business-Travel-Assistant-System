from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.langgraph_orchestrator import LangGraphTravelOrchestrator
from app.domain.schemas import ChatMessage, MessageRole


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value


class _FakeLLM:
    model = "fake-model"
    calls = 0
    messages: list[list[dict]] = []

    async def chat_completion(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.messages.append(messages)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="图编排回复",
                        tool_calls=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


class _FakeStore:
    connected = True

    def __init__(self, hits):
        self._hits = hits

    def search(self, vector, top_k=5):  # type: ignore[no-untyped-def]
        return self._hits


class _FakeEmbedder:
    async def embed_text(self, text: str) -> list[float]:
        return [0.1] * 1536


@pytest.mark.asyncio
async def test_langgraph_orchestrator_keeps_chat_response_shape() -> None:
    orch = LangGraphTravelOrchestrator(llm=_FakeLLM())

    result = await orch.run_completion(
        [ChatMessage(role=MessageRole.USER, content="你好")],
        session_id=None,
    )

    assert result["model"] == "fake-model"
    assert result["choices"][0]["message"]["content"] == "图编排回复"
    assert result["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert result["metadata"]["orchestrator"] == "langgraph"
    assert result["metadata"]["trace"]


@pytest.mark.asyncio
async def test_langgraph_guardrail_blocks_risky_booking_before_llm() -> None:
    llm = _FakeLLM()
    orch = LangGraphTravelOrchestrator(llm=llm)

    result = await orch.run_completion(
        [ChatMessage(role=MessageRole.USER, content="帮我直接预订机票")],
        session_id=None,
    )

    assert "信息不足" in result["choices"][0]["message"]["content"]
    assert llm.calls == 0
    assert result["metadata"]["risk_level"] == "medium"


def test_approval_form_detects_policy_risk_from_tool_trace() -> None:
    state = {
        "tool_trace": [
            {
                "arguments": (
                    '{"employee_id":"u1","grade":"staff","origin_city":"北京",'
                    '"destination_city":"上海","departure_date":"2026-05-12",'
                    '"purpose":"client_visit"}'
                ),
                "output": "\n".join(
                    [
                        "【北京 → 上海 商务行程】",
                        "预估总额：6000.00 CNY",
                        "策略提示：金额超过公司事前审批线 5000，需提交 OA 审批。",
                    ]
                ),
            }
        ]
    }

    form = LangGraphTravelOrchestrator._build_approval_form(state)  # type: ignore[arg-type]

    assert form is not None
    assert form["required"] is True
    assert form["status"] == "pending_human_approval"
    assert form["grade"] == "staff"


def test_booking_draft_created_from_travel_recommendation_trace() -> None:
    state = {
        "tool_trace": [
            {
                "arguments": (
                    '{"employee_id":"u1","grade":"staff","origin_city":"北京",'
                    '"destination_city":"上海","departure_date":"2026-05-20",'
                    '"purpose":"client_visit"}'
                ),
                "output": """
                {
                  "mode": "travel_recommendation",
                  "query": {
                    "employee_id": "u1",
                    "grade": "staff",
                    "origin_city": "北京",
                    "destination_city": "上海",
                    "departure_date": "2026-05-20",
                    "purpose": "client_visit"
                  },
                  "policy_warnings": ["低于提前 7 天预订要求"],
                  "policy_checks": {"advance_booking_ok": false},
                  "recommendation": {
                    "flight": {"flight_no": "CA1501", "price_cny": "1500.00"},
                    "train": {"trainCode": "G1", "ze": "有"},
                    "hotel": {"name": "上海商务精选酒店", "total_cny": "560"}
                  }
                }
                """,
            }
        ]
    }

    draft = LangGraphTravelOrchestrator._build_booking_draft(state)  # type: ignore[arg-type]
    form = LangGraphTravelOrchestrator._build_approval_form(state, draft)  # type: ignore[arg-type]

    assert draft is not None
    assert draft["draft_id"].startswith("bd_")
    assert draft["status"] == "approval_pending"
    assert draft["recommended_flight"]["flight_no"] == "CA1501"
    assert draft["recommended_train"]["train_code"] == "G1"
    assert draft["recommended_hotel"]["name"] == "上海商务精选酒店"
    assert draft["next_action"] == "submit_for_approval"
    assert form is not None
    assert form["required"] is True


def test_policy_validation_checks_booking_draft_against_constraints() -> None:
    state = {
        "policy_constraints": {
            "source": "rag_extracted",
            "confidence": 0.9,
            "constraints": {
                "hotel_limit_cny": 800,
                "approval_threshold_cny": 5000,
                "cabin_limit": "economy",
            },
        },
        "tool_trace": [],
    }
    draft = {
        "recommended_hotel": {"nightly_cny": "960"},
        "recommended_flight": {"cabin": "business"},
        "recommended_train": {},
        "estimated_total_cny": "6200",
        "policy_warnings": [],
    }

    validation = LangGraphTravelOrchestrator._build_policy_validation(state, draft)  # type: ignore[arg-type]

    assert validation["status"] == "failed"
    assert any(item["name"] == "酒店差标" for item in validation["checks"])
    assert any("超过制度上限" in item for item in validation["violations"])
    assert any("需提交审批" in item for item in validation["warnings"])


def test_heuristic_policy_constraints_extracts_common_rules() -> None:
    result = LangGraphTravelOrchestrator._heuristic_policy_constraints(
        [
            {
                "content": (
                    "staff 员工出差默认只能选择 economy 经济舱。"
                    "上海酒店标准为每晚不超过 800 CNY。员工应至少提前 7 天预订。"
                    "单次差旅行程预估金额超过 5000 CNY 时，需要提交审批。"
                )
            }
        ]
    )

    assert result["constraints"]["hotel_limit_cny"] == 800
    assert result["constraints"]["advance_booking_days"] == 7
    assert result["constraints"]["approval_threshold_cny"] == 5000
    assert result["constraints"]["cabin_limit"] == "economy"


def test_reliable_citation_detection() -> None:
    assert LangGraphTravelOrchestrator._has_reliable_citations(
        "staff 去上海酒店标准是多少？",
        [{"title": "policy-hotel-tier1", "content": "staff 员工去上海酒店标准为 800 CNY"}],
    )
    assert not LangGraphTravelOrchestrator._has_reliable_citations(
        "高级员工去济南高铁标准是什么？",
        [{"title": "policy-hotel-tier1", "content": "staff 员工去上海酒店标准为 800 CNY"}],
    )


def test_verification_flags_unsupported_rag_terms() -> None:
    result = LangGraphTravelOrchestrator._verify_answer_grounding(
        "staff 去上海酒店标准为 900 CNY，需要审批。",
        [{"title": "policy-hotel-tier1", "content": "staff 去上海酒店标准为 800 CNY。"}],
        "rag_grounded",
    )

    assert result["passed"] is False
    assert "900 CNY" in result["unsupported_terms"]


def test_verification_skips_llm_fallback() -> None:
    result = LangGraphTravelOrchestrator._verify_answer_grounding(
        "未检索到公司制度依据，以下为通用建议。",
        [],
        "llm_fallback",
    )

    assert result["passed"] is True
    assert result["reason"] == "non_rag_answer"


@pytest.mark.asyncio
async def test_memory_fusion_persists_long_term_user_facts() -> None:
    redis = _FakeRedis()
    llm = _FakeLLM()
    orch = LangGraphTravelOrchestrator(llm=llm, redis_client=redis)

    await orch.run_completion(
        [ChatMessage(role=MessageRole.USER, content="我的职位是staff，我喜欢靠近客户办公室的酒店")],
        user_id="u1",
    )

    assert "memory:long:u1" in redis.store
    assert "staff" in redis.store["memory:long:u1"]
    first_call_messages = llm.messages[0]
    memory_prompts = [m["content"] for m in first_call_messages if m["role"] == "system"]
    assert any("公司制度/RAG 引用 > 当前用户明确输入" in text for text in memory_prompts)
