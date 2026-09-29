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
        content = (
            '{"primary_intent":"general","tasks":[{"id":"task_1",'
            '"intent":"general","request":"你好","slots":{},'
            '"depends_on":[],"missing_slots":[]}],"clarification_question":null}'
            if kwargs.get("response_format") else "图编排回复"
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content,
                        tool_calls=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


class _RetryFlowLLM:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            content = (
                '{"primary_intent":"trip_planning","tasks":[{"id":"task_1",'
                '"intent":"trip_planning",'
                '"request":"我是staff，请推荐2026-10-20从北京到上海的航班、高铁和酒店",'
                '"slots":{"employee_id":"u1","grade":"staff",'
                '"origin_city":"北京","destination_city":"上海",'
                '"departure_date":"2026-10-20"},'
                '"depends_on":[],"missing_slots":[]}],"clarification_question":null}'
            )
            tool_calls = None
        elif self.calls in {2, 4}:
            content = None
            tool_calls = [
                SimpleNamespace(
                    id=f"tc-{self.calls}",
                    function=SimpleNamespace(
                        name="recommend_travel_options",
                        arguments=(
                            '{"employee_id":"u1","grade":"staff",'
                            '"origin_city":"北京","destination_city":"上海",'
                            '"departure_date":"2026-10-20"}'
                        ),
                    ),
                )
            ]
        else:
            content = "当前没有找到满足条件的完整候选。"
            tool_calls = None
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=tool_calls)
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
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
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


@pytest.mark.asyncio
async def test_validation_failure_schedules_one_travel_retry() -> None:
    orch = LangGraphTravelOrchestrator(llm=_FakeLLM())
    state = {
        "travel_attempt": 1,
        "travel_retry_count": 0,
        "tool_trace": [
            {
                "attempt": 1,
                "tool": "recommend_travel_options",
                "arguments": "{}",
                "output": (
                    '{"mode":"travel_recommendation",'
                    '"flights":[{"flight_no":"CA1"}],'
                    '"hotels":[{"name":"H1"}]}'
                ),
            }
        ],
        "booking_draft": {"recommended_hotel": {"name": "H1", "nightly_cny": "960"}},
        "policy_validation": {
            "status": "failed",
            "checks": [
                {
                    "name": "酒店差标",
                    "status": "failed",
                    "detail": "酒店每晚 960 CNY 超过制度上限 800 CNY",
                }
            ],
            "warnings": [],
            "violations": ["酒店每晚 960 CNY 超过制度上限 800 CNY"],
        },
        "trace": [],
    }

    command = await orch._travel_retry_router(state)  # type: ignore[arg-type]

    assert command.goto == "travel_react_agent"
    assert command.update["travel_retry_count"] == 1
    assert command.update["travel_retry_feedback"]["previous_attempt"] == 1


@pytest.mark.asyncio
async def test_second_failed_attempt_requires_manual_review() -> None:
    orch = LangGraphTravelOrchestrator(llm=_FakeLLM())
    state = {
        "messages": [ChatMessage(role=MessageRole.USER, content="帮我推荐合规差旅行程")],
        "effective_messages": [ChatMessage(role=MessageRole.USER, content="帮我推荐合规差旅行程")],
        "answer": "没有找到合适方案",
        "travel_attempt": 2,
        "travel_retry_count": 1,
        "tool_trace": [
            {
                "attempt": 1,
                "tool": "recommend_travel_options",
                "arguments": "{}",
                "output": '{"mode":"travel_recommendation","flights":[],"hotels":[]}',
            },
            {
                "attempt": 2,
                "tool": "recommend_travel_options",
                "arguments": "{}",
                "output": '{"mode":"travel_recommendation","flights":[],"trains":[],"hotels":[]}',
            },
        ],
        "booking_draft": None,
        "policy_validation": {
            "status": "needs_review",
            "summary": "未找到完整候选，需人工复核。",
            "checks": [],
            "warnings": ["候选为空"],
            "violations": [],
        },
        "trace": [],
    }

    command = await orch._travel_retry_router(state)  # type: ignore[arg-type]

    assert command.goto == "approval_processor"
    assert command.update["travel_retry_exhausted"] is True
    assert command.update["risk_level"] == "high"
    assert command.update["policy_validation"]["retry"]["exhausted"] is True
    assert "转人工审核" in command.update["answer"]

    approval_state = {**state, **command.update}
    approval = await orch._approval_processor(approval_state)  # type: ignore[arg-type]
    assert approval.update["approval_form"]["required"] is True
    assert approval.update["approval_form"]["status"] == "pending_human_approval"


@pytest.mark.asyncio
async def test_passed_retry_continues_without_exhaustion() -> None:
    orch = LangGraphTravelOrchestrator(llm=_FakeLLM())
    state = {
        "travel_attempt": 2,
        "travel_retry_count": 1,
        "tool_trace": [
            {
                "attempt": 2,
                "tool": "recommend_travel_options",
                "arguments": "{}",
                "output": (
                    '{"mode":"travel_recommendation",'
                    '"flights":[{"flight_no":"CA2"}],'
                    '"hotels":[{"name":"H2","nightly_cny":"700"}]}'
                ),
            }
        ],
        "booking_draft": {"recommended_hotel": {"name": "H2", "nightly_cny": "700"}},
        "policy_validation": {
            "status": "passed",
            "checks": [{"name": "酒店差标", "status": "passed", "detail": "符合标准"}],
            "warnings": [],
            "violations": [],
        },
        "trace": [],
    }

    command = await orch._travel_retry_router(state)  # type: ignore[arg-type]

    assert command.goto == "approval_processor"
    assert command.update["travel_retry_exhausted"] is False


def test_booking_draft_uses_only_latest_retry_attempt() -> None:
    state = {
        "travel_attempt": 2,
        "tool_trace": [
            {
                "attempt": 1,
                "arguments": "{}",
                "output": (
                    '{"mode":"travel_recommendation","policy_warnings":["旧候选不合规"],'
                    '"recommendation":{"hotel":{"name":"旧酒店","nightly_cny":"960"}}}'
                ),
            },
            {
                "attempt": 2,
                "arguments": "{}",
                "output": (
                    '{"mode":"travel_recommendation","policy_warnings":[],'
                    '"recommendation":{"hotel":{"name":"新酒店","nightly_cny":"700"}}}'
                ),
            },
        ],
    }

    draft = LangGraphTravelOrchestrator._build_booking_draft(state)  # type: ignore[arg-type]

    assert draft is not None
    assert draft["recommended_hotel"]["name"] == "新酒店"
    assert "旧候选不合规" not in draft["policy_warnings"]


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


@pytest.mark.asyncio
async def test_graph_retries_after_validation_then_requires_manual_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _RetryFlowLLM()
    orch = LangGraphTravelOrchestrator(llm=llm)

    async def empty_recommendation(name: str, arguments: str, user_text: str = "") -> str:
        assert name == "recommend_travel_options"
        return (
            '{"mode":"travel_recommendation","query":'
            '{"employee_id":"u1","grade":"staff","origin_city":"北京",'
            '"destination_city":"上海","departure_date":"2026-10-20"},'
            '"flights":[],"trains":[],"hotels":[],"policy_warnings":[],'
            '"recommendation":{"flight":null,"train":null,"hotel":null}}'
        )

    monkeypatch.setattr(orch, "_execute_tool", empty_recommendation)

    result = await orch.run_completion(
        [
            ChatMessage(
                role=MessageRole.USER,
                content="我是staff，请推荐2026-10-20从北京到上海的航班、高铁和酒店",
            )
        ]
    )

    metadata = result["metadata"]
    assert metadata["travel_retry"]["attempts"] == 2
    assert metadata["travel_retry"]["retry_count"] == 1
    assert metadata["travel_retry"]["exhausted"] is True
    assert [item["attempt"] for item in metadata["tool_trace"]] == [1, 2]
    assert metadata["risk_level"] == "high"
    assert metadata["approval_form"]["required"] is True
    assert metadata["approval_form"]["status"] == "pending_human_approval"
