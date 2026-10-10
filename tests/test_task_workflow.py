"""End-to-end checks for merged planning and task-level execution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent.langgraph_orchestrator import LangGraphTravelOrchestrator
from app.domain.schemas import ChatMessage, MessageRole


def _task(
    task_id: str, intent: str, request: str, *,
    slots: dict | None = None, depends_on: list[str] | None = None,
) -> dict:
    return {
        "id": task_id, "intent": intent, "request": request,
        "slots": slots or {}, "depends_on": depends_on or [], "missing_slots": [],
    }


def _plan(*tasks: dict, primary_intent: str | None = None) -> str:
    return json.dumps({
        "primary_intent": primary_intent or tasks[0]["intent"],
        "tasks": list(tasks), "clarification_question": None,
    }, ensure_ascii=False)


class _LLM:
    model = "fake-model"

    def __init__(self, plan: str) -> None:
        self.plan = plan
        self.calls: list[tuple[list[dict], dict]] = []

    async def chat_completion(self, messages: list[dict], **kwargs: object) -> object:
        self.calls.append((messages, kwargs))
        if kwargs.get("response_format"):
            content = self.plan
        else:
            user_text = next(
                (message["content"] for message in reversed(messages) if message["role"] == "user"),
                "",
            )
            content = f"已处理：{user_text}"
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content, tool_calls=None, refusal=None),
            )],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        )


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.session_writes = 0

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.values[key] = value
        if key.startswith("chat:session:"):
            self.session_writes += 1


@pytest.mark.asyncio
async def test_multi_task_plan_uses_one_planning_call_and_persists_original_question() -> None:
    llm = _LLM(_plan(
        _task("one", "general", "解释出差流程"),
        _task("two", "general", "解释行李规定"),
    ))
    redis = _Redis()
    orch = LangGraphTravelOrchestrator(llm=llm, redis_client=redis)
    request = "解释出差流程和行李规定"

    response = await orch.run_completion(
        [ChatMessage(role=MessageRole.USER, content=request)], session_id="session-1"
    )

    assert response["metadata"]["route"] == "multi_task"
    assert [task["status"] for task in response["metadata"]["task_results"]] == [
        "completed", "completed"
    ]
    assert [task["answer"] for task in response["metadata"]["task_results"]] == [
        "已处理：解释出差流程", "已处理：解释行李规定"
    ]
    assert len([kwargs for _, kwargs in llm.calls if kwargs.get("response_format")]) == 1
    assert response["usage"] == {"prompt_tokens": 6, "completion_tokens": 9, "total_tokens": 15}
    stored = json.loads(redis.values[orch._session_key("session-1")])
    assert [item["content"] for item in stored if item["role"] == "user"] == [request]
    assert [item["role"] for item in stored] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_dependency_receives_real_predecessor_answer() -> None:
    llm = _LLM(_plan(
        _task("first", "general", "解释步骤一"),
        _task("second", "general", "根据第一步解释步骤二", depends_on=["first"]),
    ))
    orch = LangGraphTravelOrchestrator(llm=llm)

    response = await orch.run_completion([
        ChatMessage(role=MessageRole.USER, content="先解释步骤一，再根据结果解释步骤二")
    ])

    assert [item["status"] for item in response["metadata"]["task_results"]] == [
        "completed", "completed"
    ]
    second_messages = llm.calls[-1][0]
    assert any("已处理：解释步骤一" in message["content"] for message in second_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_plan", [
    "not JSON",
    _plan(_task("one", "not_an_intent", "测试未知任务")),
    _plan(_task("one", "general", "测试任务", depends_on=["missing"])),
])
async def test_invalid_plan_clarifies_without_executing(invalid_plan: str) -> None:
    llm = _LLM(invalid_plan)
    orch = LangGraphTravelOrchestrator(llm=llm)

    response = await orch.run_completion([
        ChatMessage(role=MessageRole.USER, content="处理这两个请求")
    ])

    assert response["metadata"]["route"] == "clarification"
    assert "无法可靠生成任务计划" in response["choices"][0]["message"]["content"]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_required_slots_are_checked_even_if_model_omits_missing_slots() -> None:
    llm = _LLM(_plan(_task("flight", "search_flight", "查询北京到上海的航班")))
    orch = LangGraphTravelOrchestrator(llm=llm)

    response = await orch.run_completion([
        ChatMessage(role=MessageRole.USER, content="帮我查航班")
    ])

    assert response["metadata"]["route"] == "clarification"
    assert "depart_date" in response["choices"][0]["message"]["content"]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_multi_task_renumbers_citations_but_preserves_local_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _LLM(_plan(
        _task("one", "policy", "查询差旅规定一"),
        _task("two", "policy", "查询差旅规定二"),
    ))
    orch = LangGraphTravelOrchestrator(llm=llm)

    async def run_task(state: dict, task: dict, dependencies: list[dict]) -> dict:
        _ = state, dependencies
        return {
            "task_id": task["id"], "intent": task["intent"],
            "request": task["request"], "status": "completed",
            "answer": f"依据[{1}]：{task['request']}",
            "citations": [{"chunk_id": task["id"], "content": task["request"]}],
            "claim_evidence_map": [{"claim": task["request"], "status": "supported"}],
            "verification": {"passed": True}, "risk_level": "low",
            "tool_trace": [], "trace": [],
        }

    monkeypatch.setattr(orch, "_execute_planned_task", run_task)
    response = await orch.run_completion([
        ChatMessage(role=MessageRole.USER, content="查询两个差旅规定")
    ])

    assert "依据[1]" in response["choices"][0]["message"]["content"]
    assert "依据[2]" in response["choices"][0]["message"]["content"]
    assert [item["chunk_id"] for item in response["metadata"]["citations"]] == ["one", "two"]
    assert response["metadata"]["task_results"][1]["answer"].startswith("依据[1]")
    assert response["metadata"]["task_results"][1]["verification"]["passed"] is True


@pytest.mark.asyncio
async def test_inventory_task_requires_a_real_matching_tool_result() -> None:
    orch = LangGraphTravelOrchestrator(llm=_LLM(""))
    task = _task("flight", "search_flight", "查询北京到上海的航班")
    base_state = {
        "messages": [ChatMessage(role=MessageRole.USER, content=task["request"])],
        "effective_messages": [ChatMessage(role=MessageRole.USER, content=task["request"])],
        "intent": "search_flight", "answer": "有 CA123 航班", "risk_level": "medium",
        "policy_validation": {"status": "needs_review"}, "travel_attempt": 1,
    }

    failed_state = {**base_state, "tool_trace": []}
    failed_answer = (await orch._finish_task(failed_state))["answer"]
    assert "未能取得可核实的查询结果" in failed_answer
    assert orch._task_result(task, {**failed_state, "answer": failed_answer})["status"] == "needs_review"

    good_state = {
        **base_state,
        "tool_trace": [{
            "tool": "search_flights", "attempt": 1,
            "output": json.dumps({"mode": "flight", "results": [{"flight_no": "CA123"}]}),
        }],
    }
    assert orch._inventory_issue(good_state) is None
    assert orch._task_result(task, good_state)["status"] == "completed"
