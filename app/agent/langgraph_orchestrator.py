from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypedDict
from zoneinfo import ZoneInfo

from langgraph.types import Command

from app.agent.orchestrator import (
    TravelOrchestrator,
    _runtime_system_prompt,
    _to_openai_messages,
    _travel_tools,
)
from app.agent.task_scheduler import execute_task_plan
from app.config import settings
from app.core.intent.recognizer import TravelIntent
from app.domain.schemas import ChatMessage, MessageRole, StreamChunk, StreamChunkType
from app.domain.task_plan import ExecutionPlan, PlannedTask, execution_plan_response_format
from app.core.rag.evidence import AnswerDraft, EvidenceError, EvidencePack, render_draft
from app.services.evidence_qa import EvidenceQA, diagnostic_summary
from app.services.embeddings import EmbeddingService
from app.services.human_review import HumanReviewStore, prepare_review


class TravelGraphState(TypedDict, total=False):
    messages: list[ChatMessage]
    session_id: str | None
    user_id: str | None
    effective_messages: list[ChatMessage]
    openai_messages: list[dict[str, Any]]
    memory_context: str
    long_term_memories: list[str]
    current_facts: list[str]
    intent: str
    route: str
    answer: str
    response: dict[str, Any]
    usage: dict[str, int] | None
    tool_trace: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    approval_form: dict[str, Any] | None
    booking_draft: dict[str, Any] | None
    execution_plan: dict[str, Any] | None
    policy_constraints: dict[str, Any] | None
    policy_validation: dict[str, Any] | None
    travel_attempt: int
    travel_retry_count: int
    travel_retry_feedback: dict[str, Any] | None
    travel_retry_exhausted: bool
    risk_level: str | None
    answer_mode: str | None
    verification: dict[str, Any] | None
    claim_evidence_map: list[dict[str, Any]]
    rag_correction_count: int
    reflection_notes: str
    conversation_messages: list[ChatMessage]
    active_task: dict[str, Any] | None
    task_results: list[dict[str, Any]]
    plan_error: str | None
    planner_usage: dict[str, int] | None
    dependency_context: str
    rag_question: str
    rag_task_request: str
    rag_evidence: dict[str, Any] | None
    rag_draft: dict[str, Any] | None
    rag_stages: list[dict[str, Any]]
    human_review: dict[str, Any] | None


class LangGraphTravelOrchestrator(TravelOrchestrator):
    """LangGraph workflow with one tool-using ReAct agent and supporting nodes."""

    def __init__(
        self,
        *args: Any,
        document_store: Any | None = None,
        rag_retriever: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._document_store = document_store
        self._rag_retriever = rag_retriever
        self._evidence_qa = EvidenceQA(self._llm, timeout_seconds=settings.rag_evidence_timeout_seconds)
        self._task_graph = self._build_task_graph()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(TravelGraphState)
        graph.add_node("context_builder", self._context_builder)
        graph.add_node("memory_fusion", self._memory_fusion)
        graph.add_node("input_guardrail", self._input_guardrail)
        graph.add_node("planner", self._planner)
        graph.add_node("intent_router", self._intent_router)
        graph.add_node("multi_task_executor", self._multi_task_executor)
        self._add_execution_nodes(graph)
        graph.add_node("response_finalizer", self._response_finalizer)

        graph.add_edge(START, "context_builder")
        graph.add_edge("context_builder", "memory_fusion")
        graph.add_edge("memory_fusion", "input_guardrail")
        graph.add_edge("response_finalizer", END)
        return graph.compile()

    def _add_execution_nodes(self, graph: Any) -> None:
        """Reuse the same business and verification nodes in each isolated task."""
        graph.add_node("policy_reasoner", self._policy_reasoner)
        graph.add_node("rag_responder", self._rag_responder)
        graph.add_node("rag_evidence_builder", self._rag_evidence_builder)
        graph.add_node("rag_answer_generator", self._rag_answer_generator)
        graph.add_node("travel_react_agent", self._travel_react_agent)
        graph.add_node("policy_validator", self._policy_validator)
        graph.add_node("travel_retry_router", self._travel_retry_router)
        graph.add_node("approval_processor", self._approval_processor)
        graph.add_node("general_responder", self._general_responder)
        graph.add_node("grounding_verifier", self._grounding_verifier)
        graph.add_node("rag_self_corrector", self._rag_self_corrector)
        graph.add_node("response_reviewer", self._response_reviewer)

    def _build_task_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(TravelGraphState)
        graph.add_node("task_entry", self._task_entry)
        self._add_execution_nodes(graph)
        # Child tasks never persist conversations or overwrite the session's latest draft.
        graph.add_node("response_finalizer", self._finish_task)
        graph.add_edge(START, "task_entry")
        graph.add_edge("response_finalizer", END)
        return graph.compile()

    async def _task_entry(
        self, state: TravelGraphState
    ) -> Command[Literal["policy_reasoner", "rag_responder", "general_responder"]]:
        return Command(goto=self._route_after_intent(state.get("intent")))

    async def _finish_task(self, state: TravelGraphState) -> dict[str, Any]:
        inventory_issue = self._inventory_issue(state)
        answer = self._enforce_enterprise_answer_contract(
            inventory_issue or state.get("answer", ""), self._current_attempt_tool_trace(state),
            state.get("effective_messages") or state["messages"],
        )
        review = state.get("human_review") or prepare_review(state, answer)
        if review:
            answer = "【待人工审核】当前问题存在尚未确认的条件，审核后提供正式答复。\n\n候选资料摘要（尚未人工审核，不作为最终结论）：\n" + answer
        return {"answer": answer, "human_review": review, "rag_stages": self._rag_stage(
            state, "final", {"answer": answer, "answer_mode": state.get("answer_mode")},
        ) if state.get("rag_stages") else []}

    async def _context_builder(self, state: TravelGraphState) -> dict[str, Any]:
        incoming = state["messages"]
        session_id = state.get("session_id")
        effective_messages = await self._effective_messages(incoming, session_id)
        messages = await self._maybe_summarize_thread(effective_messages)
        messages = self._trim_window(messages)
        openai_messages: list[dict[str, Any]] = [{"role": "system", "content": _runtime_system_prompt()}]
        openai_messages.extend(_to_openai_messages(messages))
        return {
            "effective_messages": effective_messages,
            "conversation_messages": effective_messages,
            "openai_messages": openai_messages,
            "tool_trace": [],
            "trace": self._append_trace(state, "context_builder", "prepared"),
            "citations": [],
            "approval_form": None,
            "booking_draft": None,
            "execution_plan": None,
            "policy_constraints": None,
            "policy_validation": None,
            "travel_attempt": 0,
            "travel_retry_count": 0,
            "travel_retry_feedback": None,
            "travel_retry_exhausted": False,
            "risk_level": "low",
            "answer_mode": None,
            "verification": None,
            "claim_evidence_map": [],
            "rag_correction_count": 0,
            "memory_context": "",
            "long_term_memories": [],
            "current_facts": [],
            "active_task": None,
            "task_results": [],
            "plan_error": None,
            "planner_usage": None,
            "dependency_context": "",
            "rag_question": "", "rag_task_request": "", "rag_evidence": None,
            "rag_draft": None, "rag_stages": [], "human_review": None,
        }

    async def _memory_fusion(self, state: TravelGraphState) -> dict[str, Any]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        memory_owner = state.get("user_id") or state.get("session_id")
        current_facts = self._extract_long_term_facts(text)
        stored_memories = await self._load_long_term_memories(memory_owner)
        if current_facts:
            stored_memories = await self._save_long_term_memories(memory_owner, stored_memories, current_facts)

        memory_context = self._build_memory_context(current_facts, stored_memories)
        openai_messages = list(state["openai_messages"])
        if memory_context:
            openai_messages.insert(1, {"role": "system", "content": memory_context})
        return {
            "openai_messages": openai_messages,
            "memory_context": memory_context,
            "current_facts": current_facts,
            "long_term_memories": stored_memories,
            "trace": self._append_trace(
                state,
                "memory_fusion",
                "fused",
                {
                    "current_facts": len(current_facts),
                    "long_term_memories": len(stored_memories),
                    "owner": bool(memory_owner),
                },
            ),
        }

    async def _input_guardrail(
        self, state: TravelGraphState
    ) -> Command[Literal["planner", "response_finalizer"]]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        blocked = self._input_guardrail_message(text)
        trace = self._append_trace(
            state,
            "input_guardrail",
            "blocked" if blocked else "passed",
        )
        if blocked:
            return Command(
                update={"answer": blocked, "trace": trace, "risk_level": "medium"},
                goto="response_finalizer",
            )
        return Command(update={"trace": trace}, goto="planner")

    async def _planner(
        self, state: TravelGraphState
    ) -> Command[Literal["intent_router"]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是企业商旅意图识别与任务规划器。用一次分析完整识别本轮用户的所有任务，"
                    "只输出符合 JSON Schema 的 JSON 实例。primary_intent 必须属于某个任务的 intent。"
                    "tasks 是能够独立交付结果的业务任务，不是工具调用列表。查航班并解释报销流程拆成两个任务；"
                    "生成符合差标的完整出差方案可作为一个 trip_planning 任务，内部工具步骤不要另拆任务。"
                    "policy/rag 表示制度知识问答；info_query 表示知识查询；需要实时库存时使用对应 search 意图。"
                    "booking/application 表示草稿或审批准备，不能承诺已预订、付款或已提交。"
                    "每个任务的 request 必须独立完整，保留否定、日期、偏好和限制；slots 使用工具参数名。"
                    "结合会话上下文解析指代，用户本轮明确修改优先；不得补造未知参数、库存或制度。"
                    "未知或不适用槽位填 null，缺少执行必填参数时填写 missing_slots 和 clarification_question。"
                    "没有缺失信息时 clarification_question 为 null。依赖另一任务结果才填写 depends_on，"
                    "依赖必须使用有效任务 ID，禁止环路；不要仅为排列顺序创造依赖。"
                    "闲聊也返回一个 general 任务。航班必需 origin/destination/depart_date；"
                    "火车必需 origin_station/dest_station/depart_date；酒店必需 city/check_in/check_out；"
                    "行程、申请、预订草稿必需 employee_id/grade/origin_city/destination_city/departure_date。"
                    f"当前日期：{datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()}，时区 Asia/Shanghai。"
                ),
            }
        ]
        if state.get("memory_context"):
            messages.append({"role": "system", "content": state["memory_context"]})
        if state.get("user_id"):
            messages.append({"role": "system", "content": f"当前用户标识：{state['user_id']}"})
        history = state.get("effective_messages") or state["messages"]
        messages.extend(_to_openai_messages(history[-settings.memory_window_size:]))
        response_format = execution_plan_response_format()
        if settings.planner_response_format == "json_object":
            messages[0]["content"] += "\nJSON Schema：" + json.dumps(
                response_format["json_schema"]["schema"], ensure_ascii=False
            )
            response_format = {"type": "json_object"}

        plan: dict[str, Any] | None = None
        error: str | None = None
        usage: dict[str, int] | None = state.get("usage")
        try:
            resp = await asyncio.wait_for(
                self._llm.chat_completion(
                    messages, temperature=0.0, response_format=response_format
                ), timeout=settings.planner_timeout_seconds,
            )
            usage = self._usage_dict(resp)
            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) not in (None, "stop"):
                raise ValueError("Incomplete plan")
            if getattr(choice.message, "refusal", None):
                raise ValueError("Planning refused")
            validated = ExecutionPlan.model_validate_json(choice.message.content or "")
            plan = validated.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            # Never convert a failed multi-task plan into a guessed single task.
            error = type(exc).__name__

        return Command(
            update={
                "execution_plan": plan,
                "plan_error": error,
                "planner_usage": usage,
                "usage": None,
                "trace": self._append_trace(
                    state,
                    "planner",
                    "invalid_plan" if error else "planned",
                    {
                        "primary_intent": plan.get("primary_intent") if plan else None,
                        "task_count": len(plan["tasks"]) if plan else 0,
                        "error": error,
                    },
                ),
            },
            goto="intent_router",
        )

    async def _intent_router(
        self, state: TravelGraphState
    ) -> Command[Literal[
        "policy_reasoner", "rag_responder", "general_responder",
        "multi_task_executor", "response_finalizer",
    ]]:
        try:
            plan = ExecutionPlan.model_validate_json(json.dumps(state.get("execution_plan")))
        except ValueError:
            return self._clarify_plan(
                state, "暂时无法可靠生成任务计划。请重试，或明确列出需要完成的任务及条件。"
            )
        missing: list[str] = []
        for task in plan.tasks:
            fields = self._missing_task_slots(task)
            if fields:
                missing.append(f"{task.request}：{', '.join(fields)}")
        if missing or plan.clarification_question:
            question = plan.clarification_question or "请补充以下任务所需的信息。"
            if missing:
                question += "\n" + "\n".join(missing)
            return self._clarify_plan(state, question)

        intent = plan.primary_intent.value
        multi = len(plan.tasks) > 1
        update: dict[str, Any] = {}
        if not multi:
            update = self._task_state(state, plan.tasks[0].model_dump(mode="json"), [])
        route = "multi_task" if multi else "single_task"
        return Command(
            update={
                **update,
                "intent": intent,
                "route": route,
                "trace": self._append_trace(
                    state, "intent_router", route, {"intent": intent, "task_count": len(plan.tasks)}
                ),
            },
            goto="multi_task_executor" if multi else self._route_after_intent(intent),
        )

    def _route_after_intent(
        self, intent: str | None, text: str = ""
    ) -> Literal["policy_reasoner", "rag_responder", "general_responder"]:
        travel_intents = {
            TravelIntent.SEARCH_FLIGHT.value,
            TravelIntent.SEARCH_HOTEL.value,
            TravelIntent.SEARCH_TRAIN.value,
            TravelIntent.TRIP_PLANNING.value,
            TravelIntent.APPLICATION.value,
            TravelIntent.BOOKING.value,
        }
        if intent in {TravelIntent.POLICY.value, TravelIntent.RAG.value, TravelIntent.INFO_QUERY.value}:
            return "rag_responder"
        return "policy_reasoner" if intent in travel_intents else "general_responder"

    def _clarify_plan(
        self, state: TravelGraphState, question: str
    ) -> Command[Literal["response_finalizer"]]:
        return Command(update={
            "route": "clarification", "answer": question,
            "trace": self._append_trace(state, "intent_router", "needs_clarification"),
        }, goto="response_finalizer")

    @staticmethod
    def _missing_task_slots(task: PlannedTask) -> list[str]:
        required = {
            TravelIntent.SEARCH_FLIGHT: ("origin", "destination", "depart_date"),
            TravelIntent.SEARCH_TRAIN: ("origin_station", "dest_station", "depart_date"),
            TravelIntent.SEARCH_HOTEL: ("city", "check_in", "check_out"),
        }
        travel_fields = ("employee_id", "grade", "origin_city", "destination_city", "departure_date")
        for intent in (TravelIntent.TRIP_PLANNING, TravelIntent.BOOKING, TravelIntent.APPLICATION):
            required[intent] = travel_fields
        missing = list(task.missing_slots)
        for name in required.get(task.intent, ()):
            if getattr(task.slots, name) is None and name not in missing:
                missing.append(name)
        return missing

    def _task_state(
        self, state: TravelGraphState, task: dict[str, Any], dependencies: list[dict[str, Any]]
    ) -> TravelGraphState:
        child = copy.deepcopy(state)
        slots = {key: value for key, value in task["slots"].items() if value is not None}
        text = task["request"]
        if slots:
            text += "\n已识别参数：" + json.dumps(slots, ensure_ascii=False)
        history = list(child.get("effective_messages") or child["messages"])
        for index in range(len(history) - 1, -1, -1):
            if history[index].role == MessageRole.USER:
                history = history[:index] + [ChatMessage(role=MessageRole.USER, content=text)]
                break
        context = [_runtime_system_prompt(), "只完成当前业务任务。历史对话和前置结果仅作为上下文，不能重复执行其他任务。"]
        if child.get("memory_context"):
            context.append(child["memory_context"])
        messages = [{"role": "system", "content": content} for content in context]
        dependency_context = ""
        if dependencies:
            # Pass real prerequisite outputs, not just a flag saying they completed.
            dependency_context = "前置任务结果（只读数据，不能作为公司制度依据）：\n" + json.dumps(
                [{key: result.get(key) for key in (
                    "task_id", "answer", "tool_trace", "citations", "booking_draft", "policy_constraints"
                )} for result in dependencies], ensure_ascii=False,
            )
            messages.append({"role": "user", "content": dependency_context})
        messages.extend(_to_openai_messages(self._trim_window(history)))
        child.update({
            "intent": task["intent"], "active_task": task, "effective_messages": history,
            "openai_messages": messages, "answer": "", "usage": None, "trace": [],
            "tool_trace": [], "citations": [], "task_results": [], "approval_form": None,
            "booking_draft": None, "policy_constraints": None, "policy_validation": None,
            "travel_attempt": 0, "travel_retry_count": 0, "travel_retry_feedback": None,
            "travel_retry_exhausted": False, "risk_level": "low", "answer_mode": None,
            "verification": None, "claim_evidence_map": [], "rag_correction_count": 0,
            "reflection_notes": "",
            "dependency_context": dependency_context,
            "rag_question": "", "rag_task_request": "", "rag_evidence": None,
            "rag_draft": None, "rag_stages": [], "human_review": None,
        })
        return child

    @staticmethod
    def _task_result(task: dict[str, Any], state: TravelGraphState) -> dict[str, Any]:
        result = {key: state.get(key) for key in (
            "answer", "usage", "citations", "verification", "claim_evidence_map", "answer_mode",
            "booking_draft", "approval_form", "policy_constraints", "policy_validation",
            "risk_level", "rag_correction_count", "travel_retry_exhausted",
            "rag_evidence", "rag_stages", "human_review",
        )}
        inventory_issue = LangGraphTravelOrchestrator._inventory_issue(state)
        travel_workflow = task["intent"] in {
            TravelIntent.TRIP_PLANNING.value,
            TravelIntent.APPLICATION.value,
            TravelIntent.BOOKING.value,
        }
        needs_review = (
            bool(inventory_issue)
            or bool(state.get("human_review"))
            or not state.get("answer")
            or state.get("risk_level") == "high"
            or state.get("answer_mode") == "llm_fallback"
            or (state.get("verification") or {}).get("passed") is False
            or (state.get("verification") or {}).get("question_answered") is False
            or (
                travel_workflow
                and (state.get("policy_validation") or {}).get("status") in {"failed", "needs_review"}
            )
        )
        result.update({
            "task_id": task["id"], "intent": task["intent"], "request": task["request"],
            "status": "needs_review" if needs_review else "completed",
            "tool_trace": [{**item, "task_id": task["id"]} for item in state.get("tool_trace") or []],
            "trace": [{**item, "task_id": task["id"]} for item in state.get("trace") or []],
        })
        return result

    @staticmethod
    def _inventory_issue(state: TravelGraphState) -> str | None:
        expected = {
            TravelIntent.SEARCH_FLIGHT.value: ("search_flights", "flight"),
            TravelIntent.SEARCH_HOTEL.value: ("search_hotels", "hotel"),
            TravelIntent.SEARCH_TRAIN.value: ("search_trains", "train"),
        }.get(state.get("intent"))
        if expected is None:
            return None
        tool_name, mode = expected
        for item in LangGraphTravelOrchestrator._current_attempt_tool_trace(state):
            if item.get("tool") != tool_name:
                continue
            try:
                payload = json.loads(item.get("output") or "")
            except (TypeError, json.JSONDecodeError):
                continue
            if (
                isinstance(payload, dict) and payload.get("mode") == mode
                and not payload.get("error") and isinstance(payload.get("results"), list)
                and any(isinstance(row, dict) for row in payload["results"])
            ):
                return None
        return "当前未能取得可核实的查询结果，不能据此提供库存、价格或余票结论。"

    async def _execute_planned_task(
        self, state: TravelGraphState, task: dict[str, Any], dependencies: list[dict[str, Any]]
    ) -> dict[str, Any]:
        result = await self._task_graph.ainvoke(
            self._task_state(state, task, dependencies),
            config={"recursion_limit": 20 + 5 * max(0, settings.travel_validation_max_retries)},
        )
        return self._task_result(task, result)

    async def _multi_task_executor(
        self, state: TravelGraphState
    ) -> Command[Literal["response_finalizer"]]:
        tasks = state["execution_plan"]["tasks"]

        async def runner(task: dict[str, Any], dependencies: list[dict[str, Any]]) -> dict[str, Any]:
            return await self._execute_planned_task(state, task, dependencies)

        results = await execute_task_plan(
            tasks, runner, max_concurrency=settings.task_max_concurrency,
            timeout_seconds=lambda task: settings.rag_evidence_task_timeout_seconds
            if task["intent"] in {"policy", "rag"} else settings.task_timeout_seconds,
        )
        for result in results:
            if result.get("status") == "failed" and result.get("intent") in {"policy", "rag", "info_query"}:
                result["human_review"] = prepare_review({
                    "messages": state["messages"], "conversation_messages": state.get("conversation_messages"),
                    "rag_task_request": result["request"], "active_task": {"id": result["task_id"]},
                    "verification": {"passed": False, "stage": "task_execution", "question_answered": False},
                }, result.get("answer") or "自动制度问答任务未完成。")
                result["status"] = "needs_review"
                result["answer"] = "【待人工审核】自动制度问答任务未能完成，需人工检查后提供正式答复。"
        sections: list[str] = []
        citations: list[dict[str, Any]] = []
        trace = list(state.get("trace") or [])
        status_labels = {"completed": "已完成", "needs_review": "需复核", "failed": "失败", "blocked": "未执行"}
        for index, result in enumerate(results, start=1):
            # Each child was already verified against its own citations. Renumber only
            # after verification, keeping claim-evidence maps local to each task.
            offset = len(citations)
            local_citations = result.get("citations") or []
            answer = result.get("answer") or "该任务未返回结果。"
            if local_citations:
                answer = re.sub(
                    r"\[(\d+)\]",
                    lambda match: f"[{int(match[1]) + offset}]"
                    if 1 <= int(match[1]) <= len(local_citations) else match[0], answer,
                )
                citations.extend({**item, "metadata": {
                    **(item.get("metadata") or {}), "task_id": result["task_id"],
                }} for item in local_citations)
            sections.append(
                f"{index}. {result['request']}（{status_labels[result['status']]}）\n\n{answer}"
            )
            trace.extend(result.get("trace") or [])
        risk_order = {"low": 0, "medium": 1, "high": 2}
        return Command(update={
            "answer": "\n\n".join(sections), "task_results": results, "citations": citations,
            "usage": self._sum_usage(*(result.get("usage") for result in results)),
            "tool_trace": [item for result in results for item in result.get("tool_trace") or []],
            "risk_level": max((result.get("risk_level") or "low" for result in results), key=risk_order.get),
            "verification": {"scope": "per_task", "tasks": {
                result["task_id"]: {"status": result["status"], "verification": result.get("verification")}
                for result in results
            }},
            "trace": self._append_trace({**state, "trace": trace}, "multi_task_executor", "completed",
                                        {"task_count": len(results)}),
        }, goto="response_finalizer")

    def _route_after_answer(
        self, state: TravelGraphState
    ) -> Literal["response_reviewer", "grounding_verifier"]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        if any(k in text for k in ("反思", "检查一遍", "复核", "挑错")):
            return "response_reviewer"
        return "grounding_verifier"

    async def _policy_reasoner(
        self, state: TravelGraphState
    ) -> Command[Literal["travel_react_agent"]]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        citations: list[dict[str, Any]] = []
        constraints: dict[str, Any] = {
            "source": "unavailable",
            "constraints": {},
            "confidence": 0.0,
            "notes": ["未检索到可用制度约束，后续合规校验将要求人工复核。"],
        }
        if self._knowledge_retrieval_available():
            try:
                query = f"{text}\n差旅制度 酒店标准 舱位 提前预订 审批 金额"
                hits = await self._retrieve_knowledge(query)
                citations = [self._citation_from_hit(hit) for hit in hits]
                constraints = await self._extract_policy_constraints(text, citations, state.get("memory_context", ""))
            except Exception as exc:  # noqa: BLE001
                constraints = {
                    "source": "retrieval_failed",
                    "constraints": {},
                    "confidence": 0.0,
                    "notes": [f"制度约束检索失败：{exc!s}"],
                }

        merged_citations = list(state.get("citations") or [])
        for item in citations:
            if item and item not in merged_citations:
                merged_citations.append(item)
        return Command(
            update={
                "citations": merged_citations,
                "policy_constraints": constraints,
                "trace": self._append_trace(
                    state,
                    "policy_reasoner",
                    "constraints_extracted",
                    {
                        "source": constraints.get("source"),
                        "confidence": constraints.get("confidence"),
                        "citation_count": len(citations),
                        "retrieval": self._retrieval_trace(citations),
                    },
                ),
            },
            goto="travel_react_agent",
        )

    async def _rag_responder(
        self, state: TravelGraphState
    ) -> Command[Literal["rag_evidence_builder", "response_finalizer"]]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        question = self._last_user_text(state.get("conversation_messages") or state["messages"])
        if not self._knowledge_retrieval_available():
            return self._rag_failure(state, "retrieval", "knowledge_base_unavailable")
        try:
            hits = await self._retrieve_knowledge(text)
        except Exception as exc:
            return self._rag_failure(state, "retrieval", type(exc).__name__, exception=exc)
        citations = [self._citation_from_hit(hit) for hit in hits]
        if not citations:
            return self._rag_failure(state, "retrieval", "no_evidence")
        update = {
            "citations": citations, "rag_question": question, "rag_task_request": text,
            "rag_stages": self._rag_stage(state, "retrieval", {"query": text, **self._retrieval_trace(citations)}),
            "trace": self._append_trace(
                state, "rag_responder", "retrieved", {"hit_count": len(citations), "retrieval": self._retrieval_trace(citations)},
            ),
        }
        return Command(update=update, goto="rag_evidence_builder")

    async def _rag_evidence_builder(self, state: TravelGraphState) -> Command[Literal["rag_answer_generator", "response_finalizer"]]:
        citations = list(state.get("citations") or [])
        usage = state.get("usage")
        stages = list(state.get("rag_stages") or [])
        diagnostics: list[dict[str, Any]] = []
        try:
            pack, records, call_usage = await self._evidence_qa.build(
                state["rag_question"], citations, task_request=state.get("rag_task_request", ""),
                diagnostics=diagnostics,
            )
            usage = self._sum_usage(usage, call_usage)
            # Every retrieval still uses the configured Top K. Extra queries are bounded
            # and only fill requirements identified as missing, never loop indefinitely.
            if any(r.status == "missing" for r in pack.requirements):
                known = {c["chunk_id"] for c in citations}
                queries = list(dict.fromkeys(pack.search_queries))[:settings.rag_evidence_max_supplemental_queries]
                added = False
                for query in queries:
                    try:
                        hits = await self._retrieve_knowledge(query)
                    except Exception as exc:
                        stages.append({"stage": "supplemental_retrieval", "query": query, "error": type(exc).__name__})
                        continue
                    new = []
                    for hit in hits:
                        citation = self._citation_from_hit(hit)
                        if citation["chunk_id"] not in known:
                            known.add(citation["chunk_id"])
                            citations.append(citation)
                            new.append(citation["chunk_id"])
                    added = added or bool(new)
                    stages.append({"stage": "supplemental_retrieval", "query": query, "added_chunk_ids": new})
                if added:
                    pack, records, call_usage = await self._evidence_qa.build(
                        state["rag_question"], citations, task_request=state.get("rag_task_request", ""),
                        diagnostics=diagnostics,
                    )
                    usage = self._sum_usage(usage, call_usage)
        except Exception as exc:
            return self._rag_failure({**state, "usage": usage, "citations": citations, "rag_stages": stages}, "evidence", type(exc).__name__, exception=exc, diagnostics=diagnostics)
        evidence = {"question": state["rag_question"], "task_request": state.get("rag_task_request"), "pack": pack.model_dump(mode="json"), "facts": records}
        stages.append({"stage": "evidence", "requirements": evidence["pack"]["requirements"], "facts": records, **diagnostic_summary(diagnostics)})
        return Command(update={
            "rag_evidence": evidence, "citations": citations, "usage": usage, "rag_stages": stages,
            "trace": self._append_trace(state, "rag_evidence_builder", "prepared", {"fact_count": len(records)}),
        }, goto="rag_answer_generator")

    async def _rag_answer_generator(self, state: TravelGraphState) -> Command[Literal["grounding_verifier", "response_finalizer"]]:
        evidence = state["rag_evidence"]
        pack = EvidencePack.model_validate(evidence["pack"])
        diagnostics: list[dict[str, Any]] = []
        try:
            draft, usage = await self._evidence_qa.draft(
                state["rag_question"], pack, evidence["facts"], state["citations"],
                task_request=state.get("rag_task_request", ""),
                diagnostics=diagnostics,
            )
            answer = render_draft(draft, evidence["facts"], state["citations"])
        except Exception as exc:
            return self._rag_failure(state, "draft", type(exc).__name__, exception=exc, diagnostics=diagnostics)
        return Command(update={
            "rag_draft": draft.model_dump(mode="json"), "answer": answer, "answer_mode": "rag_grounded",
            "usage": self._sum_usage(state.get("usage"), usage),
            "rag_stages": self._rag_stage(state, "draft", {"answer": answer, "claims": draft.model_dump(mode="json")["claims"], **diagnostic_summary(diagnostics)}),
            "trace": self._append_trace(state, "rag_answer_generator", "drafted"),
        }, goto="grounding_verifier")

    @staticmethod
    def _rag_stage(state: TravelGraphState, stage: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        return [*(state.get("rag_stages") or []), {"stage": stage, **data}]

    def _rag_failure(self, state: TravelGraphState, stage: str, error: str, *, exception: Exception | None = None, diagnostics: list[dict[str, Any]] | None = None) -> Command[Literal["response_finalizer"]]:
        status = getattr(exception, "status_code", None)
        detail = {"upstream_status": status} if isinstance(status, int) else {}
        detail.update(diagnostic_summary(diagnostics or []))
        if isinstance(exception, EvidenceError):
            detail["validation_error"] = exception.diagnostic()
        answer = "当前制度问答处理失败，暂时无法给出经过核验的回答，请稍后重试。"
        if error == "no_evidence":
            answer = "当前未检索到该问题的制度依据，请补充相关制度资料或调整问题。"
        elif error == "knowledge_base_unavailable":
            answer = "知识库当前不可用，暂时无法检索制度依据，请稍后重试。"
        elif status == 402:
            answer = "模型服务返回402（账户余额或计费状态异常），暂时无法完成制度问答，请检查模型服务账户。"
        return Command(update={
            "answer": answer,
            "answer_mode": "llm_fallback", "risk_level": "medium",
            "verification": {"passed": False, "reason": error, "stage": stage, "claim_evidence_map": [], **detail},
            "rag_stages": self._rag_stage(state, stage, {"error": error, **detail}),
            "trace": self._append_trace(state, "rag_" + stage, "failed", {"error": error, **detail}),
            "usage": self._sum_usage(state.get("usage"), getattr(exception, "usage", None)) if isinstance(exception, EvidenceError) else state.get("usage"), "citations": state.get("citations") or [],
        }, goto="response_finalizer")

    async def _travel_react_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["policy_validator"]]:
        messages = list(state["openai_messages"])
        attempt = int(state.get("travel_retry_count") or 0) + 1
        if state.get("execution_plan"):
            messages.append(
                {
                    "role": "system",
                    "content": "执行计划（由 planner 节点生成）：\n"
                    + json.dumps(state.get("active_task") or state["execution_plan"], ensure_ascii=False),
                }
            )
        if state.get("policy_constraints"):
            messages.append(
                {
                    "role": "system",
                    "content": "制度约束（由 RAG Policy Reasoner 生成，必须优先遵守）：\n"
                    + json.dumps(state["policy_constraints"], ensure_ascii=False),
                }
            )
        retry_feedback = state.get("travel_retry_feedback")
        if attempt > 1 and retry_feedback:
            max_attempts = max(1, int(settings.travel_validation_max_retries) + 1)
            attempt_label = "最后一次自动尝试" if attempt >= max_attempts else "自动重试"
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"这是第 {attempt} 次候选生成（{attempt_label}）。"
                        "上一轮候选未通过校验，必须根据反馈重新调用查询/推荐工具，"
                        "优先选择满足制度约束且库存、价格信息完整的新候选；不得直接复用上一轮不合规组合。"
                        "如果仍找不到合适方案，必须明确说明没有合规候选，不得编造库存。\n"
                        "上一轮校验反馈：\n"
                        + json.dumps(retry_feedback, ensure_ascii=False)
                    ),
                }
            )
        tools = _travel_tools()
        search_tools = {
            TravelIntent.SEARCH_FLIGHT.value: "search_flights",
            TravelIntent.SEARCH_HOTEL.value: "search_hotels",
            TravelIntent.SEARCH_TRAIN.value: "search_trains",
        }
        if state.get("intent") in search_tools:
            tools = [tool for tool in tools if tool["function"]["name"] == search_tools[state["intent"]]]
        allowed_tools = {tool["function"]["name"] for tool in tools}
        tool_trace: list[dict[str, Any]] = list(state.get("tool_trace") or [])
        usage: dict[str, int] | None = None
        user_text = self._last_user_text(state.get("effective_messages") or state["messages"])

        for _ in range(settings.max_react_iterations):
            resp = await self._llm.chat_completion(messages, tools=tools, tool_choice="auto")
            choice = resp.choices[0]
            msg = choice.message
            usage = self._usage_dict(resp)

            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or "{}",
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    if tc.function.name not in allowed_tools:
                        raise ValueError("Tool call does not belong to the current task")
                    output = await self._execute_tool(tc.function.name, tc.function.arguments, user_text)
                    tool_trace.append(
                        {
                            "tool": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                            "output": output,
                            "attempt": attempt,
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": output,
                        }
                    )
                continue

            update = {
                "answer": msg.content or "",
                "usage": usage,
                "tool_trace": tool_trace,
                "travel_attempt": attempt,
                "trace": self._append_trace(
                    state,
                    "travel_react_agent",
                    "answered",
                    {
                        "attempt": attempt,
                        "tool_calls": [
                            t.get("tool")
                            for t in tool_trace
                            if not isinstance(t, dict) or t.get("attempt") == attempt
                        ],
                    },
                ),
            }
            return Command(update=update, goto="policy_validator")

        update = {
            "answer": "已达到最大推理轮次，请简化问题后重试。",
            "usage": usage,
            "tool_trace": tool_trace,
            "travel_attempt": attempt,
            "risk_level": "medium",
            "trace": self._append_trace(
                state,
                "travel_react_agent",
                "max_iterations",
                {"attempt": attempt},
            ),
        }
        return Command(update=update, goto="policy_validator")

    async def _policy_validator(
        self, state: TravelGraphState
    ) -> Command[Literal["travel_retry_router"]]:
        draft = self._build_booking_draft(state)
        validation = self._build_policy_validation(state, draft)
        validation["attempt"] = int(state.get("travel_attempt") or 1)
        return Command(
            update={
                "booking_draft": draft,
                "policy_validation": validation,
                "trace": self._append_trace(
                    state,
                    "policy_validator",
                    validation.get("status", "checked"),
                    {
                        "violations": len(validation.get("violations") or []),
                        "warnings": len(validation.get("warnings") or []),
                    },
                ),
            },
            goto="travel_retry_router",
        )

    async def _travel_retry_router(
        self, state: TravelGraphState
    ) -> Command[Literal["travel_react_agent", "approval_processor"]]:
        validation = state.get("policy_validation") or {}
        status = str(validation.get("status") or "needs_review")
        retry_count = int(state.get("travel_retry_count") or 0)
        max_retries = max(0, int(settings.travel_validation_max_retries))
        retry_reasons = self._travel_retry_reasons(
            state,
            state.get("booking_draft"),
            validation,
        )

        if status == "passed" and not retry_reasons:
            return Command(
                update={
                    "travel_retry_feedback": None,
                    "travel_retry_exhausted": False,
                    "trace": self._append_trace(
                        state,
                        "travel_retry_router",
                        "validation_passed",
                        {"attempt": state.get("travel_attempt", 1)},
                    ),
                },
                goto="approval_processor",
            )

        if retry_reasons and retry_count < max_retries:
            next_retry_count = retry_count + 1
            feedback = {
                "previous_attempt": state.get("travel_attempt", retry_count + 1),
                "validation_status": status,
                "reasons": retry_reasons,
                "violations": validation.get("violations") or [],
                "warnings": validation.get("warnings") or [],
            }
            return Command(
                update={
                    "travel_retry_count": next_retry_count,
                    "travel_retry_feedback": feedback,
                    "travel_retry_exhausted": False,
                    "trace": self._append_trace(
                        state,
                        "travel_retry_router",
                        "retry_scheduled",
                        {
                            "retry": next_retry_count,
                            "max_retries": max_retries,
                            "reasons": retry_reasons,
                        },
                    ),
                },
                goto="travel_react_agent",
            )

        exhausted = bool(retry_reasons and retry_count >= max_retries and retry_count > 0)
        answer = state.get("answer", "")
        summary = validation.get("summary")
        if summary:
            answer = f"{answer}\n\n合规校验：{summary}"
        if exhausted:
            answer = (
                f"{answer}\n\n自动重试：已完成 {retry_count} 次合规重试，仍未找到满足当前约束的完整方案，"
                "已停止自动尝试并转人工审核。"
            )
        validation = dict(validation)
        if retry_reasons and status == "passed":
            status = "needs_review"
            validation["status"] = status
            validation["summary"] = "候选完整性检查未通过，自动重试后仍需人工复核。"
        validation["retry"] = {
            "count": retry_count,
            "max_retries": max_retries,
            "exhausted": exhausted,
            "reasons": retry_reasons,
        }
        manual_review_required = bool(
            exhausted
            or status == "failed"
            or (state.get("booking_draft") is not None and status == "needs_review")
        )
        return Command(
            update={
                "answer": answer,
                "policy_validation": validation,
                "risk_level": "high" if manual_review_required else "medium",
                "travel_retry_exhausted": exhausted,
                "travel_retry_feedback": {
                    "validation_status": status,
                    "reasons": retry_reasons,
                    "violations": validation.get("violations") or [],
                    "warnings": validation.get("warnings") or [],
                },
                "trace": self._append_trace(
                    state,
                    "travel_retry_router",
                    "retry_exhausted" if exhausted else "manual_review_required",
                    {
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                        "status": status,
                        "reasons": retry_reasons,
                    },
                ),
            },
            goto="approval_processor",
        )

    async def _approval_processor(
        self, state: TravelGraphState
    ) -> Command[Literal["response_reviewer", "grounding_verifier"]]:
        draft = state.get("booking_draft") or self._build_booking_draft(state)
        form = self._build_approval_form(state, draft)
        validation = state.get("policy_validation") or {}
        force_manual_review = bool(
            state.get("travel_retry_exhausted")
            or validation.get("status") == "failed"
            or (draft is not None and validation.get("status") == "needs_review")
        )
        if force_manual_review:
            form = self._force_manual_review_form(state, form, draft, validation)
        answer = state.get("answer", "")
        risk_level = "high" if form and form.get("required") else state.get("risk_level") or "low"
        if draft:
            answer = (
                f"{answer}\n\n预订草稿：已生成 booking_draft（状态：{draft.get('status')}），"
                "当前仅用于用户确认、差标复核和审批流转，不会自动下单或付款。"
            )
        if form and form.get("required"):
            answer = (
                f"{answer}\n\n审批提示：该行程存在需人工确认/审批的风险点，"
                "已生成 approval_form，可提交主管或 OA 系统复核后再继续预订。"
            )
        update = {
            "answer": answer,
            "booking_draft": draft,
            "approval_form": form,
            "risk_level": risk_level,
            "trace": self._append_trace(
                state,
                "approval_processor",
                "approval_required" if form and form.get("required") else "not_required",
            ),
        }
        return Command(update=update, goto=self._route_after_answer({**state, **update}))

    async def _grounding_verifier(
        self, state: TravelGraphState
    ) -> Command[Literal["rag_self_corrector", "response_finalizer"]]:
        if state.get("answer_mode") != "rag_grounded":
            return Command(update={"verification": {"passed": None, "reason": "not_applicable", "status": "skipped", "claim_evidence_map": []}}, goto="response_finalizer")
        evidence, raw_draft = state.get("rag_evidence"), state.get("rag_draft")
        if not evidence or not raw_draft:
            return self._rag_failure(state, "verification", "missing_evidence_state")
        diagnostics: list[dict[str, Any]] = []
        try:
            verification, usage = await self._evidence_qa.review(
                state["rag_question"], EvidencePack.model_validate(evidence["pack"]), evidence["facts"],
                AnswerDraft.model_validate(raw_draft), state.get("citations") or [],
                task_request=state.get("rag_task_request", ""),
                diagnostics=diagnostics,
            )
        except Exception as exc:
            return self._rag_verified_subset(state, "verification_unavailable:" + type(exc).__name__, exception=exc, diagnostics=diagnostics)
        update: dict[str, Any] = {
            "verification": verification,
            "claim_evidence_map": verification.get("claim_evidence_map") or [],
            "usage": self._sum_usage(state.get("usage"), usage),
            "rag_stages": self._rag_stage(state, "verification", {"attempt": int(state.get("rag_correction_count") or 0), **verification, **diagnostic_summary(diagnostics)}),
            "trace": self._append_trace(
                state,
                "grounding_verifier",
                "passed" if verification["passed"] else "flagged",
                {
                    "failed_claim_count": sum(c["status"] != "supported" for c in verification["claim_evidence_map"]),
                    "correction_count": int(state.get("rag_correction_count") or 0),
                },
            ),
        }
        if verification["passed"]:
            update["risk_level"] = "low" if verification["question_answered"] else "medium"
            return Command(update=update, goto="response_finalizer")
        correction_count = int(state.get("rag_correction_count") or 0)
        if correction_count < 1:
            update["risk_level"] = "medium"
            return Command(update=update, goto="rag_self_corrector")
        command = self._rag_verified_subset({**state, **update}, "correction_exhausted")
        return Command(update={**update, **command.update}, goto="response_finalizer")

    def _rag_verified_subset(self, state: TravelGraphState, reason: str, *, exception: Exception | None = None, diagnostics: list[dict[str, Any]] | None = None) -> Command[Literal["response_finalizer"]]:
        """Retain only reviewed claims; failed claims never appear in user-facing fallback."""
        mapping = (state.get("verification") or {}).get("claim_evidence_map") or []
        supported = {c["claim_id"] for c in mapping if c.get("status") == "supported"}
        evidence, raw = state.get("rag_evidence"), state.get("rag_draft")
        kept = [c for c in (raw or {}).get("claims", []) if c["id"] in supported]
        answer = "当前未能完成全部结论核验，暂时无法提供完整答复。请稍后重试或补充适用条件。"
        status = getattr(exception, "status_code", None)
        detail = {"upstream_status": status} if isinstance(status, int) else {}
        detail.update(diagnostic_summary(diagnostics or []))
        if isinstance(exception, EvidenceError):
            detail["validation_error"] = exception.diagnostic()
        if status == 402:
            answer = "模型服务返回402（账户余额或计费状态异常），暂时无法完成全部结论核验，请检查模型服务账户。"
        if kept and evidence:
            answer = render_draft(AnswerDraft(claims=kept), evidence["facts"], state.get("citations") or []) + "\n\n" + answer
        final_map = [c for c in mapping if c.get("claim_id") in supported]
        verification = {
            **(state.get("verification") or {}), "passed": bool(kept), "reason": reason,
            "question_answered": False, "claim_evidence_map": final_map,
            "finalization": "supported_claims_only" if kept else "no_verified_claims",
            **detail,
        }
        return Command(update={
            "answer": answer, "answer_mode": "rag_grounded" if kept else "llm_fallback",
            "risk_level": "medium", "verification": verification, "claim_evidence_map": final_map,
            "rag_stages": self._rag_stage(state, "finalization", {"reason": reason, "kept_claim_ids": sorted(supported), **detail}),
            "usage": self._sum_usage(state.get("usage"), getattr(exception, "usage", None)) if isinstance(exception, EvidenceError) else state.get("usage"),
        }, goto="response_finalizer")

    async def _general_responder(
        self, state: TravelGraphState
    ) -> Command[Literal["response_reviewer", "grounding_verifier"]]:
        resp = await self._llm.chat_completion(state["openai_messages"], temperature=0.2)
        update = {
            "answer": resp.choices[0].message.content or "",
            "usage": self._usage_dict(resp),
            "trace": self._append_trace(state, "general_responder", "answered"),
        }
        return Command(update=update, goto=self._route_after_answer({**state, **update}))

    async def _rag_self_corrector(
        self, state: TravelGraphState
    ) -> Command[Literal["grounding_verifier", "response_finalizer"]]:
        evidence = state["rag_evidence"]
        diagnostics: list[dict[str, Any]] = []
        try:
            draft, usage = await self._evidence_qa.correct(
                state["rag_question"], EvidencePack.model_validate(evidence["pack"]), evidence["facts"],
                AnswerDraft.model_validate(state["rag_draft"]), state["verification"], state.get("citations") or [],
                task_request=state.get("rag_task_request", ""),
                diagnostics=diagnostics,
            )
            corrected = render_draft(draft, evidence["facts"], state.get("citations") or [])
        except Exception as exc:
            return self._rag_verified_subset(state, "correction_failed:" + type(exc).__name__, exception=exc, diagnostics=diagnostics)
        correction_count = int(state.get("rag_correction_count") or 0) + 1
        return Command(
            update={
                "answer": corrected, "rag_draft": draft.model_dump(mode="json"), "rag_correction_count": correction_count,
                "usage": self._sum_usage(state.get("usage"), usage),
                "rag_stages": self._rag_stage(state, "correction", {"answer": corrected, "claims": draft.model_dump(mode="json")["claims"], **diagnostic_summary(diagnostics)}),
                "trace": self._append_trace(
                    state, "rag_self_corrector", "corrected", {"correction_count": correction_count},
                ),
            },
            goto="grounding_verifier",
        )

    async def _response_reviewer(
        self, state: TravelGraphState
    ) -> Command[Literal["grounding_verifier"]]:
        if state.get("rag_evidence"):
            return Command(update={"reflection_notes": "evidence_review_delegated"}, goto="grounding_verifier")
        answer = state.get("answer", "")
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是企业差旅回复质检员。检查答案是否遗漏日期、城市、金额、"
                    "差标免责声明或审批风险。若需要修改，直接给出修订后的最终答复。"
                ),
            },
            {"role": "user", "content": f"原答复：\n{answer}"},
        ]
        resp = await self._llm.chat_completion(prompt, temperature=0.0)
        revised = resp.choices[0].message.content or answer
        return Command(
            update={
                "answer": revised,
                "reflection_notes": "response_reviewer_applied",
                "usage": self._usage_dict(resp),
                "trace": self._append_trace(state, "response_reviewer", "revised"),
            },
            goto="grounding_verifier",
        )

    async def _response_finalizer(self, state: TravelGraphState) -> dict[str, Any]:
        answer = state.get("answer", "")
        if state.get("route") not in {"multi_task", "clarification"}:
            finished = await self._finish_task(state)
            state = {**state, "answer": finished["answer"], "human_review": finished["human_review"]}
            answer = state["answer"]
        session_id = state.get("session_id")
        booking_draft = state.get("booking_draft")
        if session_id:
            await self._save_session_messages(
                session_id,
                (state.get("conversation_messages") or state.get("effective_messages", state["messages"]))
                + [ChatMessage(role=MessageRole.ASSISTANT, content=answer)],
            )
        if booking_draft:
            await self._save_booking_draft(session_id, booking_draft)
        human_reviews = []
        # Publish only after the pending response is in history. Children prepare
        # snapshots; the parent stores tickets once, without leaking private snapshots.
        for container in [state, *(state.get("task_results") or [])]:
            prepared = container.get("human_review")
            if not prepared:
                continue
            try:
                summary = await HumanReviewStore(self._redis).create(session_id or "", prepared)
            except Exception:
                summary = {"review_id": None, "status": "submission_failed", "reasons": prepared["reasons"], "task_id": prepared.get("task_id")}
                answer += "\n\n人工审核单提交失败，当前审核队列不可用，请联系制度负责人；尚未形成最终答复。"
            container["human_review"] = summary
            human_reviews.append(summary)
        if human_reviews and any(item["status"] == "submission_failed" for item in human_reviews) and session_id:
            await self._save_session_messages(session_id, (state.get("conversation_messages") or state.get("effective_messages", state["messages"])) + [ChatMessage(role=MessageRole.ASSISTANT, content=answer)])
        for result in state.get("task_results") or []:
            if result.get("booking_draft"):
                # Each draft is addressable by id; do not arbitrarily choose a "latest"
                # draft for a conversation containing several independent plans.
                await self._save_booking_draft(None, result["booking_draft"])
        return {
            "response": {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": self._llm.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}],
                "usage": self._sum_usage(state.get("planner_usage"), state.get("usage")),
                "metadata": {
                    "orchestrator": "langgraph",
                    "intent": state.get("intent"),
                    "route": state.get("route"),
                    "task_results": state.get("task_results") or [],
                    "tool_trace": state.get("tool_trace") or [],
                    "execution_plan": state.get("execution_plan"),
                    "policy_constraints": state.get("policy_constraints"),
                    "policy_validation": state.get("policy_validation"),
                    "travel_retry": {
                        "attempts": state.get("travel_attempt", 0),
                        "retry_count": state.get("travel_retry_count", 0),
                        "exhausted": bool(state.get("travel_retry_exhausted")),
                        "feedback": state.get("travel_retry_feedback"),
                    },
                    "reflection": state.get("reflection_notes"),
                    "trace": self._append_trace(state, "response_finalizer", "completed"),
                    "citations": state.get("citations") or [],
                    "approval_form": state.get("approval_form"),
                    "booking_draft": booking_draft,
                    "risk_level": state.get("risk_level"),
                    "answer_mode": state.get("answer_mode"),
                    "verification": state.get("verification"),
                    "claim_evidence_map": state.get("claim_evidence_map") or [],
                    "rag_correction_count": int(state.get("rag_correction_count") or 0),
                    "rag_evidence": state.get("rag_evidence"),
                    "rag_stages": self._rag_stage(state, "final", {"answer": answer, "answer_mode": state.get("answer_mode")}) if state.get("rag_stages") else [],
                    "human_reviews": human_reviews,
                    "answer_status": "review_submission_failed" if any(item["status"] == "submission_failed" for item in human_reviews) else "pending_human_review" if human_reviews else "automatic",
                    "memory": {
                        "current_facts": state.get("current_facts") or [],
                        "long_term_count": len(state.get("long_term_memories") or []),
                    },
                },
            }
        }

    async def run_completion(
        self,
        messages: list[ChatMessage],
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        session_id = session_id or str(uuid.uuid4())
        result = await self._graph.ainvoke(
            {"messages": messages, "session_id": session_id, "user_id": user_id},
            config={"recursion_limit": 30 + 5 * max(0, settings.travel_validation_max_retries)},
        )
        result["response"]["session_id"] = session_id
        return result["response"]

    async def stream_completion(self, messages: list[ChatMessage], *, session_id: str | None = None, user_id: str | None = None):
        result = await self.run_completion(messages, session_id=session_id, user_id=user_id)
        text = result["choices"][0]["message"]["content"]
        for index, character in enumerate(text):
            yield StreamChunk(type=StreamChunkType.CONTENT, index=index, delta=character)
        yield StreamChunk(type=StreamChunkType.DONE, index=len(text), finish_reason="stop", session_id=result["session_id"], human_reviews=result["metadata"].get("human_reviews") or [], answer_status=result["metadata"].get("answer_status"))

    async def _save_booking_draft(
        self,
        session_id: str | None,
        booking_draft: dict[str, Any],
    ) -> None:
        if self._redis is None:
            return
        draft_id = booking_draft.get("draft_id")
        if not draft_id:
            return
        payload = json.dumps(booking_draft, ensure_ascii=False)
        ttl = settings.memory_session_ttl_seconds
        await self._redis.set(f"booking:draft:{draft_id}", payload, ex=ttl)
        if session_id:
            await self._redis.set(f"booking:session:{session_id}:latest", str(draft_id), ex=ttl)

    @staticmethod
    def _last_user_text(messages: list[ChatMessage]) -> str:
        for msg in reversed(messages):
            if msg.role is MessageRole.USER:
                return msg.content
        return messages[-1].content if messages else ""

    @staticmethod
    def _append_trace(
        state: TravelGraphState,
        node: str,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        trace = list(state.get("trace") or [])
        item: dict[str, Any] = {"node": node, "event": event}
        if data:
            item["data"] = data
        trace.append(item)
        return trace

    def _long_term_memory_key(self, owner: str) -> str:
        return f"memory:long:{owner}"

    async def _load_long_term_memories(self, owner: str | None) -> list[str]:
        if not owner or self._redis is None:
            return []
        raw = await self._redis.get(self._long_term_memory_key(owner))
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [str(item) for item in payload if isinstance(item, str)]

    async def _save_long_term_memories(
        self,
        owner: str | None,
        existing: list[str],
        new_items: list[str],
    ) -> list[str]:
        merged: list[str] = []
        for item in [*existing, *new_items]:
            normalized = item.strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        merged = merged[-20:]
        if owner and self._redis is not None:
            await self._redis.set(
                self._long_term_memory_key(owner),
                json.dumps(merged, ensure_ascii=False),
                ex=settings.memory_session_ttl_seconds * 30,
            )
        return merged

    @staticmethod
    def _extract_long_term_facts(text: str) -> list[str]:
        patterns = [
            r"(?:我的职位|我的职级|我是)\s*(?:是|为|:|：)?\s*([A-Za-z0-9_\-\u4e00-\u9fa5]{2,20})",
            r"(?:我喜欢|我偏好|我希望|我倾向于|默认给我|以后默认)\s*([^。；;\n]{2,60})",
            r"(?:常用出发地|常驻城市|我在|我来自)\s*(?:是|为|:|：)?\s*([\u4e00-\u9fa5A-Za-z]{2,20})",
        ]
        facts: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = match.group(1).strip()
                if value and value not in facts:
                    facts.append(value)
        if "不记住" in text or "不要记住" in text:
            return []
        return facts[:5]

    @staticmethod
    def _build_memory_context(current_facts: list[str], long_term_memories: list[str]) -> str:
        lines = [
            "记忆与上下文优先级规则：公司制度/RAG 引用 > 当前用户明确输入 > 短期会话记忆 > 长期个人偏好 > 模型常识。",
            "如果长期偏好与当前用户输入或公司制度冲突，必须忽略长期偏好。",
        ]
        if current_facts:
            lines.append("当前用户明确输入事实：" + "；".join(current_facts))
        if long_term_memories:
            lines.append("长期个人偏好/事实：" + "；".join(long_term_memories[-10:]))
        return "\n".join(lines)

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        if not text:
            return {}
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


    async def _extract_policy_constraints(
        self,
        user_text: str,
        citations: list[dict[str, Any]],
        memory_context: str,
    ) -> dict[str, Any]:
        heuristic = self._heuristic_policy_constraints(citations)
        if not citations:
            return heuristic
        context = "\n\n".join(
            f"[{index}] {item.get('title') or '未命名'}\n{item.get('content') or ''}"
            for index, item in enumerate(citations[:5], start=1)
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是企业差旅制度约束抽取器。只能根据参考资料抽取 JSON，不要编造。"
                    "输出字段：source, constraints, confidence, notes。constraints 可包含 "
                    "hotel_limit_cny, advance_booking_days, approval_threshold_cny, cabin_limit, train_seat。"
                    "没有依据的字段不要输出。"
                ),
            },
            {
                "role": "user",
                "content": f"{memory_context}\n\n参考资料：\n{context}\n\n用户问题：{user_text}",
            },
        ]
        try:
            resp = await self._llm.chat_completion(prompt, temperature=0.0)
            parsed = self._parse_json_object(resp.choices[0].message.content or "")
        except Exception:  # noqa: BLE001
            parsed = {}
        if not parsed:
            return heuristic
        constraints = parsed.get("constraints") if isinstance(parsed.get("constraints"), dict) else {}
        merged = dict(heuristic.get("constraints") or {})
        merged.update({k: v for k, v in constraints.items() if v not in (None, "")})
        notes = parsed.get("notes") if isinstance(parsed.get("notes"), list) else heuristic.get("notes", [])
        return {
            "source": parsed.get("source") or "rag_extracted",
            "constraints": merged,
            "confidence": float(parsed.get("confidence") or heuristic.get("confidence") or 0.0),
            "notes": [str(item) for item in notes],
        }

    @staticmethod
    def _heuristic_policy_constraints(citations: list[dict[str, Any]]) -> dict[str, Any]:
        text = "\n".join(str(item.get("content") or "") for item in citations)
        constraints: dict[str, Any] = {}
        hotel_match = re.search(r"(?:酒店标准|酒店|住宿)[^0-9]{0,20}(?:不超过|≤|<=|上限)?\s*([0-9]{3,5})\s*(?:CNY|元)?", text)
        if hotel_match:
            constraints["hotel_limit_cny"] = int(hotel_match.group(1))
        advance_match = re.search(r"提前\s*([0-9]{1,2})\s*天", text)
        if advance_match:
            constraints["advance_booking_days"] = int(advance_match.group(1))
        approval_match = re.search(r"(?:超过|大于|高于)\s*([0-9]{4,6})\s*(?:CNY|元)?.{0,12}审批", text)
        if approval_match:
            constraints["approval_threshold_cny"] = int(approval_match.group(1))
        if "经济舱" in text:
            constraints["cabin_limit"] = "economy"
        if "二等座" in text:
            constraints["train_seat"] = "二等座"
        return {
            "source": "rag_heuristic" if citations else "unavailable",
            "constraints": constraints,
            "confidence": 0.7 if constraints else 0.0,
            "notes": [] if constraints else ["未从知识库中抽取到结构化制度约束。"],
        }

    @staticmethod
    def _input_guardrail_message(text: str) -> str | None:
        lowered = text.lower()
        if any(x in lowered for x in ("ignore previous", "忽略以上", "绕过审批", "伪造发票")):
            return "当前请求存在合规风险，已停止执行。请提交真实、合规的差旅需求。"
        booking_keywords = ("预订", "下单", "出票", "订票", "确认购买")
        question_indicators = (
            "几天",
            "多少",
            "要求",
            "标准",
            "可以",
            "吗",
            "制度",
            "政策",
            "怎么安排",
        )
        if any(k in text for k in booking_keywords) and not any(q in text for q in question_indicators):
            has_date = bool(re.search(r"\d{4}-\d{1,2}-\d{1,2}|明天|后天|下周|周[一二三四五六日天]", text))
            has_route = bool(re.search(r"从.+(到|去).+", text))
            if not has_date or not has_route:
                return "预订前需要补充出发日期、出发城市、目的城市和员工职级；当前信息不足，暂不执行预订。"
        return None

    def _knowledge_retrieval_available(self) -> bool:
        if self._rag_retriever is not None:
            return bool(getattr(self._rag_retriever, "connected", False))
        return bool(
            self._document_store is not None
            and getattr(self._document_store, "connected", False)
        )

    async def _retrieve_knowledge(self, query: str) -> list[dict[str, Any]]:
        vector = await EmbeddingService().embed_text(query)
        if self._rag_retriever is not None:
            if not getattr(self._rag_retriever, "connected", False):
                raise RuntimeError("hybrid retrieval dependencies unavailable")
            return await self._rag_retriever.retrieve(
                query,
                vector,
                keyword_top_k=settings.rag_keyword_top_k,
                vector_top_k=settings.rag_vector_top_k,
                rrf_k=settings.rag_rrf_k,
                candidate_top_k=settings.rag_fused_top_k,
                final_top_k=settings.rag_final_top_k,
            )
        if self._document_store is None or not getattr(self._document_store, "connected", False):
            raise RuntimeError("knowledge retrieval unavailable")
        return self._document_store.search(vector, top_k=settings.rag_final_top_k)

    @staticmethod
    def _citation_from_hit(hit: dict[str, Any]) -> dict[str, Any]:
        score = hit.get("rerank_score")
        if score is None:
            score = hit.get("rrf_score", hit.get("score"))
        return {
            "chunk_id": str(hit.get("chunk_id") or hit.get("id") or "") or None,
            "title": hit.get("title"),
            "doc_type": hit.get("doc_type"),
            "content": str(hit.get("content") or hit.get("text") or "")[:settings.rag_evidence_chunk_max_chars],
            "score": score,
            "vector_score": hit.get("vector_score"),
            "keyword_score": hit.get("keyword_score"),
            "rrf_score": hit.get("rrf_score"),
            "rerank_score": hit.get("rerank_score"),
            "vector_rank": hit.get("vector_rank"),
            "keyword_rank": hit.get("keyword_rank"),
            "metadata": dict(hit.get("metadata") or {}),
        }

    @staticmethod
    def _retrieval_trace(citations: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "keyword_top_k": settings.rag_keyword_top_k,
            "vector_top_k": settings.rag_vector_top_k,
            "rrf_k": settings.rag_rrf_k,
            "candidate_top_k": settings.rag_fused_top_k,
            "final_top_k": settings.rag_final_top_k,
            "results": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "vector_rank": item.get("vector_rank"),
                    "keyword_rank": item.get("keyword_rank"),
                    "rrf_score": item.get("rrf_score"),
                    "rerank_score": item.get("rerank_score"),
                }
                for item in citations
            ],
        }

    @staticmethod
    def _current_attempt_tool_trace(state: TravelGraphState) -> list[dict[str, Any]]:
        tool_trace = [item for item in state.get("tool_trace") or [] if isinstance(item, dict)]
        if not any("attempt" in item for item in tool_trace):
            return tool_trace
        attempt = int(state.get("travel_attempt") or 1)
        return [item for item in tool_trace if item.get("attempt") == attempt]

    @staticmethod
    def _travel_retry_reasons(
        state: TravelGraphState,
        booking_draft: dict[str, Any] | None,
        validation: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []

        def add(reason: str) -> None:
            text = reason.strip()
            if text and text not in reasons:
                reasons.append(text)

        tool_trace = LangGraphTravelOrchestrator._current_attempt_tool_trace(state)
        if not tool_trace:
            add("本轮未执行任何旅行查询或推荐工具，无法形成可校验候选")

        for item in tool_trace:
            tool_name = str(item.get("tool") or "")
            output = item.get("output")
            if not isinstance(output, str):
                if tool_name.startswith("search_") or tool_name == "recommend_travel_options":
                    add(f"{tool_name} 未返回可解析结果")
                continue
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            mode = str(payload.get("mode") or "")
            if payload.get("error"):
                add(f"{tool_name or mode} 查询失败：{payload['error']}")
            if mode in {"flight", "hotel", "train"}:
                results = payload.get("results")
                if not isinstance(results, list) or not any(
                    isinstance(row, dict) for row in results
                ):
                    add(f"{mode} 查询未找到可用候选")
            elif mode == "travel_recommendation":
                flights = [row for row in payload.get("flights") or [] if isinstance(row, dict)]
                trains = [row for row in payload.get("trains") or [] if isinstance(row, dict)]
                hotels = [row for row in payload.get("hotels") or [] if isinstance(row, dict)]
                raw_query = payload.get("query")
                query: dict[str, Any] = dict(raw_query) if isinstance(raw_query, dict) else {}
                include_trains = bool(query.get("include_trains", True))
                if not flights and (not include_trains or not trains):
                    add("综合推荐未找到可用的航班或火车交通候选")
                if not hotels:
                    add("综合推荐未找到可用酒店候选")

        retryable_checks = {"酒店差标", "航班舱位", "高铁/火车席别"}
        for check in validation.get("checks") or []:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name") or "")
            status = str(check.get("status") or "")
            detail = str(check.get("detail") or "")
            if name in retryable_checks and status in {"failed", "unknown"}:
                add(detail or f"{name}未通过")
            elif name == "审批阈值" and status == "unknown" and booking_draft:
                add(detail or "候选价格不完整，无法校验审批阈值")
        for violation in validation.get("violations") or []:
            text = str(violation).strip()
            if any(keyword in text for keyword in ("酒店", "舱位", "席别", "超出差标", "超过差标", "不符合")):
                add(text)
        return reasons

    @staticmethod
    def _force_manual_review_form(
        state: TravelGraphState,
        form: dict[str, Any] | None,
        booking_draft: dict[str, Any] | None,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        draft = booking_draft or {}
        slots_payload = state.get("active_task") or state.get("execution_plan") or {}
        slots = slots_payload.get("slots") if isinstance(slots_payload, dict) else {}
        slots = slots if isinstance(slots, dict) else {}
        result = dict(form or {})
        warnings = [
            str(item)
            for item in result.get("policy_warnings") or []
            if str(item).strip()
        ]
        for item in [*(validation.get("violations") or []), *(validation.get("warnings") or [])]:
            text = str(item).strip()
            if text and text not in warnings:
                warnings.append(text)
        if state.get("travel_retry_exhausted"):
            text = "自动合规重试已耗尽，仍未找到满足当前约束的完整方案"
            if text not in warnings:
                warnings.append(text)
        result.update(
            {
                "required": True,
                "status": "pending_human_approval",
                "employee_id": result.get("employee_id")
                or draft.get("employee_id")
                or slots.get("employee_id"),
                "grade": result.get("grade") or draft.get("grade") or slots.get("grade"),
                "origin_city": result.get("origin_city")
                or draft.get("origin_city")
                or slots.get("origin_city"),
                "destination_city": result.get("destination_city")
                or draft.get("destination_city")
                or slots.get("destination_city"),
                "departure_date": result.get("departure_date")
                or draft.get("departure_date")
                or slots.get("departure_date"),
                "return_date": result.get("return_date")
                or draft.get("return_date")
                or slots.get("return_date"),
                "estimated_total_cny": result.get("estimated_total_cny")
                or draft.get("estimated_total_cny"),
                "reason": result.get("reason") or draft.get("purpose") or slots.get("purpose"),
                "policy_warnings": warnings,
            }
        )
        return result

    @staticmethod
    def _build_approval_form(
        state: TravelGraphState,
        booking_draft: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        tool_trace = LangGraphTravelOrchestrator._current_attempt_tool_trace(state)
        if not tool_trace:
            return None
        args: dict[str, Any] = {}
        output = ""
        warnings: list[str] = []
        for item in tool_trace:
            if not isinstance(item, dict):
                continue
            raw_args = item.get("arguments")
            if isinstance(raw_args, str):
                try:
                    args.update(json.loads(raw_args))
                except json.JSONDecodeError:
                    pass
            if isinstance(item.get("output"), str):
                output = item["output"]
                warnings.extend(LangGraphTravelOrchestrator._extract_policy_warnings(output))

        amount = None
        amount_match = re.search(r"预估总额：([0-9.]+)\s*CNY", output)
        if amount_match:
            amount = amount_match.group(1)
        if amount is None and args.get("estimated_total_cny") is not None:
            amount = str(args["estimated_total_cny"])
        if amount is None and booking_draft and booking_draft.get("estimated_total_cny") is not None:
            amount = str(booking_draft["estimated_total_cny"])

        requires = any(
            any(k in warning for k in ("超出", "超过", "特批", "审批", "低于提前"))
            for warning in warnings
        )
        if booking_draft:
            requires = requires or bool(booking_draft.get("approval_required"))
            for warning in booking_draft.get("policy_warnings") or []:
                if warning and warning not in warnings:
                    warnings.append(str(warning))
        if amount is not None:
            try:
                requires = requires or float(amount) > 5000
            except ValueError:
                pass
        return {
            "required": requires,
            "status": "pending_human_approval" if requires else "not_required",
            "employee_id": args.get("employee_id") or (booking_draft or {}).get("employee_id"),
            "grade": args.get("grade") or (booking_draft or {}).get("grade"),
            "origin_city": args.get("origin_city") or (booking_draft or {}).get("origin_city"),
            "destination_city": args.get("destination_city") or (booking_draft or {}).get("destination_city"),
            "departure_date": args.get("departure_date") or (booking_draft or {}).get("departure_date"),
            "return_date": args.get("return_date") or (booking_draft or {}).get("return_date"),
            "estimated_total_cny": amount,
            "reason": args.get("purpose") or (booking_draft or {}).get("purpose"),
            "policy_warnings": warnings,
        }

    @staticmethod
    def _build_booking_draft(state: TravelGraphState) -> dict[str, Any] | None:
        tool_trace = LangGraphTravelOrchestrator._current_attempt_tool_trace(state)
        if not tool_trace:
            return None

        args: dict[str, Any] = {}
        recommendation_payload: dict[str, Any] | None = None
        policy_warnings: list[str] = []
        for item in tool_trace:
            if not isinstance(item, dict):
                continue
            raw_args = item.get("arguments")
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                    if isinstance(parsed_args, dict):
                        args.update(parsed_args)
                except json.JSONDecodeError:
                    pass
            output = item.get("output")
            if not isinstance(output, str):
                continue
            policy_warnings.extend(
                warning
                for warning in LangGraphTravelOrchestrator._extract_policy_warnings(output)
                if warning not in policy_warnings
            )
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("mode") == "travel_recommendation":
                recommendation_payload = payload
                query = payload.get("query")
                if isinstance(query, dict):
                    args.update(query)
                for warning in payload.get("policy_warnings") or []:
                    text = str(warning).strip()
                    if text and text not in policy_warnings:
                        policy_warnings.append(text)

        if recommendation_payload is None:
            return None

        rec = recommendation_payload.get("recommendation")
        rec = rec if isinstance(rec, dict) else {}
        flight = rec.get("flight") if isinstance(rec.get("flight"), dict) else {}
        train = rec.get("train") if isinstance(rec.get("train"), dict) else {}
        hotel = rec.get("hotel") if isinstance(rec.get("hotel"), dict) else {}
        if not flight:
            flight = LangGraphTravelOrchestrator._first_dict(recommendation_payload.get("flights"))
        if not train:
            train = LangGraphTravelOrchestrator._first_dict(recommendation_payload.get("trains"))
        if not hotel:
            hotel = LangGraphTravelOrchestrator._first_dict(recommendation_payload.get("hotels"))

        policy_checks = recommendation_payload.get("policy_checks")
        policy_checks = policy_checks if isinstance(policy_checks, dict) else {}
        price_snapshot = LangGraphTravelOrchestrator._booking_price_snapshot(flight, train, hotel)
        estimated_total = LangGraphTravelOrchestrator._estimate_booking_total(price_snapshot, flight, train)
        approval_required = LangGraphTravelOrchestrator._booking_requires_approval(policy_warnings, estimated_total)
        created_at = datetime.now(timezone.utc)
        confirmation_items = LangGraphTravelOrchestrator._booking_confirmation_items(
            flight,
            train,
            hotel,
            policy_warnings,
            estimated_total,
        )
        return {
            "draft_id": f"bd_{uuid.uuid4().hex[:12]}",
            "status": "approval_pending" if approval_required else "draft_created",
            "employee_id": args.get("employee_id"),
            "grade": args.get("grade"),
            "origin_city": args.get("origin_city"),
            "destination_city": args.get("destination_city"),
            "departure_date": args.get("departure_date"),
            "return_date": args.get("return_date"),
            "purpose": args.get("purpose"),
            "recommended_flight": LangGraphTravelOrchestrator._compact_flight(flight),
            "recommended_train": LangGraphTravelOrchestrator._compact_train(train),
            "recommended_hotel": LangGraphTravelOrchestrator._compact_hotel(hotel),
            "price_snapshot": price_snapshot,
            "estimated_total_cny": estimated_total,
            "policy_checks": policy_checks,
            "policy_warnings": policy_warnings,
            "approval_required": approval_required,
            "confirmation_items": confirmation_items,
            "next_action": "submit_for_approval" if approval_required else "confirm_draft",
            "created_at": created_at.isoformat(),
            "expires_at": (created_at + timedelta(minutes=30)).isoformat(),
        }

    @staticmethod
    def _first_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    return item
        return {}

    @staticmethod
    def _compact_flight(item: dict[str, Any]) -> dict[str, Any]:
        if not item:
            return {}
        return {
            "flight_no": item.get("flight_no"),
            "carrier": item.get("carrier"),
            "origin": item.get("origin"),
            "destination": item.get("destination"),
            "depart_at": item.get("depart_at"),
            "arrive_at": item.get("arrive_at"),
            "cabin": item.get("cabin"),
            "price_cny": item.get("price_cny"),
            "booking_url": item.get("booking_url"),
        }

    @staticmethod
    def _compact_train(item: dict[str, Any]) -> dict[str, Any]:
        if not item:
            return {}
        return {
            "train_code": item.get("trainCode") or item.get("train_code"),
            "from_station": item.get("fromStation") or item.get("from_station"),
            "to_station": item.get("toStation") or item.get("to_station"),
            "depart_time": item.get("departTime") or item.get("depart_time"),
            "arrive_time": item.get("arriveTime") or item.get("arrive_time"),
            "duration": item.get("duration"),
            "second_class": item.get("ze"),
            "recommendation_score": item.get("recommendation_score"),
        }

    @staticmethod
    def _compact_hotel(item: dict[str, Any]) -> dict[str, Any]:
        if not item:
            return {}
        return {
            "name": item.get("name"),
            "city": item.get("city"),
            "poi_name": item.get("poi_name"),
            "nearby": item.get("nearby"),
            "check_in": item.get("check_in"),
            "check_out": item.get("check_out"),
            "nightly_cny": item.get("nightly_cny"),
            "total_cny": item.get("total_cny"),
            "detail_url": item.get("detail_url"),
        }

    @staticmethod
    def _decimal_text(value: Any) -> str | None:
        if value in (None, ""):
            return None
        match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
        return match.group(1) if match else None

    @staticmethod
    def _booking_price_snapshot(
        flight: dict[str, Any],
        train: dict[str, Any],
        hotel: dict[str, Any],
    ) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        flight_price = LangGraphTravelOrchestrator._decimal_text(flight.get("price_cny"))
        hotel_total = LangGraphTravelOrchestrator._decimal_text(hotel.get("total_cny"))
        hotel_nightly = LangGraphTravelOrchestrator._decimal_text(hotel.get("nightly_cny"))
        if flight_price:
            snapshot["flight_cny"] = flight_price
        if train:
            snapshot["train_cny"] = "待供应商确认"
        if hotel_nightly:
            snapshot["hotel_nightly_cny"] = hotel_nightly
        if hotel_total:
            snapshot["hotel_total_cny"] = hotel_total
        return snapshot

    @staticmethod
    def _estimate_booking_total(
        price_snapshot: dict[str, str],
        flight: dict[str, Any],
        train: dict[str, Any],
    ) -> str | None:
        if flight and train:
            return None
        total = 0.0
        has_value = False
        for key in ("flight_cny", "hotel_total_cny"):
            value = LangGraphTravelOrchestrator._decimal_text(price_snapshot.get(key))
            if value is None:
                continue
            total += float(value)
            has_value = True
        return f"{total:.2f}" if has_value else None

    @staticmethod
    def _booking_requires_approval(warnings: list[str], estimated_total: str | None) -> bool:
        requires = any(
            any(k in warning for k in ("超出", "超过", "特批", "审批", "低于提前"))
            for warning in warnings
        )
        if estimated_total is not None:
            try:
                requires = requires or float(estimated_total) > 5000
            except ValueError:
                pass
        return requires

    @staticmethod
    def _booking_confirmation_items(
        flight: dict[str, Any],
        train: dict[str, Any],
        hotel: dict[str, Any],
        warnings: list[str],
        estimated_total: str | None,
    ) -> list[str]:
        items = ["确认出行人实名信息、证件信息和联系方式"]
        if flight and train:
            items.append("确认本次交通方式选择航班还是高铁/火车")
        elif flight:
            items.append("确认航班库存、票价和退改规则")
        elif train:
            items.append("确认车次余票、席别和 12306 最终票价")
        if hotel:
            items.append("确认酒店位置、房型、价格和取消规则")
        if estimated_total is None:
            items.append("当前存在待确认价格项，需在供应商页面复核总价")
        if warnings:
            items.append("存在差标或审批风险，需先完成主管/OA 复核")
        return items

    @staticmethod
    def _build_policy_validation(
        state: TravelGraphState,
        booking_draft: dict[str, Any] | None,
    ) -> dict[str, Any]:
        constraints_payload = state.get("policy_constraints") or {}
        constraints = constraints_payload.get("constraints") if isinstance(constraints_payload, dict) else {}
        constraints = constraints if isinstance(constraints, dict) else {}
        warnings: list[str] = []
        violations: list[str] = []
        passed: list[str] = []
        checks: list[dict[str, str]] = []

        def add_check(name: str, status: str, detail: str) -> None:
            checks.append({"name": name, "status": status, "detail": detail})
            if status == "passed":
                passed.append(detail)
            elif status == "failed":
                violations.append(detail)
            elif status in {"warning", "unknown"}:
                warnings.append(detail)

        draft = booking_draft or {}
        hotel = draft.get("recommended_hotel") if isinstance(draft.get("recommended_hotel"), dict) else {}
        flight = draft.get("recommended_flight") if isinstance(draft.get("recommended_flight"), dict) else {}
        train = draft.get("recommended_train") if isinstance(draft.get("recommended_train"), dict) else {}

        hotel_limit = LangGraphTravelOrchestrator._decimal_text(constraints.get("hotel_limit_cny"))
        nightly = LangGraphTravelOrchestrator._decimal_text(hotel.get("nightly_cny"))
        if hotel_limit and nightly:
            if float(nightly) <= float(hotel_limit):
                add_check("酒店差标", "passed", f"酒店每晚 {nightly} CNY <= 制度上限 {hotel_limit} CNY")
            else:
                add_check("酒店差标", "failed", f"酒店每晚 {nightly} CNY 超过制度上限 {hotel_limit} CNY")
        elif hotel:
            add_check("酒店差标", "unknown", "酒店价格或制度上限不完整，需人工复核差标")

        cabin_limit = str(constraints.get("cabin_limit") or "").lower()
        cabin = str(flight.get("cabin") or "").lower()
        if cabin_limit and cabin:
            if cabin_limit in cabin or cabin in cabin_limit:
                add_check("航班舱位", "passed", f"航班舱位 {flight.get('cabin')} 符合制度约束")
            else:
                add_check(
                    "航班舱位",
                    "failed",
                    f"航班舱位 {flight.get('cabin')} 可能不符合制度约束 {constraints.get('cabin_limit')}",
                )

        train_seat = str(constraints.get("train_seat") or "")
        if train and train_seat:
            seat_value = str(train.get("second_class") or "")
            if train_seat == "二等座" and seat_value and seat_value not in {"无", "--", "-", "候补"}:
                add_check("高铁/火车席别", "passed", "高铁/火车二等座有票，符合常规差旅席别约束")
            else:
                add_check("高铁/火车席别", "unknown", f"高铁/火车席别需按制度约束 {train_seat} 复核")

        threshold = LangGraphTravelOrchestrator._decimal_text(constraints.get("approval_threshold_cny"))
        estimated = LangGraphTravelOrchestrator._decimal_text(draft.get("estimated_total_cny"))
        if threshold and estimated:
            if float(estimated) > float(threshold):
                add_check("审批阈值", "warning", f"预估总额 {estimated} CNY 超过审批阈值 {threshold} CNY，需提交审批")
            else:
                add_check("审批阈值", "passed", f"预估总额 {estimated} CNY 未超过审批阈值 {threshold} CNY")
        elif draft and estimated is None:
            add_check("审批阈值", "unknown", "当前存在待确认价格项，暂无法完成总额审批阈值校验")

        advance_days = LangGraphTravelOrchestrator._decimal_text(constraints.get("advance_booking_days"))
        departure = draft.get("departure_date")
        if advance_days and departure:
            try:
                departure_date = datetime.strptime(str(departure), "%Y-%m-%d").date()
                today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
                days_before = (departure_date - today).days
                required_days = int(float(advance_days))
                if days_before >= required_days:
                    add_check("提前预订", "passed", f"距出发 {days_before} 天，满足提前 {required_days} 天预订要求")
                else:
                    add_check("提前预订", "warning", f"距出发 {days_before} 天，低于提前 {required_days} 天预订要求")
            except ValueError:
                add_check("提前预订", "unknown", "出发日期格式无法解析，需人工复核提前预订要求")
        elif draft and advance_days:
            add_check("提前预订", "unknown", "缺少出发日期，无法校验提前预订要求")

        for item in LangGraphTravelOrchestrator._current_attempt_tool_trace(state):
            if not isinstance(item, dict) or not isinstance(item.get("output"), str):
                continue
            for warning in LangGraphTravelOrchestrator._extract_policy_warnings(item["output"]):
                text = warning.strip()
                if text and text not in warnings and text not in violations:
                    warnings.append(text)

        for warning in draft.get("policy_warnings") or []:
            text = str(warning).strip()
            if text and text not in warnings and text not in violations:
                if any(key in text for key in ("超出差标", "超过差标", "不符合")):
                    violations.append(text)
                else:
                    warnings.append(text)

        if violations:
            status = "failed"
            summary = "存在差标或审批风险，需先人工/OA 复核。"
        elif warnings or not constraints or not checks:
            status = "needs_review"
            summary = "部分制度约束或价格信息不足，建议人工复核后再预订。"
        else:
            status = "passed"
            summary = "已基于当前制度约束完成自动合规校验。"
        return {
            "status": status,
            "summary": summary,
            "checks": checks,
            "passed_checks": passed,
            "warnings": warnings,
            "violations": violations,
            "constraint_source": constraints_payload.get("source") if isinstance(constraints_payload, dict) else None,
            "constraint_confidence": constraints_payload.get("confidence") if isinstance(constraints_payload, dict) else None,
        }

    @staticmethod
    def _extract_policy_warnings(text: str) -> list[str]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}
        values = payload.get("policy_warnings") if isinstance(payload, dict) else None
        if isinstance(values, list):
            return [str(value).strip() for value in values if str(value).strip()]
        if "策略提示：" in text:
            tail = text.split("策略提示：", 1)[1]
            return [x.strip() for x in tail.split("；") if x.strip()]
        if "差标校验结果：" in text:
            return [
                line.removeprefix("-").strip()
                for line in text.splitlines()
                if line.strip().startswith("-")
            ]
        return []

    @staticmethod
    def _sum_usage(*items: dict[str, int] | None) -> dict[str, int] | None:
        values = [item for item in items if item is not None]
        if not values:
            return None
        return {key: sum(item.get(key, 0) or 0 for item in values)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")}

    @staticmethod
    def _usage_dict(resp: Any) -> dict[str, int]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
