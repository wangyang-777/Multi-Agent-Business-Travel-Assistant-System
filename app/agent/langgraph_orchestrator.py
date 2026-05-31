from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, TypedDict
from zoneinfo import ZoneInfo

from app.agent.orchestrator import (
    TravelOrchestrator,
    _to_openai_messages,
    _runtime_system_prompt,
    _travel_tools,
)
from app.config import settings
from app.core.intent.recognizer import IntentRecognizer, TravelIntent
from app.domain.schemas import ChatMessage, MessageRole
from app.services.embeddings import EmbeddingService

from langgraph.types import Command


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
    risk_level: str | None
    answer_mode: str | None
    verification: dict[str, Any] | None
    reflection_notes: str


class LangGraphTravelOrchestrator(TravelOrchestrator):
    """LangGraph-backed multi-agent orchestration with the legacy chat API shape."""

    def __init__(self, *args: Any, document_store: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._intent = IntentRecognizer()
        self._document_store = document_store
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(TravelGraphState)
        graph.add_node("context_agent", self._context_agent)
        graph.add_node("memory_fusion_agent", self._memory_fusion_agent)
        graph.add_node("guardrail_agent", self._guardrail_agent)
        graph.add_node("planner_agent", self._planner_agent)
        graph.add_node("intent_agent", self._intent_agent)
        graph.add_node("policy_reasoner_agent", self._policy_reasoner_agent)
        graph.add_node("rag_agent", self._rag_agent)
        graph.add_node("travel_react_agent", self._travel_react_agent)
        graph.add_node("policy_validator_agent", self._policy_validator_agent)
        graph.add_node("approval_agent", self._approval_agent)
        graph.add_node("general_agent", self._general_agent)
        graph.add_node("verification_agent", self._verification_agent)
        graph.add_node("reflection_agent", self._reflection_agent)
        graph.add_node("finalizer_agent", self._finalizer_agent)

        graph.add_edge(START, "context_agent")
        graph.add_edge("context_agent", "memory_fusion_agent")
        graph.add_edge("memory_fusion_agent", "guardrail_agent")
        graph.add_edge("verification_agent", "finalizer_agent")
        graph.add_edge("reflection_agent", "finalizer_agent")
        graph.add_edge("finalizer_agent", END)
        return graph.compile()

    async def _context_agent(self, state: TravelGraphState) -> dict[str, Any]:
        incoming = state["messages"]
        session_id = state.get("session_id")
        effective_messages = await self._effective_messages(incoming, session_id)
        messages = await self._maybe_summarize_thread(effective_messages)
        messages = self._trim_window(messages)
        openai_messages: list[dict[str, Any]] = [{"role": "system", "content": _runtime_system_prompt()}]
        openai_messages.extend(_to_openai_messages(messages))
        return {
            "effective_messages": effective_messages,
            "openai_messages": openai_messages,
            "tool_trace": [],
            "trace": self._append_trace(state, "context_agent", "prepared"),
            "citations": [],
            "approval_form": None,
            "booking_draft": None,
            "execution_plan": None,
            "policy_constraints": None,
            "policy_validation": None,
            "risk_level": "low",
            "answer_mode": None,
            "verification": None,
            "memory_context": "",
            "long_term_memories": [],
            "current_facts": [],
        }

    async def _memory_fusion_agent(self, state: TravelGraphState) -> dict[str, Any]:
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
                "memory_fusion_agent",
                "fused",
                {
                    "current_facts": len(current_facts),
                    "long_term_memories": len(stored_memories),
                    "owner": bool(memory_owner),
                },
            ),
        }

    async def _guardrail_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["planner_agent", "finalizer_agent"]]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        blocked = self._input_guardrail_message(text)
        trace = self._append_trace(
            state,
            "guardrail_agent",
            "blocked" if blocked else "passed",
        )
        if blocked:
            return Command(
                update={"answer": blocked, "trace": trace, "risk_level": "medium"},
                goto="finalizer_agent",
            )
        return Command(update={"trace": trace}, goto="planner_agent")

    async def _planner_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["intent_agent"]]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是企业商旅 Planner Agent。请把用户目标拆成结构化执行计划，只输出 JSON，"
                    "字段包括 goal, slots, required_tools, missing_slots, steps, needs_clarification, rationale。"
                    "required_tools 只能从 recommend_travel_options, search_flights, search_trains, search_hotels, "
                    "check_travel_policy, rag_policy_lookup 中选择。不要编造工具结果。"
                ),
            }
        ]
        if state.get("memory_context"):
            messages.append({"role": "system", "content": state["memory_context"]})
        messages.append({"role": "user", "content": f"用户问题：{text}"})

        plan: dict[str, Any]
        usage: dict[str, int] | None = state.get("usage")
        try:
            resp = await self._llm.chat_completion(messages, temperature=0.0)
            usage = self._usage_dict(resp)
            plan = self._coerce_execution_plan(self._parse_json_object(resp.choices[0].message.content or ""), text)
        except Exception as exc:  # noqa: BLE001
            plan = self._fallback_execution_plan(text, error=str(exc))

        return Command(
            update={
                "execution_plan": plan,
                "usage": usage,
                "trace": self._append_trace(
                    state,
                    "planner_agent",
                    "planned",
                    {
                        "required_tools": plan.get("required_tools", []),
                        "needs_clarification": plan.get("needs_clarification", False),
                    },
                ),
            },
            goto="intent_agent",
        )

    async def _intent_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["policy_reasoner_agent", "rag_agent", "general_agent"]]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        result = await self._intent.recognize(text)
        intent = result.intent.value
        return Command(
            update={
                "intent": intent,
                "trace": self._append_trace(
                    state, "intent_agent", "classified", {"intent": intent}
                ),
            },
            goto=self._route_after_intent(intent, text),
        )

    def _route_after_intent(
        self, intent: str | None, text: str
    ) -> Literal["policy_reasoner_agent", "rag_agent", "general_agent"]:
        travel_intents = {
            TravelIntent.SEARCH_FLIGHT.value,
            TravelIntent.SEARCH_HOTEL.value,
            TravelIntent.SEARCH_TRAIN.value,
            TravelIntent.TRIP_PLANNING.value,
            TravelIntent.APPLICATION.value,
            TravelIntent.POLICY.value,
            TravelIntent.BOOKING.value,
        }
        if self._is_inventory_or_planning_request(text):
            return "policy_reasoner_agent"
        if self._should_use_rag(intent, text):
            return "rag_agent"
        return "policy_reasoner_agent" if intent in travel_intents else "general_agent"

    def _route_after_answer(
        self, state: TravelGraphState
    ) -> Literal["reflection_agent", "verification_agent"]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        if any(k in text for k in ("反思", "检查一遍", "复核", "挑错")):
            return "reflection_agent"
        return "verification_agent"

    async def _policy_reasoner_agent(
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
        store = self._document_store
        if store is not None and getattr(store, "connected", False):
            try:
                query = f"{text}\n差旅制度 酒店标准 舱位 提前预订 审批 金额"
                vector = await EmbeddingService().embed_text(query)
                hits = store.search(vector, top_k=5)
                citations = [
                    {
                        "title": h.get("title"),
                        "doc_type": h.get("doc_type"),
                        "content": str(h.get("content") or "")[:800],
                        "score": h.get("score"),
                    }
                    for h in hits
                ]
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
                    "policy_reasoner_agent",
                    "constraints_extracted",
                    {
                        "source": constraints.get("source"),
                        "confidence": constraints.get("confidence"),
                        "citation_count": len(citations),
                    },
                ),
            },
            goto="travel_react_agent",
        )

    async def _rag_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["reflection_agent", "finalizer_agent"]]:
        text = self._last_user_text(state.get("effective_messages") or state["messages"])
        store = self._document_store
        if store is None or not getattr(store, "connected", False):
            answer, usage = await self._llm_fallback_answer(
                text,
                reason="知识库当前不可用，未能检索公司制度依据。",
                memory_context=state.get("memory_context", ""),
            )
            update = {
                "answer": answer,
                "citations": [],
                "usage": usage,
                "risk_level": "medium",
                "answer_mode": "llm_fallback",
                "trace": self._append_trace(state, "rag_agent", "knowledge_base_unavailable"),
            }
            return Command(update=update, goto=self._route_after_answer({**state, **update}))

        try:
            vector = await EmbeddingService().embed_text(text)
            hits = store.search(vector, top_k=5)
        except Exception as exc:
            answer, usage = await self._llm_fallback_answer(
                text,
                reason=f"知识库检索失败：{exc!s}",
                memory_context=state.get("memory_context", ""),
            )
            update = {
                "answer": answer,
                "citations": [],
                "usage": usage,
                "risk_level": "medium",
                "answer_mode": "llm_fallback",
                "trace": self._append_trace(
                    state, "rag_agent", "retrieval_failed", {"error": str(exc)}
                ),
            }
            return Command(update=update, goto=self._route_after_answer({**state, **update}))

        citations = [
            {
                "title": h.get("title"),
                "doc_type": h.get("doc_type"),
                "content": str(h.get("content") or "")[:800],
                "score": h.get("score"),
            }
            for h in hits
        ]
        if not self._has_reliable_citations(text, citations):
            answer, usage = await self._llm_fallback_answer(
                text,
                reason="当前知识库未检索到足够相关的公司制度依据。",
                memory_context=state.get("memory_context", ""),
            )
            update = {
                "answer": answer,
                "usage": usage,
                "citations": [],
                "risk_level": "medium",
                "answer_mode": "llm_fallback",
                "trace": self._append_trace(
                    state, "rag_agent", "fallback_no_reliable_citation", {"hit_count": len(citations)}
                ),
            }
            return Command(update=update, goto=self._route_after_answer({**state, **update}))

        context = "\n\n".join(
            f"[{i}] 标题：{c.get('title') or '未命名'}\n内容：{c.get('content')}"
            for i, c in enumerate(citations, start=1)
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是企业差旅制度问答 Agent。只能基于给定参考资料回答；"
                    "资料不足时明确说明，不要编造制度。回答末尾列出引用编号。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{state.get('memory_context', '')}\n\n"
                    f"参考资料：\n{context or '无'}\n\n用户问题：{text}"
                ),
            },
        ]
        resp = await self._llm.chat_completion(prompt, temperature=0.1)
        update = {
            "answer": resp.choices[0].message.content or "",
            "usage": self._usage_dict(resp),
            "citations": citations,
            "risk_level": "low" if citations else "medium",
            "answer_mode": "rag_grounded",
            "trace": self._append_trace(
                state, "rag_agent", "retrieved", {"hit_count": len(citations)}
            ),
        }
        return Command(update=update, goto=self._route_after_answer({**state, **update}))

    async def _travel_react_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["policy_validator_agent"]]:
        messages = list(state["openai_messages"])
        if state.get("execution_plan"):
            messages.append(
                {
                    "role": "system",
                    "content": "执行计划（由 Planner Agent 生成）：\n"
                    + json.dumps(state["execution_plan"], ensure_ascii=False),
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
        tools = _travel_tools()
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
                    output = await self._execute_tool(tc.function.name, tc.function.arguments, user_text)
                    tool_trace.append(
                        {
                            "tool": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                            "output": output,
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
                "trace": self._append_trace(
                    state,
                    "travel_react_agent",
                    "answered",
                    {"tool_calls": [t.get("tool") for t in tool_trace]},
                ),
            }
            return Command(update=update, goto="policy_validator_agent")

        update = {
            "answer": "已达到最大推理轮次，请简化问题后重试。",
            "usage": usage,
            "tool_trace": tool_trace,
            "risk_level": "medium",
            "trace": self._append_trace(state, "travel_react_agent", "max_iterations"),
        }
        return Command(update=update, goto="policy_validator_agent")

    async def _policy_validator_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["approval_agent"]]:
        draft = self._build_booking_draft(state)
        validation = self._build_policy_validation(state, draft)
        risk_level = state.get("risk_level") or "low"
        if validation.get("status") in {"failed", "needs_review"}:
            risk_level = "high" if validation.get("violations") else "medium"
        answer = state.get("answer", "")
        if validation.get("summary") and validation.get("status") != "passed":
            answer = f"{answer}\n\n合规校验：{validation['summary']}"
        return Command(
            update={
                "answer": answer,
                "booking_draft": draft,
                "policy_validation": validation,
                "risk_level": risk_level,
                "trace": self._append_trace(
                    state,
                    "policy_validator_agent",
                    validation.get("status", "checked"),
                    {
                        "violations": len(validation.get("violations") or []),
                        "warnings": len(validation.get("warnings") or []),
                    },
                ),
            },
            goto="approval_agent",
        )

    async def _approval_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["reflection_agent", "finalizer_agent"]]:
        draft = state.get("booking_draft") or self._build_booking_draft(state)
        form = self._build_approval_form(state, draft)
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
                "approval_agent",
                "approval_required" if form and form.get("required") else "not_required",
            ),
        }
        return Command(update=update, goto=self._route_after_answer({**state, **update}))

    async def _verification_agent(self, state: TravelGraphState) -> dict[str, Any]:
        answer = state.get("answer", "")
        citations = state.get("citations") or []
        answer_mode = state.get("answer_mode")
        verification = self._verify_answer_grounding(answer, citations, answer_mode)
        updated_answer = answer
        risk_level = state.get("risk_level") or "low"
        mode = answer_mode
        if not verification["passed"]:
            risk_level = "high"
            mode = "llm_fallback"
            unsupported = "；".join(verification["unsupported_terms"])
            updated_answer = (
                f"{answer}\n\n核验提示：以下关键信息未能从当前引用资料中确认：{unsupported}。"
                "请以公司制度或人工审批为准。"
            )
        return {
            "answer": updated_answer,
            "risk_level": risk_level,
            "answer_mode": mode,
            "verification": verification,
            "trace": self._append_trace(
                state,
                "verification_agent",
                "passed" if verification["passed"] else "flagged",
                {"unsupported_count": len(verification["unsupported_terms"])},
            ),
        }

    async def _general_agent(
        self, state: TravelGraphState
    ) -> Command[Literal["reflection_agent", "finalizer_agent"]]:
        resp = await self._llm.chat_completion(state["openai_messages"], temperature=0.2)
        update = {
            "answer": resp.choices[0].message.content or "",
            "usage": self._usage_dict(resp),
            "trace": self._append_trace(state, "general_agent", "answered"),
        }
        return Command(update=update, goto=self._route_after_answer({**state, **update}))

    async def _reflection_agent(self, state: TravelGraphState) -> dict[str, Any]:
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
        return {
            "answer": revised,
            "reflection_notes": "reflection_agent_applied",
            "usage": self._usage_dict(resp),
            "trace": self._append_trace(state, "reflection_agent", "revised"),
        }

    async def _finalizer_agent(self, state: TravelGraphState) -> dict[str, Any]:
        answer = self._enforce_enterprise_answer_contract(
            state.get("answer", ""),
            state.get("tool_trace") or [],
            state.get("effective_messages", state["messages"]),
        )
        session_id = state.get("session_id")
        booking_draft = state.get("booking_draft")
        if session_id:
            await self._save_session_messages(
                session_id,
                state.get("effective_messages", state["messages"])
                + [ChatMessage(role=MessageRole.ASSISTANT, content=answer)],
            )
        if booking_draft:
            await self._save_booking_draft(session_id, booking_draft)
        return {
            "response": {
                "id": str(uuid.uuid4()),
                "created": int(time.time()),
                "model": self._llm.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}}],
                "usage": state.get("usage"),
                "metadata": {
                    "orchestrator": "langgraph",
                    "intent": state.get("intent"),
                    "tool_trace": state.get("tool_trace") or [],
                    "execution_plan": state.get("execution_plan"),
                    "policy_constraints": state.get("policy_constraints"),
                    "policy_validation": state.get("policy_validation"),
                    "reflection": state.get("reflection_notes"),
                    "trace": self._append_trace(state, "finalizer_agent", "completed"),
                    "citations": state.get("citations") or [],
                    "approval_form": state.get("approval_form"),
                    "booking_draft": booking_draft,
                    "risk_level": state.get("risk_level"),
                    "answer_mode": state.get("answer_mode"),
                    "verification": state.get("verification"),
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
        result = await self._graph.ainvoke(
            {"messages": messages, "session_id": session_id, "user_id": user_id}
        )
        return result["response"]

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

    @staticmethod
    def _coerce_execution_plan(raw: dict[str, Any], user_text: str) -> dict[str, Any]:
        fallback = LangGraphTravelOrchestrator._fallback_execution_plan(user_text)
        if not raw:
            return fallback
        allowed_tools = {
            "recommend_travel_options",
            "search_flights",
            "search_trains",
            "search_hotels",
            "check_travel_policy",
            "rag_policy_lookup",
        }
        tools = [
            str(item)
            for item in raw.get("required_tools", [])
            if str(item) in allowed_tools
        ]
        if not tools:
            tools = fallback["required_tools"]
        steps = raw.get("steps")
        if not isinstance(steps, list) or not steps:
            steps = fallback["steps"]
        normalized_steps: list[dict[str, Any]] = []
        for index, item in enumerate(steps, start=1):
            if isinstance(item, dict):
                tool = str(item.get("tool") or "")
                normalized_steps.append(
                    {
                        "order": int(item.get("order") or index),
                        "tool": tool if tool in allowed_tools else "",
                        "reason": str(item.get("reason") or item.get("task") or ""),
                    }
                )
            else:
                normalized_steps.append({"order": index, "tool": "", "reason": str(item)})
        slots = raw.get("slots") if isinstance(raw.get("slots"), dict) else fallback["slots"]
        missing = raw.get("missing_slots") if isinstance(raw.get("missing_slots"), list) else []
        return {
            "goal": str(raw.get("goal") or fallback["goal"]),
            "slots": slots,
            "required_tools": tools,
            "missing_slots": [str(item) for item in missing],
            "steps": normalized_steps,
            "needs_clarification": bool(raw.get("needs_clarification", False)),
            "rationale": str(raw.get("rationale") or fallback["rationale"]),
            "planner": "llm",
        }

    @staticmethod
    def _fallback_execution_plan(user_text: str, error: str | None = None) -> dict[str, Any]:
        tools: list[str] = []
        steps: list[dict[str, Any]] = []
        lower = user_text.lower()
        if any(word in user_text for word in ("政策", "制度", "差标", "报销", "审批", "标准")):
            tools.append("rag_policy_lookup")
            steps.append({"order": len(steps) + 1, "tool": "rag_policy_lookup", "reason": "检索企业差旅制度约束"})
        if any(word in user_text for word in ("航班", "机票", "飞机")):
            tools.append("search_flights")
            steps.append({"order": len(steps) + 1, "tool": "search_flights", "reason": "查询航班候选"})
        if any(word in user_text for word in ("高铁", "火车", "动车", "车次")):
            tools.append("search_trains")
            steps.append({"order": len(steps) + 1, "tool": "search_trains", "reason": "查询高铁/火车候选"})
        if any(word in user_text for word in ("酒店", "住宿")):
            tools.append("search_hotels")
            steps.append({"order": len(steps) + 1, "tool": "search_hotels", "reason": "查询酒店候选"})
        if any(word in user_text for word in ("规划", "推荐", "安排", "出差", "差旅", "booking", "预订")) or "trip" in lower:
            tools = ["recommend_travel_options"]
            steps = [{"order": 1, "tool": "recommend_travel_options", "reason": "综合查询交通和酒店并形成推荐组合"}]
        if not tools:
            tools = []
            steps = [{"order": 1, "tool": "", "reason": "直接回答通用问题"}]
        rationale = "基于关键词兜底生成执行计划"
        if error:
            rationale = f"{rationale}；LLM planner 失败：{error}"
        return {
            "goal": user_text[:120],
            "slots": {},
            "required_tools": tools,
            "missing_slots": [],
            "steps": steps,
            "needs_clarification": False,
            "rationale": rationale,
            "planner": "fallback",
        }

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
                    "你是企业差旅制度约束抽取 Agent。只能根据参考资料抽取 JSON，不要编造。"
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

    @staticmethod
    def _is_inventory_or_planning_request(text: str) -> bool:
        action_words = (
            "查询",
            "查一下",
            "看看",
            "推荐",
            "比较",
            "规划",
            "安排",
            "生成行程",
            "制定行程",
            "商旅规划",
            "出差方案",
        )
        inventory_words = (
            "航班",
            "机票",
            "飞机",
            "高铁",
            "火车",
            "动车",
            "车次",
            "酒店",
            "住宿",
            "北京",
            "上海",
            "广州",
            "深圳",
            "杭州",
        )
        return any(w in text for w in action_words) and any(w in text for w in inventory_words)

    @staticmethod
    def _should_use_rag(intent: str | None, text: str) -> bool:
        if LangGraphTravelOrchestrator._is_inventory_or_planning_request(text):
            return False
        if intent in {TravelIntent.RAG.value, TravelIntent.INFO_QUERY.value}:
            return True
        policy_words = (
            "制度",
            "政策",
            "报销",
            "标准",
            "差标",
            "发票",
            "审批",
            "补贴",
            "舱位",
            "酒店",
            "提前",
            "金额",
            "预订",
        )
        question_indicators = ("几天", "多少", "要求", "标准", "可以", "吗", "是什么", "怎么安排")
        if any(w in text for w in policy_words) and any(q in text for q in question_indicators):
            return True
        action_words = ("规划", "安排", "生成行程", "检查差标", "预订", "下单")
        return any(w in text for w in policy_words) and not any(w in text for w in action_words)

    async def _llm_fallback_answer(
        self,
        text: str,
        *,
        reason: str,
        memory_context: str = "",
    ) -> tuple[str, dict[str, int]]:
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是企业差旅助手。当前没有可靠的公司制度依据时，可以给出通用商旅建议，"
                    "但必须明确标注“未检索到公司制度依据”，不得把建议表述为公司规定。"
                    "涉及金额、报销、审批、舱位、酒店标准时，提醒用户以公司制度或人工审批为准。"
                ),
            },
            {
                "role": "user",
                "content": f"{memory_context}\n\n原因：{reason}\n\n用户问题：{text}",
            },
        ]
        resp = await self._llm.chat_completion(prompt, temperature=0.2)
        content = resp.choices[0].message.content or ""
        if "未检索到公司制度依据" not in content:
            content = f"未检索到公司制度依据。以下为模型基于通用商旅实践的建议：\n\n{content}"
        return content, self._usage_dict(resp)

    @staticmethod
    def _has_reliable_citations(text: str, citations: list[dict[str, Any]]) -> bool:
        if not citations:
            return False
        query_terms = [
            term
            for term in (
                "staff",
                "高级员工",
                "高级",
                "经理",
                "上海",
                "济南",
                "酒店",
                "舱位",
                "高铁",
                "预订",
                "审批",
                "报销",
                "金额",
                "5000",
                "7",
            )
            if term in text
        ]
        if not query_terms:
            return True
        combined = "\n".join(
            f"{item.get('title') or ''}\n{item.get('content') or ''}" for item in citations[:3]
        )
        specificity_terms = [
            term for term in query_terms if term not in {"酒店", "舱位", "预订", "审批", "报销", "金额"}
        ]
        terms_to_check = specificity_terms or query_terms
        return any(term in combined for term in terms_to_check)

    @staticmethod
    def _verify_answer_grounding(
        answer: str,
        citations: list[dict[str, Any]],
        answer_mode: str | None,
    ) -> dict[str, Any]:
        if answer_mode != "rag_grounded":
            return {
                "passed": True,
                "reason": "non_rag_answer",
                "checked_terms": [],
                "unsupported_terms": [],
            }
        citation_text = "\n".join(
            f"{item.get('title') or ''}\n{item.get('content') or ''}" for item in citations
        )
        checked_terms = LangGraphTravelOrchestrator._extract_verifiable_terms(answer)
        unsupported = [term for term in checked_terms if term not in citation_text]
        return {
            "passed": not unsupported,
            "reason": "all_terms_supported" if not unsupported else "unsupported_terms_found",
            "checked_terms": checked_terms,
            "unsupported_terms": unsupported,
        }

    @staticmethod
    def _extract_verifiable_terms(text: str) -> list[str]:
        terms: list[str] = []
        patterns = [
            r"\b\d+(?:\.\d+)?\s*CNY\b",
            r"\b\d+(?:\.\d+)?\s*元\b",
            r"\b\d+\s*天\b",
            r"\b\d+\s*晚\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = re.sub(r"\s+", " ", match.group(0)).strip()
                if value not in terms:
                    terms.append(value)
        keywords = (
            "staff",
            "manager",
            "director",
            "executive",
            "高级员工",
            "经济舱",
            "商务舱",
            "一等座",
            "二等座",
            "审批",
            "特批",
            "上海",
            "济南",
            "北京",
            "酒店",
            "高铁",
        )
        for keyword in keywords:
            if keyword in text and keyword not in terms:
                terms.append(keyword)
        return terms

    @staticmethod
    def _build_approval_form(
        state: TravelGraphState,
        booking_draft: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        tool_trace = state.get("tool_trace") or []
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
        tool_trace = state.get("tool_trace") or []
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

        for item in state.get("tool_trace") or []:
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
    def _usage_dict(resp: Any) -> dict[str, int]:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
