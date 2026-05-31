"""
Agent 编排器 - 系统核心

负责：
1. 接收用户请求
2. 调用意图识别引擎
3. 根据意图选择合适的 Agent 模式（ReAct/Planner/Reflection）
4. 路由到对应的子 Agent（行程规划/信息查询/差标管控/预订/RAG）
5. 管理记忆和上下文
6. 返回结果
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from app.core.agent.planner import PlanRunResult, PlanningAgent
from app.core.agent.react_agent import ReActAgent, Tool, ToolResult
from app.core.agent.reflection import ReflectionAgent
from app.core.intent.recognizer import IntentRecognizer, IntentResult, TravelIntent
from app.core.memory.short_term import ChatTurn, ShortTermMemory
from app.core.prompt.state_machine import ConversationState, PromptStateMachine
from app.core.prompt.templates import PromptTemplateLibrary
from app.core.rag.generator import RAGAnswerGenerator
from app.core.rag.retriever import MultiChannelRetriever, RetrievedChunk
from app.core.tools import policy_query as policy_tool
from app.core.tools.registry import ToolRegistry, invoke as registry_invoke
from app.infrastructure.llm.client import ChatMessage, LLMClient
from app.infrastructure.observability.tracer import child_span, get_trace, start_trace


class AgentMode(str, Enum):
    """编排层选用的推理模式。"""

    REACT = "react"
    PLANNER = "planner"
    REFLECTION = "reflection"


@dataclass
class OrchestratorConfig:
    max_react_iterations: int = 8
    reflection_keywords: tuple[str, ...] = ("反思", "检查一遍", "复核", "挑错")


@dataclass
class OrchestratorResponse:
    session_id: str
    message: str
    intent: TravelIntent
    mode: AgentMode
    trace_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


class _ReActLLM:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def propose_step(self, prompt: str) -> str:
        r = await self._llm.chat([ChatMessage(role="user", content=prompt)])
        return r.content


class _PlannerLLM:
    def __init__(self, llm: LLMClient, templates: PromptTemplateLibrary) -> None:
        self._llm = llm
        self._templates = templates

    async def expand_plan(self, goal: str, context: str) -> str:
        bundle = self._templates.planner_decomposition(goal)
        r = await self._llm.chat(
            [
                ChatMessage(role="system", content=bundle.system),
                ChatMessage(role="user", content=f"{bundle.user_prefix}\n\n上下文：{context}"),
            ]
        )
        return r.content


class _ReflectLLM:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def critique_and_revise(self, system: str, user: str) -> str:
        r = await self._llm.chat(
            [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]
        )
        return r.content


class _RegistryTool:
    def __init__(self, registry: ToolRegistry, name: str, description: str) -> None:
        self._registry = registry
        self.name = name
        self.description = description

    async def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            data = await registry_invoke(self._registry, self.name, arguments)
            return ToolResult(self.name, True, data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(self.name, False, None, str(exc))


class _NoopTool:
    name = "noop"
    description = "未注册业务工具时的占位，提示配置 ToolRegistry。"

    async def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        return ToolResult("noop", True, {"hint": "no tools registered", "args": dict(arguments)})


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


@dataclass(slots=True)
class OrchestratorResult:
    answer: str
    intent: TravelIntent
    retrieved: list[RetrievedChunk]
    tool_output: Any | None


class TravelAgentOrchestrator:
    """面向 RAG 管道的编排：意图 → 检索 → 可选差标工具 → 生成。"""

    def __init__(
        self,
        *,
        retriever: MultiChannelRetriever,
        generator: RAGAnswerGenerator,
        intent_recognizer: IntentRecognizer | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._intent = intent_recognizer or IntentRecognizer()
        self._embedder = embedder

    async def _embedding(self, text: str) -> list[float]:
        if self._embedder is None:
            raise RuntimeError("embedder is required for retrieval")
        return await self._embedder.embed(text)

    async def run(self, user_message: str) -> OrchestratorResult:
        ir = await self._intent.recognize(user_message)
        intent_name = ir.intent.value
        vec = await self._embedding(user_message)
        chunks = await self._retriever.retrieve(
            user_message,
            vec,
            intent=intent_name,
            top_k=8,
        )
        tool_out: Any | None = None
        if ir.intent is TravelIntent.POLICY:
            tool_out = await policy_tool.policy_query(employee_level="staff")
        gen = await self._generator.generate(user_message, chunks)
        return OrchestratorResult(
            answer=gen.text,
            intent=ir.intent,
            retrieved=chunks,
            tool_output=tool_out,
        )


class AgentOrchestrator:
    """主对话编排：意图识别、模式选择、子 Agent 执行、会话记忆与 trace 事件。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        intent_recognizer: IntentRecognizer | None = None,
        tool_registry: ToolRegistry | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._llm = llm
        self._intent = intent_recognizer or IntentRecognizer()
        self._registry = tool_registry or ToolRegistry()
        self._config = config or OrchestratorConfig()
        self._sessions: dict[str, ShortTermMemory] = {}
        self._machines: dict[str, PromptStateMachine] = {}
        self._templates = PromptTemplateLibrary()

    def _memory(self, session_id: str) -> ShortTermMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemory()
        return self._sessions[session_id]

    def _machine(self, session_id: str) -> PromptStateMachine:
        if session_id not in self._machines:
            self._machines[session_id] = PromptStateMachine()
        return self._machines[session_id]

    async def process_message(self, session_id: str, user_input: str) -> OrchestratorResponse:
        if get_trace() is None:
            start_trace("agent.orchestrator")
        span = child_span("orchestrator.process_message")
        try:
            span.add_event("user_input", length=len(user_input))
            ir = await self._recognize_intent(user_input)
            span.add_event("intent", value=ir.intent.value, confidence=ir.confidence)
            mode = self._select_mode(ir, user_input)
            sm = self._machine(session_id)
            conv_state = sm.transition(user_input, intent=ir.intent)
            ctx = await self._build_context(session_id, user_input, ir, conv_state)
            text, extra = await self._route_to_agent(mode, ir, ctx, user_input)
            mem = self._memory(session_id)
            mem.append(ChatTurn(role="user", content=user_input))
            mem.append(ChatTurn(role="assistant", content=text))
            trace = get_trace()
            tid = trace.trace_id if trace else str(uuid.uuid4())
            return OrchestratorResponse(
                session_id=session_id,
                message=text,
                intent=ir.intent,
                mode=mode,
                trace_id=tid,
                metadata={"conversation_state": conv_state.value, **extra},
            )
        finally:
            span.finish()

    async def process_message_stream(
        self, session_id: str, user_input: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        resp = await self.process_message(session_id, user_input)
        body = resp.message
        step = max(12, len(body) // 32 or 1)
        for i in range(0, len(body), step):
            await asyncio.sleep(0)
            yield {"event": "delta", "text": body[i : i + step]}
        yield {"event": "done", "response": resp}

    async def _recognize_intent(self, user_input: str) -> IntentResult:
        return await self._intent.recognize(user_input)

    def _select_mode(self, ir: IntentResult, user_input: str) -> AgentMode:
        if any(k in user_input for k in self._config.reflection_keywords):
            return AgentMode.REFLECTION
        if ir.intent in (TravelIntent.TRIP_PLANNING, TravelIntent.APPLICATION):
            return AgentMode.PLANNER
        return AgentMode.REACT

    async def _build_context(
        self,
        session_id: str,
        user_input: str,
        ir: IntentResult,
        state: ConversationState,
    ) -> str:
        mem = self._memory(session_id)
        history = mem.as_messages()
        sys = self._templates.system_for_state(state)
        names = tuple(t.name for t in self._registry.list_tools())
        tool_line = self._templates.react_tools_section(names)
        tail = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-10:])
        return "\n\n".join(
            [
                sys,
                tool_line,
                f"意图：{ir.intent.value}，置信度 {ir.confidence:.2f}",
                "对话节选：",
                tail or "（空）",
                f"用户输入：{user_input}",
            ]
        )

    async def _route_to_agent(
        self,
        mode: AgentMode,
        ir: IntentResult,
        context: str,
        user_input: str,
    ) -> tuple[str, dict[str, Any]]:
        domain = self._intent_to_domain(ir.intent)
        if mode is AgentMode.PLANNER:
            return await self._execute_planner(context, user_input, domain)
        if mode is AgentMode.REFLECTION:
            return await self._execute_reflection(context, user_input, domain)
        return await self._execute_react(context, user_input, domain)

    def _intent_to_domain(self, intent: TravelIntent) -> str:
        table: dict[TravelIntent, str] = {
            TravelIntent.TRIP_PLANNING: "trip_planning",
            TravelIntent.SEARCH_FLIGHT: "trip_planning",
            TravelIntent.SEARCH_HOTEL: "trip_planning",
            TravelIntent.SEARCH_TRAIN: "trip_planning",
            TravelIntent.INFO_QUERY: "info_query",
            TravelIntent.POLICY: "policy",
            TravelIntent.BOOKING: "booking",
            TravelIntent.APPLICATION: "trip_planning",
            TravelIntent.RAG: "rag",
            TravelIntent.GENERAL: "general",
        }
        return table.get(intent, "general")

    def _react_tool_map(self) -> dict[str, Tool]:
        out: dict[str, Tool] = {}
        for rt in self._registry.list_tools():
            out[rt.name] = _RegistryTool(self._registry, rt.name, rt.description)
        if not out:
            out["noop"] = _NoopTool()
        return out

    async def _execute_react(self, context: str, user_input: str, domain: str) -> tuple[str, dict[str, Any]]:
        head = self._templates.subagent_system_prompt(domain)
        task = f"{head}\n\n{context}"
        agent = ReActAgent(
            _ReActLLM(self._llm),
            self._react_tool_map(),
            max_iterations=self._config.max_react_iterations,
        )
        answer, trace = await agent.run(task, extra_context=user_input)
        return answer, {"react_trace": trace, "domain": domain}

    async def _execute_planner(self, context: str, user_input: str, domain: str) -> tuple[str, dict[str, Any]]:
        planner = PlanningAgent(_PlannerLLM(self._llm, self._templates))
        try:
            result = await planner.run(user_input, context)
        except Exception as exc:  # noqa: BLE001
            result = PlanRunResult(
                goal=user_input,
                steps=["（规划异常）"],
                summary=f"规划失败：{exc}",
            )
        return result.summary, {"plan_steps": result.steps, "domain": domain}

    async def _execute_reflection(self, context: str, user_input: str, domain: str) -> tuple[str, dict[str, Any]]:
        draft_r = await self._llm.chat(
            [
                ChatMessage(role="system", content=self._templates.subagent_system_prompt(domain)),
                ChatMessage(role="user", content=f"{context}\n{user_input}"),
            ]
        )
        draft = draft_r.content
        agent = ReflectionAgent(_ReflectLLM(self._llm))
        out = await agent.reflect(draft, criteria={"domain": domain})
        msg = f"问题：{out.critique}\n\n修订：{out.revised}"
        return msg, {"reflection_passed": out.passed, "domain": domain}
