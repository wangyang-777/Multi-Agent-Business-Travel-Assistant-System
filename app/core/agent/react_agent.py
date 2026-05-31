"""
ReAct Agent：推理-行动-观察循环，支持异步工具调用与迭代上限。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


@dataclass
class ToolResult:
    """工具执行结果。"""

    name: str
    ok: bool
    data: Any
    error: Optional[str] = None


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    async def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        ...


@dataclass
class ReActStep:
    """单步模型输出解析结果。"""

    thought: str
    action: str
    action_input: dict[str, Any]
    final_answer: Optional[str] = None


@runtime_checkable
class ReActModel(Protocol):
    async def propose_step(self, prompt: str) -> str:
        """返回包含 JSON 的模型原文，供解析为 ReActStep。"""


class ReActAgent:
    """
    Reason-Act-Observe：模型每步输出 thought / action / action_input 或 final_answer，
    调用工具并追加观察，达到上限或收到 final_answer 时结束。
    """

    def __init__(
        self,
        model: ReActModel,
        tools: dict[str, Tool],
        *,
        max_iterations: int = 8,
    ) -> None:
        self._model = model
        self._tools = tools
        self._max_iterations = max(1, max_iterations)

    async def run(
        self,
        task_prompt: str,
        *,
        extra_context: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        history: list[dict[str, Any]] = []
        context = task_prompt if not extra_context else f"{task_prompt}\n\n{extra_context}"

        for i in range(self._max_iterations):
            raw = await self._model.propose_step(self._build_prompt(context, history))
            step = self._parse_step(raw)

            if step.final_answer is not None:
                history.append(
                    {
                        "iteration": i + 1,
                        "thought": step.thought,
                        "final_answer": step.final_answer,
                    }
                )
                return step.final_answer, history

            tool = self._tools.get(step.action)
            if tool is None:
                obs = f"错误：未知工具 {step.action}"
                history.append(
                    {
                        "iteration": i + 1,
                        "thought": step.thought,
                        "action": step.action,
                        "action_input": step.action_input,
                        "observation": obs,
                        "error": "unknown_tool",
                    }
                )
                context = f"{context}\n观察：{obs}"
                continue

            try:
                result = await tool.invoke(step.action_input)
                obs = self._format_observation(result)
            except Exception as exc:  # noqa: BLE001
                obs = f"工具执行异常：{exc}"
                history.append(
                    {
                        "iteration": i + 1,
                        "thought": step.thought,
                        "action": step.action,
                        "action_input": step.action_input,
                        "observation": obs,
                        "error": "tool_exception",
                    }
                )
                context = f"{context}\n观察：{obs}"
                continue

            history.append(
                {
                    "iteration": i + 1,
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": obs,
                }
            )
            context = f"{context}\n观察：{obs}"

        return "达到最大迭代次数仍未得到最终答案，请缩小任务或检查工具。", history

    def _build_prompt(self, base: str, history: list[dict[str, Any]]) -> str:
        lines = [base, "\n已执行步骤："]
        if not history:
            lines.append("（无）")
        for h in history:
            if "final_answer" in h:
                continue
            lines.append(json.dumps(h, ensure_ascii=False))
        lines.append(
            "\n请输出 JSON："
            '{"thought":"...","action":"工具名或 answer","action_input":{},"final_answer":null}'
            "若已可回复用户，将 action 设为 answer，action_input 为空对象，并填写 final_answer。"
        )
        return "\n".join(lines)

    def _parse_step(self, raw: str) -> ReActStep:
        text = raw.strip()
        try:
            payload = self._extract_json(text)
        except json.JSONDecodeError:
            return ReActStep(
                thought="parse_error",
                action="answer",
                action_input={},
                final_answer=text[:4000],
            )

        thought = str(payload.get("thought", ""))
        action = str(payload.get("action", "answer"))
        action_input = payload.get("action_input") if isinstance(payload.get("action_input"), dict) else {}
        final = payload.get("final_answer")
        if action == "answer" and final is not None:
            return ReActStep(thought, action, action_input, final_answer=str(final))
        return ReActStep(thought, action, dict(action_input), None)

    def _extract_json(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        m = _JSON_FENCE.search(raw)
        if m:
            return json.loads(m.group(1).strip())
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise json.JSONDecodeError("no json", raw, 0)

    def _format_observation(self, result: ToolResult) -> str:
        if result.ok:
            return json.dumps({"ok": True, "data": result.data}, ensure_ascii=False)
        return json.dumps(
            {"ok": False, "error": result.error or "failed", "data": result.data},
            ensure_ascii=False,
        )
