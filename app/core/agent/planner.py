"""
规划 Agent：将目标拆解为步骤并按序执行（每步可为子调用或占位）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class PlannerModel(Protocol):
    async def expand_plan(self, goal: str, context: str) -> str:
        """返回纯文本步骤列表。"""


StepExecutor = Callable[[str, str], Awaitable[str]]


@dataclass
class PlanRunResult:
    """规划执行结果。"""

    goal: str
    steps: List[str]
    step_outputs: List[str] = field(default_factory=list)
    summary: str = ""


class PlanningAgent:
    """
    将用户目标分解为有序步骤，依次调用执行器并汇总。

    若未提供 `step_executor`，仅返回解析后的步骤列表与占位摘要。
    """

    _STEP_LINE = re.compile(r"^\s*\d+[\).、]\s*(.+)$")

    def __init__(
        self,
        model: PlannerModel,
        *,
        step_executor: Optional[StepExecutor] = None,
    ) -> None:
        self._model = model
        self._executor = step_executor

    async def run(self, goal: str, context: str = "") -> PlanRunResult:
        raw = await self._model.expand_plan(goal, context)
        steps = self._parse_steps(raw)
        outputs: list[str] = []

        if self._executor is None:
            summary = f"已生成 {len(steps)} 步计划，尚未逐步执行。"
            return PlanRunResult(goal=goal, steps=steps, step_outputs=outputs, summary=summary)

        for step in steps:
            out = await self._executor(step, context)
            outputs.append(out)

        summary = self._summarize(goal, steps, outputs)
        return PlanRunResult(goal=goal, steps=steps, step_outputs=outputs, summary=summary)

    def _parse_steps(self, raw: str) -> list[str]:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        steps: list[str] = []
        for ln in lines:
            m = self._STEP_LINE.match(ln)
            if m:
                steps.append(m.group(1).strip())
            elif ln.startswith(("- ", "• ", "* ")):
                steps.append(ln[2:].strip())
            elif not steps and len(ln) < 200:
                steps.append(ln)
        if not steps:
            return [raw.strip()] if raw.strip() else ["（未能解析出步骤）"]
        return steps

    def _summarize(self, goal: str, steps: list[str], outputs: list[str]) -> str:
        parts = [f"目标：{goal}", "执行摘要："]
        for i, (s, o) in enumerate(zip(steps, outputs), start=1):
            parts.append(f"{i}. {s} -> {o[:500]}")
        return "\n".join(parts)
