"""
反思 Agent：对草稿输出进行评估并给出修订稿。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class ReflectionModel(Protocol):
    async def critique_and_revise(self, system: str, user: str) -> str:
        ...


@dataclass
class ReflectionOutcome:
    """反思结果。"""

    critique: str
    revised: str
    passed: bool


class ReflectionAgent:
    """
    根据标准对草稿进行审查，生成问题说明与修订文本。

    `strict=True` 时仅在明确无问题时视为通过。
    """

    def __init__(self, model: ReflectionModel) -> None:
        self._model = model

    async def reflect(
        self,
        draft: str,
        *,
        criteria: Optional[Mapping[str, Any]] = None,
        strict: bool = False,
    ) -> ReflectionOutcome:
        criteria = criteria or {"合规": "符合公司差标", "完整性": "覆盖用户问题要点"}
        sys = (
            "你是严格的质量审查员。先列出问题（若有），再给出修订版正文。"
            "若草稿已满足标准，问题写「无」，修订版可与草稿相同。"
        )
        user = (
            f"草稿：\n{draft}\n\n审查标准：{dict(criteria)}\n"
            "输出格式：第一段「问题：...」，第二段「修订：...」。"
        )
        text = await self._model.critique_and_revise(sys, user)
        critique, revised = self._split_sections(text)
        passed = self._is_pass(critique, strict)
        return ReflectionOutcome(critique=critique, revised=revised or draft, passed=passed)

    def _split_sections(self, text: str) -> tuple[str, str]:
        lines = text.strip().splitlines()
        critique_lines: list[str] = []
        revised_lines: list[str] = []
        mode = "critique"
        for ln in lines:
            if ln.startswith("修订：") or ln.startswith("修订:"):
                mode = "revised"
                rest = ln.split("：", 1)[-1].split(":", 1)[-1].strip()
                if rest:
                    revised_lines.append(rest)
                continue
            if mode == "critique":
                if ln.startswith("问题：") or ln.startswith("问题:"):
                    critique_lines.append(ln.split("：", 1)[-1].split(":", 1)[-1].strip())
                else:
                    critique_lines.append(ln)
            else:
                revised_lines.append(ln)
        critique = "\n".join(critique_lines).strip()
        revised = "\n".join(revised_lines).strip()
        return critique, revised

    def _is_pass(self, critique: str, strict: bool) -> bool:
        if "无" in critique[:20] and "问题" in critique[:20]:
            return True
        if not strict and ("无严重" in critique or "基本可用" in critique):
            return True
        return strict and "无" in critique[:10]
