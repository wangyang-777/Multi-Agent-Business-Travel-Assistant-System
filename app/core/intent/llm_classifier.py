"""
LLM 意图分类器（慢车道）- 对复杂表述进行结构化分类。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class StructuredLLMClient(Protocol):
    """可注入的异步 LLM，返回结构化 JSON 字符串或对象。"""

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_content: str,
        response_format: str,
    ) -> str:
        """返回 JSON 字符串，包含 intent_slug 与 confidence。"""


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


@dataclass
class LLMClassification:
    """LLM 分类输出。"""

    intent_slug: str
    confidence: float
    rationale: str = ""


class LLMIntentClassifier:
    """
    使用 LLM 进行意图分类，要求模型输出固定 JSON 结构以便解析与校验。
    """

    def __init__(
        self,
        llm: StructuredLLMClient,
        *,
        allowed_slugs: tuple[str, ...],
    ) -> None:
        self._llm = llm
        self._allowed = frozenset(allowed_slugs)

    async def classify(self, text: str) -> LLMClassification:
        """对文本进行异步分类；解析失败时回退为 general 低置信度。"""
        system = (
            "你是商旅助手意图分类器。根据用户输入，从给定意图中选一个最匹配的。"
            "必须只输出一个 JSON 对象，字段：intent_slug (string), confidence (0-1 小数), "
            "rationale (简短中文理由)。不要输出其它文字。"
        )
        user = (
            f"用户输入：\n{text}\n\n"
            f"可选意图 slug 列表：{sorted(self._allowed)}"
        )
        raw = await self._llm.complete_structured(
            system_prompt=system,
            user_content=user,
            response_format="json",
        )
        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> LLMClassification:
        try:
            payload = self._extract_json(raw)
            slug = str(payload.get("intent_slug", "general")).strip()
            conf = float(payload.get("confidence", 0.5))
            rationale = str(payload.get("rationale", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            return LLMClassification("general", 0.35, "parse_error")

        conf = max(0.0, min(1.0, conf))
        if slug not in self._allowed:
            return LLMClassification("general", conf * 0.6, rationale or "unknown_slug")
        return LLMClassification(slug, conf, rationale)

    def _extract_json(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        match = _JSON_BLOCK.search(raw)
        if not match:
            raise json.JSONDecodeError("no json object", raw, 0)
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise TypeError("expected object")
        return parsed
