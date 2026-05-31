"""
规则引擎（快车道）- 基于关键词与正则的意图快速匹配。

适用于高频、表述相对固定的商旅场景意图。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RuleSpec:
    """单条规则：意图 slug、权重、匹配模式。"""

    intent_slug: str
    weight: float
    patterns: tuple[re.Pattern[str], ...]


_TIE_BREAK_PRIORITY: dict[str, int] = {
    "policy": 100,
    "trip_planning": 92,
    "application": 90,
    "booking": 88,
    "search_flight": 72,
    "search_hotel": 70,
    "search_train": 70,
    "info_query": 60,
    "rag": 55,
}


class RuleEngine:
    """
    基于正则与关键词的规则分类器。

    匹配策略：遍历规则，对每条规则累计命中次数加权，取得分最高意图；
    分数相同时按业务优先级打破平局（如「差标」优先于单独「酒店」词）。
    """

    def __init__(self) -> None:
        self._rules: list[RuleSpec] = [
            RuleSpec(
                "search_flight",
                1.0,
                (
                    re.compile(r"(机票|航班|飞机|直飞)"),
                    re.compile(r"(飞|乘机).{0,4}(北京|上海|广州|深圳)"),
                ),
            ),
            RuleSpec(
                "search_hotel",
                1.0,
                (
                    re.compile(r"(酒店|住宿|入住)"),
                    re.compile(r"订房"),
                ),
            ),
            RuleSpec(
                "search_train",
                1.0,
                (
                    re.compile(r"(高铁|火车|动车|铁路)"),
                    re.compile(r"车次"),
                ),
            ),
            RuleSpec(
                "trip_planning",
                1.0,
                (
                    re.compile(r"(规划|安排|制定).{0,8}(行程|出差|差旅)"),
                    re.compile(r"(出差|差旅).{0,6}(去|到|飞)"),
                    re.compile(r"行程\s*(规划|安排)"),
                ),
            ),
            RuleSpec(
                "application",
                1.0,
                (
                    re.compile(r"(提|提交|发起).{0,6}(申请|审批|报备)"),
                    re.compile(r"(出差|差旅).{0,6}申请"),
                ),
            ),
            RuleSpec(
                "policy",
                1.0,
                (
                    re.compile(r"(差标|标准|额度|政策|合规)"),
                    re.compile(r"(超标|违规|不允许)"),
                ),
            ),
            RuleSpec(
                "booking",
                1.0,
                (
                    re.compile(r"(订|预订|购买).{0,6}(票|酒店|机|火)"),
                    re.compile(r"(机票|火车票|酒店).{0,4}(订|买)"),
                ),
            ),
            RuleSpec(
                "info_query",
                0.9,
                (
                    re.compile(r"(查询|查一下|看看).{0,8}(航班|天气|政策)"),
                    re.compile(r"(几点|多少钱|多久|在哪)"),
                ),
            ),
            RuleSpec(
                "rag",
                0.85,
                (
                    re.compile(r"(公司|内部).{0,6}(规定|制度|手册)"),
                    re.compile(r"(根据|依据).{0,6}(文档|知识)"),
                ),
            ),
        ]

    def classify(self, text: str) -> Optional[tuple[str, float]]:
        """
        对输入文本进行规则匹配。

        Returns:
            (意图 slug, 规则置信度 0~1) 或无可信匹配时返回 None。
        """
        normalized = text.strip()
        if not normalized:
            return None

        scores: dict[str, float] = {}
        for rule in self._rules:
            hits = 0
            for pat in rule.patterns:
                if pat.search(normalized):
                    hits += 1
            if hits:
                scores[rule.intent_slug] = scores.get(rule.intent_slug, 0.0) + hits * rule.weight

        if not scores:
            return None

        best_slug = max(
            scores,
            key=lambda s: (scores[s], _TIE_BREAK_PRIORITY.get(s, 0)),
        )
        best_rule = next(r for r in self._rules if r.intent_slug == best_slug)
        hits = sum(1 for pat in best_rule.patterns if pat.search(normalized))
        confidence = min(1.0, 0.82 + 0.06 * max(0, hits - 1))
        return best_slug, confidence
