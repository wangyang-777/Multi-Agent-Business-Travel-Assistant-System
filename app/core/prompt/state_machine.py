"""
动态提示状态机 - 根据对话状态与用户输入建议下一状态与提示策略。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from app.core.intent.recognizer import BusinessIntent  # 与 TravelIntent 同义，供状态迁移使用


class ConversationState(str, Enum):
    """对话生命周期状态。"""

    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    PLANNING = "planning"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    COMPLETED = "completed"


_CONFIRM_RE = re.compile(r"(确认|同意|可以|就这样|没问题|下单|预订)")
_DENY_RE = re.compile(r"(修改|不对|取消|再改|换一个)")
_GREETING_RE = re.compile(r"(你好|您好|在吗|帮忙|谢谢)")


class PromptStateMachine:
    """
    根据当前状态、用户输入与意图，计算下一状态与是否进入执行。
    """

    def __init__(self, initial: ConversationState = ConversationState.GREETING) -> None:
        self._state = initial

    @property
    def state(self) -> ConversationState:
        return self._state

    def reset(self, state: ConversationState = ConversationState.GREETING) -> None:
        self._state = state

    def transition(
        self,
        user_text: str,
        intent: Optional[BusinessIntent] = None,
    ) -> ConversationState:
        """
        根据用户文本与（可选）意图更新内部状态并返回新状态。
        """
        text = user_text.strip()
        next_state = self._infer_next(self._state, text, intent)
        self._state = next_state
        return next_state

    def _infer_next(
        self,
        current: ConversationState,
        text: str,
        intent: Optional[BusinessIntent],
    ) -> ConversationState:
        if not text:
            return current

        if _GREETING_RE.search(text) and current == ConversationState.GREETING:
            return ConversationState.COLLECTING_INFO

        if intent == BusinessIntent.TRIP_PLANNING and current in (
            ConversationState.GREETING,
            ConversationState.COLLECTING_INFO,
        ):
            return ConversationState.PLANNING

        if intent in (BusinessIntent.BOOKING, BusinessIntent.APPLICATION) and current in (
            ConversationState.PLANNING,
            ConversationState.CONFIRMING,
        ):
            if _CONFIRM_RE.search(text):
                return ConversationState.EXECUTING
            if _DENY_RE.search(text):
                return ConversationState.COLLECTING_INFO

        if current == ConversationState.PLANNING and _CONFIRM_RE.search(text):
            return ConversationState.CONFIRMING

        if current == ConversationState.CONFIRMING and _CONFIRM_RE.search(text):
            return ConversationState.EXECUTING

        if current == ConversationState.EXECUTING and (
            _CONFIRM_RE.search(text) or "完成" in text or "好了" in text
        ):
            return ConversationState.COMPLETED

        if current == ConversationState.COMPLETED and len(text) > 6:
            return ConversationState.COLLECTING_INFO

        if intent == BusinessIntent.INFO_QUERY and current == ConversationState.GREETING:
            return ConversationState.COLLECTING_INFO

        return current

    def should_use_structured_planning(self, state: ConversationState) -> bool:
        return state in (ConversationState.PLANNING, ConversationState.CONFIRMING)
