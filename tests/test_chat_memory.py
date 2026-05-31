from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent.orchestrator import SYSTEM_PROMPT, TravelOrchestrator
from app.domain.schemas import ChatMessage, MessageRole


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.store[key] = value


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self._reply_no = 0
        self.model = "fake-model"

    async def chat_completion(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        self._reply_no += 1
        return SimpleNamespace(
            id=f"resp-{self._reply_no}",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"assistant reply {self._reply_no}",
                        tool_calls=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


@pytest.mark.asyncio
async def test_session_messages_are_loaded_from_redis() -> None:
    redis = _FakeRedis()
    llm = _FakeLLM()
    orch = TravelOrchestrator(llm=llm, redis_client=redis)

    first_turn = [ChatMessage(role=MessageRole.USER, content="我叫小王")]
    await orch.run_completion(first_turn, session_id="s1")

    second_turn = [ChatMessage(role=MessageRole.USER, content="你还记得我叫什么吗")]
    await orch.run_completion(second_turn, session_id="s1")

    second_call = llm.calls[-1]
    contents = [msg["content"] for msg in second_call]
    assert contents[0].startswith(SYSTEM_PROMPT)
    assert "当前日期：" in contents[0]
    assert contents[1:] == ["我叫小王", "assistant reply 1", "你还记得我叫什么吗"]

    saved = json.loads(redis.store["chat:session:s1"])
    assert [item["content"] for item in saved] == [
        "我叫小王",
        "assistant reply 1",
        "你还记得我叫什么吗",
        "assistant reply 2",
    ]


@pytest.mark.asyncio
async def test_full_history_resubmission_does_not_duplicate_messages() -> None:
    redis = _FakeRedis()
    llm = _FakeLLM()
    orch = TravelOrchestrator(llm=llm, redis_client=redis)

    await orch.run_completion(
        [ChatMessage(role=MessageRole.USER, content="第一句")],
        session_id="s2",
    )

    resent_history = [
        ChatMessage(role=MessageRole.USER, content="第一句"),
        ChatMessage(role=MessageRole.ASSISTANT, content="assistant reply 1"),
        ChatMessage(role=MessageRole.USER, content="第二句"),
    ]
    await orch.run_completion(resent_history, session_id="s2")

    saved = json.loads(redis.store["chat:session:s2"])
    assert [item["content"] for item in saved] == [
        "第一句",
        "assistant reply 1",
        "第二句",
        "assistant reply 2",
    ]
