"""OpenAI-compatible async chat client for multiple LLM providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from openai import AsyncOpenAI

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None


@dataclass(slots=True)
class ChatCompletionResult:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    raw: dict[str, Any]


class LLMClient:
    """Multi-provider LLM client using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "gpt-4o-mini",
        timeout_s: float = 120.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            default_headers=default_headers,
        )
        self._default_model = default_model

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        mname = model or self._default_model
        payload: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": mname,
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage
        return ChatCompletionResult(
            content=content,
            model=resp.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            raw=resp.model_dump(),
        )
