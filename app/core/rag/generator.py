"""LLM answer generation conditioned on retrieved RAG context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from app.core.rag.retriever import RetrievedChunk
from app.infrastructure.llm.client import ChatMessage, LLMClient


class EmbeddingFn(Protocol):
    def __call__(self, text: str) -> Sequence[float]: ...


@dataclass(slots=True)
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    raw: dict[str, Any]


class RAGAnswerGenerator:
    """Builds a grounded prompt from retrieved chunks and calls the LLM."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        system_prompt: str | None = None,
        max_context_chars: int = 12000,
    ) -> None:
        self._llm = llm
        self._system = system_prompt or (
            "你是企业商旅助手。仅根据提供的参考资料回答用户问题；"
            "若资料不足请明确说明，不要编造航班号或政策条款。"
        )
        self._max_context_chars = max_context_chars

    def _format_context(self, chunks: Sequence[RetrievedChunk]) -> str:
        parts: list[str] = []
        used = 0
        for i, ch in enumerate(chunks, start=1):
            block = f"[{i}] (来源:{ch.source}, id={ch.chunk_id})\n{ch.text.strip()}\n"
            if used + len(block) > self._max_context_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n".join(parts).strip()

    async def generate(
        self,
        user_query: str,
        chunks: Sequence[RetrievedChunk],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra_system: str | None = None,
    ) -> GenerationResult:
        context = self._format_context(chunks)
        sys_parts = [self._system]
        if extra_system:
            sys_parts.append(extra_system)
        system_content = "\n\n".join(sys_parts)
        user_content = (
            f"参考资料：\n{context}\n\n用户问题：{user_query.strip()}"
            if context
            else f"用户问题：{user_query.strip()}"
        )
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_content),
            ChatMessage(role="user", content=user_content),
        ]
        out = await self._llm.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        return GenerationResult(
            text=out.content,
            model=out.model,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            raw=out.raw,
        )
