"""Priority-based LLM routing with circuit breakers and automatic failover."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from app.infrastructure.llm.circuit_breaker import CircuitBreaker
from app.infrastructure.llm.client import ChatCompletionResult, ChatMessage, LLMClient

logger = logging.getLogger(__name__)


@dataclass(order=True, slots=True)
class ModelRoute:
    priority: int
    name: str
    model: str
    client: LLMClient
    breaker: CircuitBreaker


class LLMRouter:
    """Selects the highest-priority healthy route and fails over on errors."""

    def __init__(self, routes: Sequence[ModelRoute]) -> None:
        if not routes:
            raise ValueError("at least one ModelRoute is required")
        self._routes = sorted(routes, key=lambda r: r.priority)

    @property
    def routes(self) -> tuple[ModelRoute, ...]:
        return tuple(self._routes)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        last_error: Exception | None = None
        for route in self._routes:

            async def _call(r: ModelRoute = route) -> ChatCompletionResult:
                return await r.client.chat(
                    messages,
                    model=r.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )

            try:
                return await route.breaker.call(_call)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "llm.route_failed",
                    extra={"route": route.name, "model": route.model, "error": str(exc)},
                )
                continue
        if last_error:
            raise last_error
        raise RuntimeError("no LLM routes configured")
