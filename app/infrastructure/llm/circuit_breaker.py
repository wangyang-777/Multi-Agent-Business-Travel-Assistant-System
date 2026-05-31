"""Three-state circuit breaker for LLM and external dependency calls."""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """CLOSED: normal. Failures reach threshold -> OPEN. After cooldown -> HALF_OPEN probe."""

    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    half_open_max_calls: int = 1
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _half_open_inflight: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            await self._transition_on_cooldown()
            if self._state is CircuitState.OPEN:
                raise RuntimeError("circuit breaker is OPEN")
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_inflight >= self.half_open_max_calls:
                    raise RuntimeError("circuit breaker HALF_OPEN probe already running")
                self._half_open_inflight += 1

        try:
            result = await fn()
        except Exception:
            async with self._lock:
                await self._on_failure()
            raise

        async with self._lock:
            await self._on_success()
        return result

    async def _transition_on_cooldown(self) -> None:
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_inflight = 0

    async def _on_failure(self) -> None:
        self._failures += 1
        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._half_open_inflight = 0
            self._failures = self.failure_threshold
            return
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    async def _on_success(self) -> None:
        self._failures = 0
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
        self._state = CircuitState.CLOSED

    async def reset(self) -> None:
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None
            self._half_open_inflight = 0

    @property
    def state(self) -> CircuitState:
        return self._state
