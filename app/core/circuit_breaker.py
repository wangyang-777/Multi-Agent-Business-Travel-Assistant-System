from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    异步熔断器：连续失败后打开，经 recovery_timeout 后半开试恢复。
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    _failures: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            await self._transition_on_call()
            if self._state == CircuitState.OPEN:
                raise RuntimeError("circuit breaker is OPEN")

        try:
            result = await fn()
        except Exception:
            async with self._lock:
                await self._on_failure()
            raise

        async with self._lock:
            await self._on_success()
        return result

    async def _transition_on_call(self) -> None:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise RuntimeError("circuit breaker half-open call limit exceeded")
            self._half_open_calls += 1

    async def _on_success(self) -> None:
        self._failures = 0
        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            self._state = CircuitState.CLOSED

    async def _on_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.monotonic()
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN

    @property
    def state(self) -> CircuitState:
        return self._state
