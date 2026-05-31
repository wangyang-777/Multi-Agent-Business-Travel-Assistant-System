"""In-process metrics: latency samples, token totals, success counters."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricsCollector:
    """Thread-safe counters and bounded latency samples (approximate p50)."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    counters: dict[str, int] = field(default_factory=dict)
    latencies_ms: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.counters = defaultdict(int, self.counters)
        self.latencies_ms = defaultdict(list, self.latencies_ms)

    def increment_counter(self, name: str, value: int = 1) -> None:
        with self._lock:
            self.counters[name] += value

    def observe_latency_ms(self, name: str, ms: float) -> None:
        with self._lock:
            bucket = self.latencies_ms[name]
            bucket.append(ms)
            if len(bucket) > 10_000:
                del bucket[: len(bucket) - 5_000]

    def record_tokens(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.counters["tokens_prompt"] += prompt
            self.counters["tokens_completion"] += completion

    def record_success(self, operation: str, ok: bool) -> None:
        self.increment_counter(f"{operation}_success" if ok else f"{operation}_failure")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = dict(self.counters)
            for name, samples in self.latencies_ms.items():
                if not samples:
                    out[f"{name}_p50_ms"] = 0.0
                    continue
                sorted_s = sorted(samples)
                out[f"{name}_p50_ms"] = round(sorted_s[len(sorted_s) // 2], 3)
            return out

    def time_block(self, name: str):
        start = time.perf_counter()

        class _Ctx:
            def __enter__(self_inner) -> _Ctx:
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb) -> None:
                self.observe_latency_ms(name, (time.perf_counter() - start) * 1000.0)

        return _Ctx()
