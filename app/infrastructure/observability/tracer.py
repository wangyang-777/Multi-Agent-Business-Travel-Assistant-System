"""Lightweight trace and span model for request-scoped observability."""

from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(slots=True)
class Span:
    span_id: str
    name: str
    trace_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def finish(self) -> None:
        if self.end_ns is None:
            self.end_ns = time.time_ns()

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append({"name": name, "ts_ns": time.time_ns(), **attrs})

    @property
    def duration_ms(self) -> float | None:
        if self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000.0


@dataclass(slots=True)
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)

    def new_span(self, name: str, parent: Span | None = None) -> Span:
        span = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            trace_id=self.trace_id,
            parent_id=parent.span_id if parent else None,
            start_ns=time.time_ns(),
        )
        self.spans.append(span)
        return span


_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "current_trace", default=None
)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None
)


def start_trace(name: str | None = None) -> Trace:
    trace = Trace(trace_id=str(uuid.uuid4()))
    root = trace.new_span(name or "root")
    _current_trace.set(trace)
    _current_span.set(root)
    return trace


def get_trace() -> Trace | None:
    return _current_trace.get()


def get_span() -> Span | None:
    return _current_span.get()


def child_span(name: str) -> Span:
    trace = _current_trace.get() or start_trace("implicit")
    parent = _current_span.get()
    span = trace.new_span(name, parent=parent)
    _current_span.set(span)
    return span


@contextmanager
def use_span(span: Span) -> Iterator[Span]:
    token = _current_span.set(span)
    try:
        yield span
    finally:
        span.finish()
        _current_span.reset(token)


def end_trace() -> None:
    span = _current_span.get()
    if span:
        span.finish()
    _current_trace.set(None)
    _current_span.set(None)
