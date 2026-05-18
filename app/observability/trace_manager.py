from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generator
from uuid import uuid4


@dataclass(slots=True)
class TraceSpan:
    name: str
    span_id: str = ""
    parent_id: str | None = None
    trace_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[TraceSpan] = field(default_factory=list)

    def close(self) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


_current_span: ContextVar[TraceSpan | None] = ContextVar("current_span", default=None)
_current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="")


class TraceManager:
    """Manages nested trace spans with parent-child relationships."""

    def __init__(self, name: str = "root") -> None:
        trace_id = _current_trace_id.get() or str(uuid4())
        self._root = TraceSpan(
            name=name,
            span_id=str(uuid4()),
            trace_id=trace_id,
            start_time=time.perf_counter(),
        )
        self._stack: list[TraceSpan] = [self._root]

    @property
    def root(self) -> TraceSpan:
        return self._root

    @property
    def current(self) -> TraceSpan:
        return self._stack[-1]

    def start_span(self, name: str, **metadata: Any) -> TraceSpan:
        parent = self.current
        span = TraceSpan(
            name=name,
            span_id=str(uuid4()),
            parent_id=parent.span_id,
            trace_id=self._root.trace_id,
            start_time=time.perf_counter(),
            metadata=metadata,
        )
        parent.children.append(span)
        self._stack.append(span)
        return span

    def end_span(self) -> TraceSpan:
        span = self._stack.pop()
        span.close()
        if self._stack:
            _current_span.set(self._stack[-1])
        return span

    def close(self) -> TraceSpan:
        while len(self._stack) > 1:
            self.end_span()
        self._root.close()
        return self._root

    def to_dict(self) -> dict[str, Any]:
        return self._root.to_dict()

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Generator[TraceSpan, None, None]:
        span = self.start_span(name, **metadata)
        _current_span.set(span)
        try:
            yield span
        finally:
            self.end_span()
            _current_span.set(self._stack[-1] if self._stack else None)
