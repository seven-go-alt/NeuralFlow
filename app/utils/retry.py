from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TypeVar


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass(slots=True)
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    _failures: int = 0
    _state: CircuitState = CircuitState.CLOSED
    _last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and time.monotonic() - self._last_failure_time > self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, fn: Callable[[], Any]) -> Any:
        if self.state is CircuitState.OPEN:
            raise CircuitBreakerOpenError("Circuit breaker is open")
        try:
            result = fn()
        except Exception:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
            raise
        self._reset()
        return result

    async def acall(self, fn: Callable[[], Any]) -> Any:
        if self.state is CircuitState.OPEN:
            raise CircuitBreakerOpenError("Circuit breaker is open")
        try:
            result = await fn()
        except Exception:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
            raise
        self._reset()
        return result

    def _reset(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker prevents a call."""


RT = TypeVar("RT")


async def retry(
    fn: Callable[[], RT],
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> RT:
    """Retry an async function with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except retryable_exceptions as e:
            last_exc = e
            if attempt < max_attempts - 1:
                delay = min(base_delay * (backoff_factor**attempt), max_delay)
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]
