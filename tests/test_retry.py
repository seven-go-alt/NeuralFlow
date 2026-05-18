from __future__ import annotations

import pytest

from app.utils.retry import CircuitBreaker, CircuitBreakerOpenError, retry


class TestRetry:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self) -> None:
        async def fn() -> str:
            return "ok"

        result = await retry(fn)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("not yet")
            return "ok"

        result = await retry(fn, max_attempts=5)
        assert result == "ok"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_exhaust_retries(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            raise ValueError("always fail")

        with pytest.raises(ValueError, match="always fail"):
            await retry(fn, max_attempts=3)
        assert calls == 3

    @pytest.mark.asyncio
    async def test_retryable_exceptions_filter(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await retry(fn, max_attempts=3, retryable_exceptions=(ValueError,))
        assert calls == 1  # no retry on TypeError


class TestCircuitBreaker:
    def test_closed_by_default(self) -> None:
        cb = CircuitBreaker()
        assert cb.state.name == "CLOSED"

    def test_open_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=999)

        def fail() -> None:
            raise ValueError("fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(fail)

        assert cb.state.name == "OPEN"

    def test_blocks_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999)

        def fail() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            cb.call(fail)

        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "should not reach")

    def test_resets_after_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=999)
        calls = 0

        def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("fail")
            return "ok"

        with pytest.raises(ValueError):
            cb.call(flaky)
        assert cb.state.name == "CLOSED"  # not yet at threshold
        with pytest.raises(ValueError):
            cb.call(flaky)
        assert cb.state.name == "OPEN"

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        def fail() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            cb.call(fail)

        # state is now OPEN, but after recovery_timeout it becomes HALF_OPEN
        import time

        time.sleep(0.02)
        assert cb.state.name == "HALF_OPEN"

    @pytest.mark.asyncio
    async def test_acall(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=999)

        async def fail() -> None:
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await cb.acall(fail)

        with pytest.raises(CircuitBreakerOpenError):
            await cb.acall(fail)
