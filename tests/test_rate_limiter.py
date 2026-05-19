from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from app.middleware.ratelimit import InMemoryRateLimiter, RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware


@pytest.mark.asyncio
async def test_security_headers_middleware() -> None:
    """SecurityHeadersMiddleware adds security headers to all responses."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/test")
    async def test_endpoint() -> JSONResponse:
        return JSONResponse({"ok": True})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test")

    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert (
        response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    )
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


@pytest.mark.asyncio
async def test_rate_limit_allows_normal_requests() -> None:
    """RateLimitMiddleware allows requests under the limit."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

    @app.get("/test")
    async def test_endpoint() -> JSONResponse:
        return JSONResponse({"ok": True})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(5):
            response = await client.get("/test")
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_exceeded() -> None:
    """RateLimitMiddleware returns 429 when limit is exceeded."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, max_requests=3, window_seconds=60)

    @app.get("/test")
    async def test_endpoint() -> JSONResponse:
        return JSONResponse({"ok": True})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            response = await client.get("/test")
            assert response.status_code == 200

        # 4th request should be rate limited
        response = await client.get("/test")
        assert response.status_code == 429
        assert response.json() == {"detail": "Too many requests"}


@pytest.mark.asyncio
async def test_rate_limit_resets_after_window() -> None:
    """RateLimitMiddleware resets the counter after the window expires."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, max_requests=2, window_seconds=1)

    @app.get("/test")
    async def test_endpoint() -> JSONResponse:
        return JSONResponse({"ok": True})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(2):
            response = await client.get("/test")
            assert response.status_code == 200

        response = await client.get("/test")
        assert response.status_code == 429

        # Wait for window to expire and check it resets
        await asyncio.sleep(1.1)

        response = await client.get("/test")
        assert response.status_code == 200


def test_in_memory_rate_limiter_allows_within_limit() -> None:
    """InMemoryRateLimiter allows requests under the max."""
    limiter = InMemoryRateLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert limiter.check("1.2.3.4") is True


def test_in_memory_rate_limiter_blocks_excess() -> None:
    """InMemoryRateLimiter blocks requests over the max."""
    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        assert limiter.check("1.2.3.4") is True
    assert limiter.check("1.2.3.4") is False


def test_in_memory_rate_limiter_per_ip() -> None:
    """InMemoryRateLimiter tracks IPs independently."""
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.check("1.1.1.1") is True
    assert limiter.check("1.1.1.1") is True
    assert limiter.check("1.1.1.1") is False  # blocked

    # Different IP is still allowed
    assert limiter.check("2.2.2.2") is True
    assert limiter.check("2.2.2.2") is True
    assert limiter.check("2.2.2.2") is False


def test_in_memory_rate_limiter_unknown_ip() -> None:
    """InMemoryRateLimiter handles 'unknown' IP."""
    limiter = InMemoryRateLimiter(max_requests=1, window_seconds=60)
    assert limiter.check("unknown") is True
    assert limiter.check("unknown") is False
