from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class RedisRateLimiter:
    """Redis-backed rate limiter. Falls back to in-memory if Redis is unavailable."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def check(self, client_ip: str) -> bool:
        import redis.asyncio as aioredis

        try:
            r = aioredis.Redis(host="localhost", port=6379, db=0)
            key = f"ratelimit:{client_ip}"
            current = await r.get(key)
            if current is None:
                await r.setex(key, self.window_seconds, 1)
                await r.aclose()
                return True
            count = int(current)
            if count >= self.max_requests:
                await r.aclose()
                return False
            await r.incr(key)
            await r.aclose()
            return True
        except Exception:
            return True  # fail open


class InMemoryRateLimiter:
    """Simple in-memory fallback rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: dict[str, list[float]] = defaultdict(list)

    def check(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self._clients[client_ip] = [t for t in self._clients[client_ip] if t > window_start]
        if len(self._clients[client_ip]) >= self.max_requests:
            return False
        self._clients[client_ip].append(now)
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app: ASGIApp, max_requests: int = 100, window_seconds: int = 60, use_redis: bool = False
    ) -> None:
        super().__init__(app)
        self._redis_limiter = RedisRateLimiter(max_requests, window_seconds)
        self._in_memory_limiter = InMemoryRateLimiter(max_requests, window_seconds)
        self._use_redis = use_redis

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        client_ip = request.client.host if request.client else "unknown"

        if self._use_redis:
            allowed = await self._redis_limiter.check(client_ip)
            if not allowed:
                return JSONResponse(status_code=429, content={"detail": "Too many requests"})
        else:
            if not self._in_memory_limiter.check(client_ip):
                return JSONResponse(status_code=429, content={"detail": "Too many requests"})

        return await call_next(request)
