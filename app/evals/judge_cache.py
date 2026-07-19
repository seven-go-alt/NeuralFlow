from __future__ import annotations

import json
from typing import Any

from app.utils.cache import TTLCache


class JudgeCache:
    """Cache Judge evaluation results keyed by (question, answer, model).

    Uses Redis-backed persistence when available, with in-memory fallback.
    The cache key is a hash of (question, answer, judge_model) so repeated
    eval runs don't re-spend LLM tokens on identical inputs.
    """

    def __init__(
        self,
        use_redis: bool = True,
        ttl_seconds: float = 3600.0,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        self._use_redis = use_redis
        self._ttl = ttl_seconds
        self._redis_url = redis_url
        self._local = TTLCache(max_size=5000, default_ttl_seconds=ttl_seconds)

    def _build_key(self, question: str, answer: str, model: str) -> str:
        return TTLCache.build_key("judge", question, answer, model)

    def get(self, question: str, answer: str, model: str) -> dict[str, Any] | None:
        key = self._build_key(question, answer, model)

        # try Redis first
        if self._use_redis:
            try:
                import redis.asyncio as aioredis

                async def _redis_get() -> Any | None:
                    r = aioredis.Redis.from_url(self._redis_url)
                    data = await r.get(key)
                    await r.aclose()
                    return json.loads(data) if data else None

                import asyncio

                result = asyncio.run(_redis_get())
                if result is not None:
                    return result
            except Exception:
                pass

        # fall back to local
        return self._local.get(key)

    def set(
        self,
        question: str,
        answer: str,
        model: str,
        result: dict[str, Any],
    ) -> None:
        key = self._build_key(question, answer, model)
        self._local.set(key, result, ttl_seconds=self._ttl)

        if self._use_redis:
            try:
                import redis.asyncio as aioredis

                async def _redis_set() -> None:
                    r = aioredis.Redis.from_url(self._redis_url)
                    await r.setex(key, int(self._ttl), json.dumps(result, default=str))
                    await r.aclose()

                import asyncio

                asyncio.run(_redis_set())
            except Exception:
                pass
