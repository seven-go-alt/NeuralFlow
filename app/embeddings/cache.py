from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_redis_pool: Any = None  # redis.ConnectionPool singleton


def _get_redis_client() -> Any:
    """Return a Redis client from the shared connection pool (lazily created)."""
    global _redis_pool
    try:
        import redis as redis_module

        from app.config import get_settings

        if _redis_pool is None:
            s = get_settings()
            _redis_pool = redis_module.ConnectionPool(
                host=s.redis_host,
                port=s.redis_port,
                db=s.redis_db,
                decode_responses=True,
            )
        return redis_module.Redis(connection_pool=_redis_pool)
    except Exception as exc:
        logger.warning("Failed to obtain Redis connection: %s", exc)
        return None


class EmbeddingCache:
    """Two-tier embedding cache: in-memory (L1) + Redis (L2) with connection pooling."""

    _REDIS_TTL = 3600

    def __init__(self, use_redis: bool = True) -> None:
        self._store: dict[str, list[float]] = {}
        self._use_redis = use_redis

    def build_key(self, model: str, text: str) -> str:
        return hashlib.sha256(
            json.dumps({"model": model, "text": text}, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        key = self.build_key(model, text)
        cached = self._store.get(key)
        if cached is not None:
            return cached
        if self._use_redis:
            client = _get_redis_client()
            if client is not None:
                try:
                    data: str | None = client.get(f"emb:{key}")
                    if data:
                        vector = json.loads(data)
                        self._store[key] = vector
                        return vector
                except Exception as exc:
                    logger.warning("Redis cache get failed: %s", exc)
        return None

    def set(self, model: str, text: str, vector: list[float]) -> None:
        key = self.build_key(model, text)
        self._store[key] = vector
        if self._use_redis:
            client = _get_redis_client()
            if client is not None:
                try:
                    client.setex(f"emb:{key}", self._REDIS_TTL, json.dumps(vector))
                except Exception as exc:
                    logger.warning("Redis cache set failed: %s", exc)
