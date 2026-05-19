from __future__ import annotations

import hashlib
import json


class EmbeddingCache:
    def __init__(self, use_redis: bool = True) -> None:
        self._store: dict[str, list[float]] = {}
        self._use_redis = use_redis

    def build_key(self, model: str, text: str) -> str:
        return hashlib.sha256(
            json.dumps({"model": model, "text": text}, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        key = self.build_key(model, text)
        # Try in-memory first
        cached = self._store.get(key)
        if cached is not None:
            return cached
        # Try Redis
        if self._use_redis:
            try:
                import redis as redis_module

                from app.config import get_settings

                s = get_settings()
                r = redis_module.Redis(
                    host=s.redis_host, port=s.redis_port, db=s.redis_db, decode_responses=True
                )
                data: str | None = r.get(f"emb:{key}")
                if data:
                    vector = json.loads(data)
                    self._store[key] = vector
                    return vector
            except Exception:
                pass
        return None

    def set(self, model: str, text: str, vector: list[float]) -> None:
        key = self.build_key(model, text)
        self._store[key] = vector
        if self._use_redis:
            try:
                import redis as redis_module

                from app.config import get_settings

                s = get_settings()
                r = redis_module.Redis(
                    host=s.redis_host, port=s.redis_port, db=s.redis_db, decode_responses=True
                )
                r.setex(f"emb:{key}", 3600, json.dumps(vector))
            except Exception:
                pass
