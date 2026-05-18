from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    value: Any
    expires_at: float


_T = Any


class TTLCache:
    """In-memory cache with TTL and LRU eviction."""

    def __init__(self, max_size: int = 1000, default_ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> _T | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return entry.value

    def set(self, key: str, value: _T, ttl_seconds: float | None = None) -> None:
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        expires_at = time.monotonic() + (ttl_seconds or self._default_ttl)
        self._store[key] = CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> bool:
        try:
            del self._store[key]
            return True
        except KeyError:
            return False

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @staticmethod
    def build_key(*parts: str) -> str:
        raw = json.dumps(list(parts), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CacheManager:
    """Multi-namespace cache manager."""

    def __init__(self, default_ttl_seconds: float = 300.0) -> None:
        self._caches: dict[str, TTLCache] = {}
        self._default_ttl = default_ttl_seconds

    def namespace(self, name: str, ttl_seconds: float | None = None) -> TTLCache:
        if name not in self._caches:
            self._caches[name] = TTLCache(default_ttl_seconds=ttl_seconds or self._default_ttl)
        return self._caches[name]

    def clear_all(self) -> None:
        for cache in self._caches.values():
            cache.clear()

    def clear_namespace(self, name: str) -> None:
        cache = self._caches.get(name)
        if cache:
            cache.clear()
