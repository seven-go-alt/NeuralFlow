from __future__ import annotations

import time

from app.utils.cache import CacheManager, TTLCache


class TestTTLCache:
    def test_get_set(self) -> None:
        cache = TTLCache(max_size=100, default_ttl_seconds=60.0)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self) -> None:
        cache = TTLCache()
        assert cache.get("missing") is None

    def test_expired_entry(self) -> None:
        cache = TTLCache(default_ttl_seconds=0.01)
        cache.set("key", "value")
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_lru_eviction(self) -> None:
        cache = TTLCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)
        assert cache.size == 3
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("d") is not None

    def test_lru_hit_preserves(self) -> None:
        cache = TTLCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # hit "a", so "b" should be evicted next
        cache.set("c", 3)
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_delete(self) -> None:
        cache = TTLCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None

    def test_delete_missing(self) -> None:
        cache = TTLCache()
        assert cache.delete("missing") is False

    def test_clear(self) -> None:
        cache = TTLCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.size == 0

    def test_build_key(self) -> None:
        k1 = TTLCache.build_key("a", "b")
        k2 = TTLCache.build_key("b", "a")
        k3 = TTLCache.build_key("a", "b")
        assert len(k1) == 16
        assert k1 == k3  # same input = same key
        assert k1 != k2  # different order = different key (sorted)

    def test_set_overwrites(self) -> None:
        cache = TTLCache()
        cache.set("k", "v1")
        cache.set("k", "v2")
        assert cache.get("k") == "v2"


class TestCacheManager:
    def test_namespace(self) -> None:
        mgr = CacheManager()
        c1 = mgr.namespace("ns1")
        c2 = mgr.namespace("ns1")
        assert c1 is c2  # same namespace = same cache

    def test_isolated_namespaces(self) -> None:
        mgr = CacheManager()
        c1 = mgr.namespace("ns1")
        c2 = mgr.namespace("ns2")
        c1.set("k", "v1")
        assert c2.get("k") is None

    def test_clear_all(self) -> None:
        mgr = CacheManager()
        mgr.namespace("ns1").set("k", "v")
        mgr.namespace("ns2").set("k", "v")
        mgr.clear_all()
        assert mgr.namespace("ns1").size == 0
        assert mgr.namespace("ns2").size == 0

    def test_clear_namespace(self) -> None:
        mgr = CacheManager()
        mgr.namespace("ns1").set("k", "v")
        mgr.namespace("ns2").set("k", "v")
        mgr.clear_namespace("ns1")
        assert mgr.namespace("ns1").size == 0
        assert mgr.namespace("ns2").size > 0

    def test_custom_ttl_per_namespace(self) -> None:
        mgr = CacheManager(default_ttl_seconds=60.0)
        c = mgr.namespace("fast", ttl_seconds=1.0)
        c.set("k", "v")
        assert c.get("k") is not None
