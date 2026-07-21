from __future__ import annotations

from app.evals.judge_cache import JudgeCache


class TestJudgeCache:
    def test_cache_hit(self) -> None:
        cache = JudgeCache(use_redis=False)
        assert cache.get("q", "a", "gpt-4o-mini") is None
        cache.set("q", "a", "gpt-4o-mini", {"relevance": 0.9})
        result = cache.get("q", "a", "gpt-4o-mini")
        assert result == {"relevance": 0.9}

    def test_cache_key_uniqueness(self) -> None:
        cache = JudgeCache(use_redis=False)
        cache.set("q1", "a1", "m1", {"score": 1.0})
        cache.set("q2", "a2", "m1", {"score": 0.5})
        assert cache.get("q1", "a1", "m1") == {"score": 1.0}
        assert cache.get("q2", "a2", "m1") == {"score": 0.5}

    def test_cache_ttl_expiry(self) -> None:
        import time

        cache = JudgeCache(use_redis=False, ttl_seconds=0.01)
        cache.set("q", "a", "m", {"val": 1})
        time.sleep(0.02)
        assert cache.get("q", "a", "m") is None

    def test_cache_json_roundtrip(self) -> None:
        cache = JudgeCache(use_redis=False)
        data = {
            "relevance": 0.85,
            "faithfulness": 0.92,
            "completeness": 0.78,
            "reason": "good answer",
        }
        cache.set("q", "a", "m", data)
        assert cache.get("q", "a", "m") == data
