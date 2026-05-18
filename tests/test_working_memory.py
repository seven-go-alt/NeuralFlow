import fakeredis
import redis

from app.memory.working import WorkingMemory


def test_working_memory_keeps_recent_messages_only() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    memory = WorkingMemory(session_id="demo", max_turns=3, client=client)

    memory.add_message("user", "one")
    memory.add_message("assistant", "two")
    memory.add_message("user", "three")
    memory.add_message("assistant", "four")

    assert memory.get_messages() == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def test_working_memory_uses_tenant_prefixed_keys() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)

    memory_a = WorkingMemory(session_id="shared", tenant_id="tenant-a", client=client)
    memory_b = WorkingMemory(session_id="shared", tenant_id="tenant-b", client=client)

    memory_a.add_message("user", "from-a")
    memory_b.add_message("user", "from-b")

    assert memory_a.key == "tenant:tenant-a:session:shared:history"
    assert memory_b.key == "tenant:tenant-b:session:shared:history"
    assert memory_a.get_messages() == [{"role": "user", "content": "from-a"}]
    assert memory_b.get_messages() == [{"role": "user", "content": "from-b"}]


def test_working_memory_add_delegates_to_add_message() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    memory = WorkingMemory(session_id="test-delegate", max_turns=5, client=client)

    memory.add("user", "via-add")
    assert memory.get_messages() == [{"role": "user", "content": "via-add"}]


def test_working_memory_fallback_on_redis_error() -> None:
    class BrokenRedis:
        def lpush(self, key, value):
            raise redis.RedisError("redis is down")

        def lrange(self, key, start, end):
            raise redis.RedisError("redis is down")

        def ltrim(self, key, start, end):
            raise redis.RedisError("redis is down")

        def delete(self, key):
            raise redis.RedisError("redis is down")

    memory = WorkingMemory(session_id="fallback-test", max_turns=5, client=BrokenRedis())

    memory.add_message("user", "first")
    memory.add_message("assistant", "second")

    messages = memory.get_messages()
    assert messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    assert memory._fallback_enabled


def test_working_memory_pop_all_messages_works_in_fallback() -> None:
    class BrokenRedis:
        def lpush(self, key, value):
            raise redis.RedisError("down")

        def lrange(self, key, start, end):
            raise redis.RedisError("down")

        def ltrim(self, key, start, end):
            raise redis.RedisError("down")

        def delete(self, key):
            raise redis.RedisError("down")

    memory = WorkingMemory(session_id="pop-test", max_turns=5, client=BrokenRedis())
    memory.add_message("user", "to-pop")
    popped = memory.pop_all_messages()

    assert popped == [{"role": "user", "content": "to-pop"}]
    assert memory.get_messages() == []


def test_working_memory_pop_archive_batch_edge_cases() -> None:
    class BrokenRedis:
        def lpush(self, key, value):
            raise redis.RedisError("down")

        def lrange(self, key, start, end):
            raise redis.RedisError("down")

        def ltrim(self, key, start, end):
            raise redis.RedisError("down")

        def delete(self, key):
            raise redis.RedisError("down")

    memory = WorkingMemory(session_id="archive-test", max_turns=2, client=BrokenRedis())

    # Triggers archive via overflow from max_turns=2
    memory.add_message("user", "m1")
    memory.add_message("assistant", "m2")
    memory.add_message("user", "m3")

    # pop_archive_batch retrieves archived messages
    batch = memory.pop_archive_batch()
    assert len(batch) == 1
    assert batch[0]["content"] == "m1"

    # clear_archive_batch in fallback mode
    memory.clear_archive_batch()


def test_working_memory_clear_archive_batch_via_fallback() -> None:
    class BrokenRedis:
        def __init__(self):
            self._data: dict[str, list[str]] = {}

        def lpush(self, key, value):
            raise redis.RedisError("down")

        def lrange(self, key, start, end):
            raise redis.RedisError("down")

        def ltrim(self, key, start, end):
            raise redis.RedisError("down")

        def delete(self, key):
            raise redis.RedisError("down")

    memory = WorkingMemory(session_id="clear-test", max_turns=1, client=BrokenRedis())
    memory.add_message("user", "a")
    memory.add_message("assistant", "b")  # triggers overflow → archive

    assert memory._fallback_enabled
    assert len(memory._fallback_archive) == 1

    memory.clear_archive_batch(batch_size=1)
    assert len(memory._fallback_archive) == 0


def test_working_memory_overflow_archives_old_messages() -> None:
    client = fakeredis.FakeStrictRedis(decode_responses=True)
    memory = WorkingMemory(session_id="overflow-archive", max_turns=2, client=client)

    memory.add_message("user", "old")
    memory.add_message("assistant", "mid")
    memory.add_message("user", "new")

    assert memory.get_messages() == [
        {"role": "assistant", "content": "mid"},
        {"role": "user", "content": "new"},
    ]

    batch = memory.pop_archive_batch()
    assert len(batch) == 1
    assert batch[0]["content"] == "old"

    memory.clear_archive_batch(batch_size=1)
    assert memory.pop_archive_batch() == []
