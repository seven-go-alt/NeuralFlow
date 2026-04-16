import fakeredis

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
