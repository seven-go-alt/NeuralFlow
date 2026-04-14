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
