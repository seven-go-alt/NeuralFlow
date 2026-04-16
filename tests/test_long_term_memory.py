import asyncio

from app.memory.long_term import LongTermMemory
from app.utils.vector_client import InMemoryVectorClient


def test_long_term_memory_save_summary_persists_metadata_and_document() -> None:
    client = InMemoryVectorClient()
    memory = LongTermMemory(client=client, collection_name="test_collection")

    item_id = memory.save_summary(
        summary="用户喜欢简洁回答，并在上次讨论了 Redis 滑动窗口。",
        metadata={"session_id": "session-1", "source": "archive"},
    )

    collection = client.get_or_create_collection("test_collection")
    stored = collection.documents[0]

    assert item_id
    assert stored["id"] == item_id
    assert stored["document"] == "用户喜欢简洁回答，并在上次讨论了 Redis 滑动窗口。"
    assert stored["metadata"]["session_id"] == "session-1"
    assert stored["metadata"]["source"] == "archive"
    assert "created_at" in stored["metadata"]


def test_long_term_memory_search_returns_relevant_documents() -> None:
    client = InMemoryVectorClient()
    memory = LongTermMemory(client=client, collection_name="memory_search")
    memory.save_summary("用户上次在讨论 ChromaDB 部署方案。", {"session_id": "s1"})
    memory.save_summary("用户偏好简洁中文回复。", {"session_id": "s1"})

    results = asyncio.run(memory.search("ChromaDB", top_k=2))

    assert results == ["用户上次在讨论 ChromaDB 部署方案。"]


def test_long_term_memory_search_isolated_by_tenant() -> None:
    client = InMemoryVectorClient()
    tenant_a = LongTermMemory(client=client, collection_name="memory_search_tenant", tenant_id="tenant-a")
    tenant_b = LongTermMemory(client=client, collection_name="memory_search_tenant", tenant_id="tenant-b")

    tenant_a.save_summary("tenant-a 的 ChromaDB 记录。", {"session_id": "s1"})
    tenant_b.save_summary("tenant-b 的 ChromaDB 记录。", {"session_id": "s1"})

    results_a = asyncio.run(tenant_a.search("ChromaDB", top_k=2, session_id="s1"))
    results_b = asyncio.run(tenant_b.search("ChromaDB", top_k=2, session_id="s1"))

    assert results_a == ["tenant-a 的 ChromaDB 记录。"]
    assert results_b == ["tenant-b 的 ChromaDB 记录。"]
