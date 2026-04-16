import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.intent_router import IntentDetectionResult, IntentPolicy
from app.main import app
from app.memory.working import WorkingMemory
from app.skills.registry import SkillRegistry


class StubWorkingMemory:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))

    def get_messages(self):
        return [{"role": role, "content": content} for role, content in self.messages]


class StubContextBuilder:
    def __init__(self, session_id: str, working_mem=None, long_mem=None) -> None:
        self.session_id = session_id
        self.working_mem = working_mem

    async def build_prompt(
        self,
        user_query: str,
        intent: str,
        memory_strategy: str | None = None,
        skill_whitelist: list[str] | None = None,
        skill_results: list[dict] | None = None,
    ) -> str:
        rendered_results = ", ".join(
            f"{item['skill']}: {item['result']['result']}" for item in (skill_results or [])
        )
        return (
            f"intent={intent}\n"
            f"memory={memory_strategy}\n"
            f"skills={skill_whitelist}\n"
            f"results={rendered_results}\n"
            f"query={user_query}"
        )


class StubLLMClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "ok"


class StubMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name: str, payload: dict) -> dict:
        self.calls.append((tool_name, payload))
        return {"result": f"{tool_name} handled {payload['input']}"}


class BrokenRedisClient:
    def lpush(self, *args, **kwargs):
        raise RedisConnectionError("redis unavailable")

    def lrange(self, *args, **kwargs):
        raise RedisConnectionError("redis unavailable")

    def ltrim(self, *args, **kwargs):
        raise RedisConnectionError("redis unavailable")

    def delete(self, *args, **kwargs):
        raise RedisConnectionError("redis unavailable")


class StubRouter:
    async def detect(self, text: str) -> IntentDetectionResult:
        assert text == "帮我规划并顺手查一下历史"
        return IntentDetectionResult(
            intents=["planning"],
            primary_intent="planning",
            used_fallback=False,
            policies={
                "planning": IntentPolicy(
                    memory_strategy="working_only",
                    skill_whitelist=["planner", "memory"],
                )
            },
        )


@pytest.mark.asyncio
async def test_skill_registry_filters_requested_skills() -> None:
    registry = SkillRegistry()
    registry.register("planner", "规划任务")
    registry.register("memory", "查询历史")

    filtered = registry.get_allowed_skills(["memory", "unknown", "planner"])

    assert [skill.name for skill in filtered] == ["memory", "planner"]



def test_chat_endpoint_invokes_whitelisted_skills_and_includes_results(monkeypatch) -> None:
    registry = SkillRegistry()
    registry.register("planner", "规划任务")
    registry.register("memory", "查询历史")
    registry.register("python", "执行代码")

    llm = StubLLMClient()
    mcp = StubMCPClient()

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.skill_registry", registry)
    monkeypatch.setattr("app.main.mcp_client", mcp)
    monkeypatch.setattr("app.main.llm_client", llm)
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)

    client = TestClient(app)
    response = client.post("/chat", json={"session_id": "s1", "message": "帮我规划并顺手查一下历史"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "planning"
    assert body["used_skills"] == ["planner", "memory"]
    assert body["skill_results"] == [
        {"skill": "planner", "result": {"result": "planner handled 帮我规划并顺手查一下历史"}},
        {"skill": "memory", "result": {"result": "memory handled 帮我规划并顺手查一下历史"}},
    ]
    assert len(mcp.calls) == 2
    assert [call[0] for call in mcp.calls] == ["planner", "memory"]
    assert "planner handled" in body["prompt"]
    assert "memory handled" in body["prompt"]
    assert llm.prompts and llm.prompts[0] == body["prompt"]



def test_chat_endpoint_degrades_gracefully_when_redis_is_unavailable(monkeypatch) -> None:
    registry = SkillRegistry()
    llm = StubLLMClient()

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.skill_registry", registry)
    monkeypatch.setattr("app.main.mcp_client", StubMCPClient())
    monkeypatch.setattr("app.main.llm_client", llm)
    monkeypatch.setattr(
        "app.main.WorkingMemory",
        lambda session_id: WorkingMemory(session_id=session_id, client=BrokenRedisClient()),
    )
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)

    client = TestClient(app)
    response = client.post("/chat", json={"session_id": "s-redis-down", "message": "帮我规划并顺手查一下历史"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "ok"
    assert body["used_skills"] == []
    assert llm.prompts and "query=帮我规划并顺手查一下历史" in llm.prompts[0]



def test_skills_endpoint_lists_registered_skills(monkeypatch) -> None:
    registry = SkillRegistry()
    registry.register("planner", "规划任务")
    registry.register("memory", "查询历史")
    monkeypatch.setattr("app.main.skill_registry", registry)

    client = TestClient(app)
    response = client.get("/api/skills")

    assert response.status_code == 200
    assert response.json() == {
        "skills": [
            {"name": "planner", "description": "规划任务"},
            {"name": "memory", "description": "查询历史"},
        ]
    }
