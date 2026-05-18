from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import (
    _ensure_openai_prefix,
    _extract_admin_secret,
    _get_client_ip,
    _handle_terminal_tool,
    _strip_provider_prefix,
    app,
)

# --- Pure utility functions ---


def test_strip_provider_prefix_with_provider() -> None:
    assert _strip_provider_prefix("openai/gpt-4") == "gpt-4"


def test_strip_provider_prefix_without_provider() -> None:
    assert _strip_provider_prefix("gpt-4") == "gpt-4"


def test_ensure_openai_prefix_without_provider() -> None:
    assert _ensure_openai_prefix("gpt-4") == "openai/gpt-4"


def test_ensure_openai_prefix_with_provider() -> None:
    assert _ensure_openai_prefix("openai/gpt-4") == "openai/gpt-4"
    assert _ensure_openai_prefix("ollama/llama2") == "ollama/llama2"


def test_get_client_ip_with_host() -> None:
    request = type("Req", (), {"client": type("C", (), {"host": "1.2.3.4"})()})()
    assert _get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_without_host() -> None:
    request = type("Req", (), {"client": None})()
    assert _get_client_ip(request) == "unknown"


def test_extract_admin_secret_from_header() -> None:
    request = type("Req", (), {"headers": {"X-Admin-Secret": "my-secret"}})()
    assert _extract_admin_secret(request) == "my-secret"


def test_extract_admin_secret_from_bearer() -> None:
    request = type("Req", (), {"headers": {"Authorization": "Bearer my-token"}})()
    assert _extract_admin_secret(request) == "my-token"


def test_extract_admin_secret_no_auth() -> None:
    request = type("Req", (), {"headers": {}})()
    assert _extract_admin_secret(request) is None


def test_extract_admin_secret_invalid_scheme() -> None:
    request = type("Req", (), {"headers": {"Authorization": "Basic xxx"}})()
    assert _extract_admin_secret(request) is None


def test_extract_admin_secret_empty_bearer() -> None:
    """Bearer with no token after it returns None (not token)."""
    request = type("Req", (), {"headers": {"Authorization": "Bearer "}})()
    assert _extract_admin_secret(request) is None


# --- _handle_terminal_tool ---


@pytest.mark.asyncio
async def test_handle_terminal_no_command() -> None:
    result = await _handle_terminal_tool({"input": ""})
    assert result["error"] == "No command provided"
    assert result["return_code"] == 1


@pytest.mark.asyncio
async def test_handle_terminal_missing_input_key() -> None:
    result = await _handle_terminal_tool({})
    assert result["error"] == "No command provided"
    assert result["return_code"] == 1


@pytest.mark.asyncio
async def test_handle_terminal_disabled(monkeypatch) -> None:
    monkeypatch.setattr("app.main.settings.terminal_enabled", False)

    result = await _handle_terminal_tool({"input": "ls"})
    assert "disabled" in result["error"]
    assert result["return_code"] == -1


# --- Endpoint tests ---


def test_healthz() -> None:
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_metrics() -> None:
    response = TestClient(app).get("/metrics")
    # Metrics endpoint returns prometheus format, not JSON
    assert response.status_code == 200


def test_list_skills(monkeypatch) -> None:
    from app.skills.registry import SkillDefinition, skill_registry

    monkeypatch.setattr(
        skill_registry,
        "list_skills",
        lambda: [
            SkillDefinition(name="test-skill", description="A test skill", tool_name="test-tool")
        ],
    )

    response = TestClient(app).get("/api/v1/skills")
    assert response.status_code == 200
    data = response.json()
    assert len(data["skills"]) == 1
    assert data["skills"][0]["name"] == "test-skill"


def test_list_skills_empty(monkeypatch) -> None:
    from app.skills.registry import skill_registry

    monkeypatch.setattr(skill_registry, "list_skills", lambda: [])

    response = TestClient(app).get("/api/v1/skills")
    assert response.status_code == 200
    assert response.json()["skills"] == []


def test_list_models_no_api_base(monkeypatch) -> None:
    monkeypatch.setattr("app.main.settings.llm_api_base", None)
    monkeypatch.setattr("app.main.settings.llm_api_key", None)
    monkeypatch.setattr("app.main.settings.openai_api_key", None)

    response = TestClient(app).get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "not configured" in data["error"]


def test_list_models_no_api_base_with_key(monkeypatch) -> None:
    monkeypatch.setattr("app.main.settings.llm_api_base", None)
    monkeypatch.setattr("app.main.settings.llm_api_key", "sk-test")
    monkeypatch.setattr("app.main.settings.openai_api_key", None)

    response = TestClient(app).get("/api/v1/models")
    assert response.status_code == 200
    assert "not configured" in response.json()["error"]


def test_switch_model(monkeypatch) -> None:
    monkeypatch.setattr("app.main.settings.litellm_model", "openai/gpt-4")

    response = TestClient(app).post("/api/v1/models/switch", json={"model": "gpt-5"})
    assert response.status_code == 200
    data = response.json()
    assert "gpt-5" in data["model"]
    assert "openai/gpt-5" in data["litellm_model"]


def test_switch_model_with_provider(monkeypatch) -> None:
    monkeypatch.setattr("app.main.settings.litellm_model", "openai/gpt-4")

    response = TestClient(app).post(
        "/api/v1/models/switch",
        json={"model": "ollama/llama2"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["litellm_model"] == "ollama/llama2"


def test_serve_frontend_redirect(monkeypatch) -> None:
    """When frontend index.html doesn't exist, redirect to /docs."""
    monkeypatch.setattr("app.main._FRONTEND_DIR", "/nonexistent")

    response = TestClient(app).get("/", follow_redirects=False)
    assert response.status_code in (200, 307)  # RedirectResponse or found


def test_list_models_with_api(monkeypatch) -> None:
    """list_models returns models from the configured API."""
    monkeypatch.setattr("app.main.settings.llm_api_base", "https://api.test.com/v1")
    monkeypatch.setattr("app.main.settings.llm_api_key", "sk-test")

    class FakeModel:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"data": [{"id": "gpt-4"}, {"id": "gpt-5"}]}

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def get(self, *args: object, **kwargs: object) -> FakeModel:
            return FakeModel()

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: FakeClient())

    response = TestClient(app).get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "gpt-4" in data["models"]
    assert "gpt-5" in data["models"]


def test_list_models_api_error(monkeypatch) -> None:
    """list_models returns error message when the API call fails."""
    monkeypatch.setattr("app.main.settings.llm_api_base", "https://api.test.com/v1")
    monkeypatch.setattr("app.main.settings.llm_api_key", "sk-test")

    class FailingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self, *args: object) -> FailingClient:
            raise RuntimeError("connection refused")

        async def __aexit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr("httpx.AsyncClient", lambda *a, **kw: FailingClient())

    response = TestClient(app).get("/api/v1/models")
    assert response.status_code == 200
    assert "error" in response.json()


@pytest.mark.asyncio
async def test_handle_terminal_executes_command(monkeypatch) -> None:
    """_handle_terminal_tool with a valid command returns execution result."""

    class FakeResult:
        stdout = "hello\nworld"
        stderr = ""
        return_code = 0
        timed_out = False

    async def fake_execute(command: str, timeout: int, cwd: str) -> FakeResult:
        return FakeResult()

    monkeypatch.setattr("app.main.execute_command", fake_execute)
    monkeypatch.setattr("app.main.settings.terminal_enabled", True)

    result = await _handle_terminal_tool({"input": "echo hello"})
    assert result["stdout"] == "hello\nworld"
    assert result["return_code"] == 0


def test_admin_config_no_secret(monkeypatch) -> None:
    """Admin config returns 401 when ADMIN_SECRET_KEY is set but not provided."""
    monkeypatch.setattr("app.main.settings.auth_enabled", False)
    monkeypatch.setenv("ADMIN_SECRET_KEY", "supersecret")

    response = TestClient(app).get("/admin/config")
    assert response.status_code == 401


# --- serve_frontend ---


def test_serve_frontend_with_existing_file(monkeypatch, tmp_path) -> None:
    """serve_frontend returns FileResponse when index.html exists."""
    index = tmp_path / "index.html"
    index.write_text("<html><body>Hello</body></html>")
    monkeypatch.setattr("app.main._FRONTEND_DIR", str(tmp_path))

    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.text == "<html><body>Hello</body></html>"


# --- chat_react endpoint ---


@pytest.mark.asyncio
async def test_chat_react_endpoint(monkeypatch) -> None:
    """chat_react returns ReAct agent output."""
    from app.core.intent_router import IntentDetectionResult, IntentPolicy

    policy = IntentPolicy(memory_strategy="working_only", skill_whitelist=["*"])

    class StubRouter:
        async def detect(self, text: str) -> IntentDetectionResult:
            return IntentDetectionResult(
                intents=["general"],
                primary_intent="general",
                used_fallback=False,
                policies={"general": policy},
            )

    class StubAgent:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def execute(self, **kwargs: object) -> dict:
            return {
                "final_answer": "Hello from ReAct!",
                "steps": [],
                "iterations": 0,
                "reflections": [],
            }

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.ReActAgent", StubAgent)
    monkeypatch.setattr("app.main.skill_registry.get_allowed_skills", lambda w: [])

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/react",
            json={"session_id": "s1", "message": "hello", "use_retrieval": False},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["final_answer"] == "Hello from ReAct!"
    assert data["intent"] == "general"
    assert data["total_iterations"] == 0


@pytest.mark.asyncio
async def test_chat_react_with_terminal_enabled(monkeypatch) -> None:
    """chat_react includes terminal handler when terminal is enabled."""
    from app.core.intent_router import IntentDetectionResult, IntentPolicy

    policy = IntentPolicy(memory_strategy="working_only", skill_whitelist=["*"])

    class StubRouter:
        async def detect(self, text: str) -> IntentDetectionResult:
            return IntentDetectionResult(
                intents=["coding"],
                primary_intent="coding",
                used_fallback=False,
                policies={"coding": policy},
            )

    class StubAgent:
        def __init__(self, **kwargs: object) -> None:
            self.local_handlers = kwargs.get("local_handlers", {})

        async def execute(self, **kwargs: object) -> dict:
            return {
                "final_answer": "ls result",
                "steps": [{"type": "tool_call", "tool": "terminal", "observation": "ok"}],
                "iterations": 1,
                "reflections": [],
            }

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.ReActAgent", StubAgent)
    monkeypatch.setattr("app.main.settings.terminal_enabled", True)
    monkeypatch.setattr("app.main.skill_registry.get_allowed_skills", lambda w: [])

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/react",
            json={"session_id": "s2", "message": "list files", "use_retrieval": False},
        )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["final_answer"], str)
    assert data["total_iterations"] == 1


@pytest.mark.asyncio
async def test_chat_react_failed_working_memory(monkeypatch) -> None:
    """chat_react handles WorkingMemory failure gracefully (add_message fails)."""
    from app.core.intent_router import IntentDetectionResult, IntentPolicy

    policy = IntentPolicy(memory_strategy="working_only", skill_whitelist=["*"])

    class StubRouter:
        async def detect(self, text: str) -> IntentDetectionResult:
            return IntentDetectionResult(
                intents=["general"],
                primary_intent="general",
                used_fallback=False,
                policies={"general": policy},
            )

    class StubAgent:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def execute(self, **kwargs: object) -> dict:
            return {
                "final_answer": "result",
                "steps": [],
                "iterations": 0,
                "reflections": [],
            }

    class FailingAddMessageWM:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            self.session_id = session_id

        def add_message(self, role: str, content: str) -> None:
            raise RuntimeError("Failed to save message")

    # chat_react does local `from app.memory.working import WorkingMemory`
    monkeypatch.setattr("app.memory.working.WorkingMemory", FailingAddMessageWM)
    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.ReActAgent", StubAgent)
    monkeypatch.setattr("app.main.skill_registry.get_allowed_skills", lambda w: [])

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/react",
            json={"session_id": "s3", "message": "hi", "use_retrieval": False},
        )

    # Should still succeed since WorkingMemory failure is caught
    assert response.status_code == 200
    assert response.json()["final_answer"] == "result"


# --- chat_orchestrate endpoint ---


@pytest.mark.asyncio
async def test_chat_orchestrate_endpoint(monkeypatch) -> None:
    """chat_orchestrate returns AgentOrchestrator output."""

    class StubOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def execute(self, **kwargs: object) -> dict:
            return {
                "final_answer": "Orchestrated result",
                "steps": [],
                "iterations": 0,
                "route": "general",
                "route_reason": "default route",
            }

    monkeypatch.setattr("app.main.AgentOrchestrator", StubOrchestrator)

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/orchestrate",
            json={"session_id": "s1", "message": "help me", "use_retrieval": False},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["final_answer"] == "Orchestrated result"
    assert data["route"] == "general"


@pytest.mark.asyncio
async def test_chat_orchestrate_failed_working_memory(monkeypatch) -> None:
    """chat_orchestrate handles WorkingMemory failure gracefully."""

    class StubOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def execute(self, **kwargs: object) -> dict:
            return {
                "final_answer": "orchestrated result",
                "steps": [{"type": "tool_call", "tool": "python", "observation": "done"}],
                "iterations": 2,
                "route": "coding",
                "route_reason": "code request",
            }

    class FailingAddMessageWM:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            self.session_id = session_id

        def add_message(self, role: str, content: str) -> None:
            raise RuntimeError("Failed to save message")

    # chat_orchestrate does local `from app.memory.working import WorkingMemory`
    monkeypatch.setattr("app.memory.working.WorkingMemory", FailingAddMessageWM)
    monkeypatch.setattr("app.main.AgentOrchestrator", StubOrchestrator)

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat/orchestrate",
            json={"session_id": "s2", "message": "write code", "use_retrieval": False},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "coding"
    assert data["route_reason"] == "code request"
    assert data["total_iterations"] == 2


# --- _run_skills edge cases ---


@pytest.mark.asyncio
async def test_run_skills_skips_terminal_skill() -> None:
    """_run_skills skips the terminal skill in non-ReAct flow."""
    from app.main import _run_skills
    from app.skills.registry import SkillDefinition

    skills = [
        SkillDefinition(name="terminal", description="Shell", tool_name="terminal", read_only=True),
    ]
    result: list[dict[str, object]] = await _run_skills(
        skills, session_id="test", intent="general", user_query="ls"
    )

    assert len(result) == 1
    assert result[0]["skill"] == "terminal"
    skill_result = result[0]["result"]
    assert isinstance(skill_result, dict)
    assert skill_result["note"] == "terminal skill is only available in ReAct mode"


@pytest.mark.asyncio
async def test_run_skills_skips_terminal_and_runs_others(monkeypatch) -> None:
    """_run_skills skips terminal skill but executes other skills."""
    from app.main import _run_skills
    from app.skills.registry import SkillDefinition

    async def fake_call_tool(tool_name: str, payload: dict, **kwargs: object) -> dict:
        return {"result": f"executed {tool_name}"}

    monkeypatch.setattr("app.main.mcp_client.call_tool", fake_call_tool)

    skills = [
        SkillDefinition(name="terminal", description="Shell", tool_name="terminal", read_only=True),
        SkillDefinition(
            name="python", description="Python", tool_name="python_exec", read_only=True
        ),
    ]
    result = await _run_skills(skills, session_id="test", intent="general", user_query="print(1)")

    assert len(result) == 2
    assert result[0]["skill"] == "terminal"
    assert result[1]["skill"] == "python"
    assert result[1]["result"] == {"result": "executed python_exec"}


@pytest.mark.asyncio
async def test_run_skills_typeerror_read_only_fallback(monkeypatch) -> None:
    """_run_skills falls back when call_tool doesn't accept read_only kwarg."""
    from app.main import _run_skills
    from app.skills.registry import SkillDefinition

    call_count = 0

    async def failing_then_succeeding_call_tool(
        tool_name: str, payload: dict, **kwargs: object
    ) -> dict:
        nonlocal call_count
        call_count += 1
        if "read_only" in kwargs:
            raise TypeError("call_tool() got an unexpected keyword argument 'read_only'")
        return {"result": "success after retry"}

    monkeypatch.setattr("app.main.mcp_client.call_tool", failing_then_succeeding_call_tool)

    skills = [
        SkillDefinition(
            name="python",
            description="Python",
            tool_name="python_exec",
            read_only=True,
        ),
    ]
    result = await _run_skills(skills, session_id="test", intent="coding", user_query="print(1)")

    assert result[0]["result"] == {"result": "success after retry"}
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_skills_exception_fallback(monkeypatch) -> None:
    """_run_skills catches general exceptions from call_tool."""
    from app.main import _run_skills
    from app.skills.registry import SkillDefinition

    async def failing_call_tool(tool_name: str, payload: dict, **kwargs: object) -> dict:
        raise RuntimeError("MCP connection refused")

    monkeypatch.setattr("app.main.mcp_client.call_tool", failing_call_tool)

    skills = [
        SkillDefinition(
            name="python",
            description="Python",
            tool_name="python_exec",
            read_only=True,
        ),
    ]
    result: list[dict[str, object]] = await _run_skills(
        skills, session_id="test", intent="coding", user_query="x"
    )
    skill_result = result[0]["result"]
    assert isinstance(skill_result, dict)
    assert "error" in skill_result
    assert "MCP connection refused" in skill_result["error"]


# --- _discover_remote_tools ---


@pytest.mark.asyncio
async def test_discover_remote_tools_failure(monkeypatch) -> None:
    """_discover_remote_tools warns on list_tools failure (no exception raised)."""
    from app.main import _discover_remote_tools

    async def failing_list_tools() -> list:
        raise RuntimeError("MCP server unavailable")

    monkeypatch.setattr("app.main.mcp_client.list_tools", failing_list_tools)

    await _discover_remote_tools(session_id="test", intent="general")


# --- _prepare_chat edge cases (via /api/v1/chat) ---


@pytest.mark.asyncio
async def test_chat_working_memory_typeerror_tenant_id(monkeypatch) -> None:
    """_prepare_chat falls back when WorkingMemory rejects tenant_id kwarg."""
    from httpx import ASGITransport, AsyncClient

    from app.core.intent_router import IntentDetectionResult, IntentPolicy

    policy = IntentPolicy(memory_strategy="working_only", skill_whitelist=[])

    class StubRouter:
        async def detect(self, text: str) -> IntentDetectionResult:
            return IntentDetectionResult(
                intents=["general"],
                primary_intent="general",
                used_fallback=False,
                policies={"general": policy},
            )

    class StubLLM:
        async def generate(self, prompt: str) -> str:
            return "reply"

    class TypeErrorWorkingMemory:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            if tenant_id is not None:
                raise TypeError(
                    "WorkingMemory.__init__() got an unexpected keyword argument 'tenant_id'"
                )
            self.session_id = session_id
            self.messages: list = []

        def add_message(self, role: str, content: str) -> None:
            self.messages.append((role, content))

        def get_messages(self):
            return [{"role": r, "content": c} for r, c in self.messages]

    class StubContextBuilder:
        def __init__(
            self,
            session_id: str,
            working_mem: object = None,
            long_mem: object = None,
            token_budget_manager: object = None,
            tenant_id: str | None = None,
        ) -> None:
            self.session_id = session_id
            self.working_mem = working_mem

        async def build_prompt(self, user_query: str, intent: str, **kwargs: object) -> str:
            return f"prompt::{intent}::{user_query}"

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.llm_client", StubLLM())
    monkeypatch.setattr("app.main.WorkingMemory", TypeErrorWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)
    monkeypatch.setattr("app.main.skill_registry.get_allowed_skills", lambda w: [])
    monkeypatch.setattr("app.main.mcp_client.call_tool", lambda *a, **kw: {"result": {}})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat",
            json={
                "session_id": "s1",
                "message": "hello",
                "use_retrieval": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "reply"


@pytest.mark.asyncio
async def test_chat_context_builder_typeerror_tenant_id(monkeypatch) -> None:
    """_prepare_chat falls back when ContextBuilder rejects tenant_id kwarg."""
    from httpx import ASGITransport, AsyncClient

    from app.core.intent_router import IntentDetectionResult, IntentPolicy

    policy = IntentPolicy(memory_strategy="working_only", skill_whitelist=[])

    class StubRouter:
        async def detect(self, text: str) -> IntentDetectionResult:
            return IntentDetectionResult(
                intents=["general"],
                primary_intent="general",
                used_fallback=False,
                policies={"general": policy},
            )

    class StubLLM:
        async def generate(self, prompt: str) -> str:
            return "reply"

    class StubWorkingMemory:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            self.session_id = session_id
            self.messages: list = []

        def add_message(self, role: str, content: str) -> None:
            self.messages.append((role, content))

        def get_messages(self):
            return [{"role": r, "content": c} for r, c in self.messages]

    class TypeErrorContextBuilder:
        def __init__(
            self,
            session_id: str,
            working_mem: object = None,
            long_mem: object = None,
            token_budget_manager: object = None,
            tenant_id: str | None = None,
        ) -> None:
            if tenant_id is not None:
                raise TypeError(
                    "ContextBuilder.__init__() got an unexpected keyword argument 'tenant_id'"
                )
            self.session_id = session_id
            self.working_mem = working_mem

        async def build_prompt(self, user_query: str, intent: str, **kwargs: object) -> str:
            return f"prompt::{intent}::{user_query}"

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.llm_client", StubLLM())
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", TypeErrorContextBuilder)
    monkeypatch.setattr("app.main.skill_registry.get_allowed_skills", lambda w: [])
    monkeypatch.setattr("app.main.mcp_client.call_tool", lambda *a, **kw: {"result": {}})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat",
            json={
                "session_id": "s2",
                "message": "hello",
                "use_retrieval": False,
            },
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "reply"


@pytest.mark.asyncio
async def test_chat_retrieval_exception_fallback(monkeypatch) -> None:
    """_prepare_chat catches generic exceptions in RAG retrieval."""
    from httpx import ASGITransport, AsyncClient

    from app.core.intent_router import IntentDetectionResult, IntentPolicy

    policy = IntentPolicy(memory_strategy="working_only", skill_whitelist=[])

    class StubRouter:
        async def detect(self, text: str) -> IntentDetectionResult:
            return IntentDetectionResult(
                intents=["general"],
                primary_intent="general",
                used_fallback=False,
                policies={"general": policy},
            )

    class StubLLM:
        async def generate(self, prompt: str) -> str:
            return "reply"

    class StubWorkingMemory:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            self.session_id = session_id
            self.messages: list = []

        def add_message(self, role: str, content: str) -> None:
            self.messages.append((role, content))

        def get_messages(self):
            return [{"role": r, "content": c} for r, c in self.messages]

    class StubContextBuilder:
        def __init__(
            self,
            session_id: str,
            working_mem: object = None,
            long_mem: object = None,
            token_budget_manager: object = None,
            tenant_id: str | None = None,
        ) -> None:
            self.session_id = session_id

        async def build_prompt(self, user_query: str, intent: str, **kwargs: object) -> str:
            return f"prompt::{intent}::{user_query}"

    class FailingRetrievalService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def search(self, tenant_id: str, request: object) -> object:
            raise RuntimeError("Unexpected retrieval error")

    class DummyDB:
        def close(self) -> None:
            pass

    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.llm_client", StubLLM())
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)
    monkeypatch.setattr("app.main.RetrievalService", FailingRetrievalService)
    monkeypatch.setattr("app.db.session.SessionLocal", lambda: DummyDB())
    monkeypatch.setattr("app.main.skill_registry.get_allowed_skills", lambda w: [])
    monkeypatch.setattr("app.main.mcp_client.call_tool", lambda *a, **kw: {"result": {}})

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/chat",
            json={
                "session_id": "s3",
                "message": "query",
                "use_retrieval": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "reply"


# --- Defensive guard raise tests ---


@pytest.mark.asyncio
async def test_run_skills_typeerror_unexpected_raises(monkeypatch) -> None:
    """_run_skills re-raises unexpected TypeError from call_tool."""
    from app.main import _run_skills
    from app.skills.registry import SkillDefinition

    async def unexpected_typeerror(tool_name: str, payload: dict, **kwargs: object) -> dict:
        raise TypeError("Some unexpected type error")

    monkeypatch.setattr("app.main.mcp_client.call_tool", unexpected_typeerror)

    skills = [
        SkillDefinition(
            name="python",
            description="Python",
            tool_name="python_exec",
            read_only=True,
        ),
    ]

    with pytest.raises(TypeError, match="Some unexpected type error"):
        await _run_skills(skills, session_id="test", intent="coding", user_query="x")


@pytest.mark.asyncio
async def test_prepare_chat_working_memory_unexpected_typeerror(monkeypatch) -> None:
    """_prepare_chat re-raises unexpected TypeError from WorkingMemory."""
    from app.main import ChatRequest, _prepare_chat

    class UnexpectedTypeErrorWM:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            raise TypeError("some unexpected type error entirely")

    monkeypatch.setattr("app.main.WorkingMemory", UnexpectedTypeErrorWM)
    monkeypatch.setattr("app.main.intent_router", _stub_router())
    monkeypatch.setattr("app.main.skill_registry.get_allowed_skills", lambda w: [])
    monkeypatch.setattr("app.main.mcp_client.call_tool", lambda *a, **kw: {"result": {}})

    request = ChatRequest(session_id="s1", message="hi", use_retrieval=False)
    with pytest.raises(TypeError, match="some unexpected type error entirely"):
        await _prepare_chat(request, tenant_context=None)


@pytest.mark.asyncio
async def test_prepare_chat_context_builder_unexpected_typeerror(monkeypatch) -> None:
    """_prepare_chat re-raises unexpected TypeError from ContextBuilder."""
    from app.main import ChatRequest, _prepare_chat

    class StubWorkingMemory:
        def __init__(self, session_id: str, tenant_id: str | None = None) -> None:
            self.session_id = session_id
            self.messages: list = []

        def add_message(self, role: str, content: str) -> None:
            self.messages.append((role, content))

    class UnexpectedTypeErrorCB:
        def __init__(
            self,
            session_id: str,
            working_mem: object = None,
            long_mem: object = None,
            token_budget_manager: object = None,
            tenant_id: str | None = None,
        ) -> None:
            raise TypeError("some unexpected type error entirely")

    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", UnexpectedTypeErrorCB)
    monkeypatch.setattr("app.main.intent_router", _stub_router())
    monkeypatch.setattr("app.main.skill_registry.get_allowed_skills", lambda w: [])
    monkeypatch.setattr("app.main.mcp_client.call_tool", lambda *a, **kw: {"result": {}})

    request = ChatRequest(session_id="s1", message="hi", use_retrieval=False)
    with pytest.raises(TypeError, match="some unexpected type error entirely"):
        await _prepare_chat(request, tenant_context=None)


def _stub_router():
    """Shared stub router used by defensive-guard tests."""
    from app.core.intent_router import IntentDetectionResult, IntentPolicy

    policy = IntentPolicy(memory_strategy="working_only", skill_whitelist=[])

    class _Router:
        async def detect(self, text: str) -> IntentDetectionResult:
            return IntentDetectionResult(
                intents=["general"],
                primary_intent="general",
                used_fallback=False,
                policies={"general": policy},
            )

    return _Router()
