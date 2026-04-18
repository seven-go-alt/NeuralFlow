from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.skills.registry import SkillRegistry


class StubWorkingMemory:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: list[tuple[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append((role, content))


class StubContextBuilder:
    def __init__(self, session_id: str, working_mem=None, long_mem=None, tenant_id: str | None = None) -> None:
        self.session_id = session_id

    async def build_prompt(self, user_query: str, intent: str, **kwargs) -> str:
        return f"intent={intent};query={user_query}"


class StubRouter:
    async def detect(self, text: str):
        from app.core.intent_router import IntentDetectionResult, IntentPolicy

        return IntentDetectionResult(
            intents=["general"],
            primary_intent="general",
            used_fallback=False,
            policies={"general": IntentPolicy(memory_strategy="working_only", skill_whitelist=[])},
        )


class StubLLMClient:
    async def generate(self, prompt: str) -> str:
        return "base reply"


def test_chat_endpoint_runs_on_response_generated_plugin(monkeypatch, tmp_path, capsys) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "test_logger.py").write_text(
        "def on_response_generated(payload):\n"
        "    print(f\"PLUGIN_HOOK response={payload['reply']} tenant={payload['tenant_id']}\")\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("NEURALFLOW_PLUGIN_DIR", str(plugin_dir))
    monkeypatch.setattr("app.main.intent_router", StubRouter())
    monkeypatch.setattr("app.main.skill_registry", SkillRegistry())
    monkeypatch.setattr("app.main.llm_client", StubLLMClient())
    monkeypatch.setattr("app.main.WorkingMemory", StubWorkingMemory)
    monkeypatch.setattr("app.main.ContextBuilder", StubContextBuilder)

    from app.plugins.manager import PluginManager

    monkeypatch.setattr("app.main.plugin_manager", PluginManager.from_env())

    client = TestClient(app)
    response = client.post(
        "/chat",
        headers={"X-Tenant-ID": "tenant-hook"},
        json={"session_id": "plugin-session", "message": "hello"},
    )

    assert response.status_code == 200
    captured = capsys.readouterr()
    assert "PLUGIN_HOOK response=base reply tenant=tenant-hook" in captured.out
