from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agents.orchestrator import AgentOrchestrator, _get_skills_for_route


def test_get_skills_for_known_route() -> None:
    skills = _get_skills_for_route("planning")
    names = [s.name for s in skills]
    assert "planner" in names


def test_get_skills_for_unknown_route_returns_empty() -> None:
    assert _get_skills_for_route("unknown_route") == []


def test_get_skills_for_general_route_returns_empty() -> None:
    assert _get_skills_for_route("general") == []


@pytest.mark.asyncio
async def test_classify_route_returns_general_on_api_failure(monkeypatch) -> None:
    async def fake_acompletion(**kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr("app.agents.orchestrator.acompletion", fake_acompletion)

    orchestrator = AgentOrchestrator(mcp_client=AsyncMock(), max_iterations=3)
    result = await orchestrator._classify_route("hello world")

    assert result["route"] == "general"
    assert result["reason"] == "classification failed"


@pytest.mark.asyncio
async def test_classify_route_cleans_markdown_json(monkeypatch) -> None:
    async def fake_acompletion(**kwargs):
        class FakeChoice:
            message = type(
                "Msg", (), {"content": '```json\n{"route": "coding", "reason": "needs code"}\n```'}
            )()

        class FakeResponse:
            choices = [FakeChoice()]

        return FakeResponse()

    monkeypatch.setattr("app.agents.orchestrator.acompletion", fake_acompletion)

    orchestrator = AgentOrchestrator(mcp_client=AsyncMock())
    result = await orchestrator._classify_route("write a python script")

    assert result["route"] == "coding"
    assert result["reason"] == "needs code"


@pytest.mark.asyncio
async def test_classify_route_falls_back_to_general_for_invalid_route(monkeypatch) -> None:
    async def fake_acompletion(**kwargs):
        class FakeChoice:
            message = type("Msg", (), {"content": '{"route": "invalid_route", "reason": "bad"}'})()

        class FakeResponse:
            choices = [FakeChoice()]

        return FakeResponse()

    monkeypatch.setattr("app.agents.orchestrator.acompletion", fake_acompletion)

    orchestrator = AgentOrchestrator(mcp_client=AsyncMock())
    result = await orchestrator._classify_route("do something")

    assert result["route"] == "general"


@pytest.mark.asyncio
async def test_execute_returns_correct_structure(monkeypatch) -> None:
    async def fake_acompletion(**kwargs):
        class FakeChoice:
            message = type(
                "Msg", (), {"content": '{"route": "general", "reason": "simple question"}'}
            )()

        class FakeResponse:
            choices = [FakeChoice()]

        return FakeResponse()

    monkeypatch.setattr("app.agents.orchestrator.acompletion", fake_acompletion)

    # Mock the ReAct loop to avoid full execution
    async def fake_react_loop(self, **kwargs):  # noqa: ARG002
        return ("answer", [], [])

    monkeypatch.setattr("app.agents.react.ReActAgent._run_react_loop", fake_react_loop)

    orchestrator = AgentOrchestrator(mcp_client=AsyncMock())
    result = await orchestrator.execute(query="hello", session_id="s1", tenant_context=None)

    assert result["query"] == "hello"
    assert result["route"] == "general"
    assert result["final_answer"] == "answer"
    assert result["iterations"] == 0
