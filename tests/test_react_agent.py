"""Tests for ReAct Agent with Reflection."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.react import ReActAgent, _parse_reflection
from app.skills.mcp_client import MCPClient
from app.skills.registry import SkillDefinition


def _make_choice(content: str = "", tool_calls: list | None = None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def _make_tool_call(name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class StubMCP:
    def __init__(self, responses: dict | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._responses = responses or {}

    async def call_tool(self, tool_name, payload, read_only=True):
        self.calls.append((tool_name, payload))
        return self._responses.get(tool_name, {"result": f"done: {tool_name}"})


@pytest.fixture
def coding_skills():
    return [
        SkillDefinition(name="python_exec", description="执行代码", tool_name="python_exec"),
        SkillDefinition(name="file_read", description="读文件", tool_name="file_read"),
    ]


@pytest.mark.asyncio
async def test_agent_direct_answer(coding_skills):
    """LLM 直接回答，不调用工具。"""
    mcp = StubMCP()
    agent = ReActAgent(mcp_client=mcp, max_iterations=3)

    with patch("app.agents.react.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _make_choice(content="Python 是一种编程语言。")
        result = await agent.execute(
            query="什么是 Python？",
            skills=coding_skills,
            session_id="s1",
            enable_reflection=False,
        )

    assert result["final_answer"] == "Python 是一种编程语言。"
    assert result["iterations"] == 1
    assert len(mcp.calls) == 0


@pytest.mark.asyncio
async def test_agent_calls_tool_then_answers(coding_skills):
    """LLM 先调用工具，再给出最终回答。"""
    mcp = StubMCP({"python_exec": {"stdout": "5\n", "stderr": "", "return_code": 0}})
    agent = ReActAgent(mcp_client=mcp, max_iterations=3)

    tc = _make_tool_call("python_exec", json.dumps({"input": "print(2+3)"}))

    with patch("app.agents.react.acompletion", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            _make_choice(content="", tool_calls=[tc]),
            _make_choice(content="2+3 的结果是 5。"),
        ]
        result = await agent.execute(
            query="计算 2+3",
            skills=coding_skills,
            session_id="s1",
            enable_reflection=False,
        )

    assert "5" in result["final_answer"]
    assert len(mcp.calls) == 1
    assert mcp.calls[0][0] == "python_exec"
    tool_steps = [s for s in result["steps"] if s["type"] == "tool_call"]
    assert len(tool_steps) == 1
    assert tool_steps[0]["tool"] == "python_exec"


@pytest.mark.asyncio
async def test_reflection_passes_on_first_try(coding_skills):
    """回答质量好，Reflection 第一次就通过。"""
    mcp = StubMCP()
    agent = ReActAgent(mcp_client=mcp, max_iterations=3, max_reflections=2)

    with patch("app.agents.react.acompletion", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            _make_choice(content="Python 是一种高级编程语言。"),
            _make_choice(content='{"pass": true, "feedback": ""}'),
        ]
        result = await agent.execute(
            query="什么是 Python？",
            skills=coding_skills,
            session_id="s1",
            enable_reflection=True,
        )

    assert result["final_answer"] == "Python 是一种高级编程语言。"
    assert len(result["reflections"]) == 1
    assert result["reflections"][0]["passed"] is True


@pytest.mark.asyncio
async def test_reflection_triggers_retry(coding_skills):
    """回答不完整，Reflection 触发重试后通过。"""
    mcp = StubMCP()
    agent = ReActAgent(mcp_client=mcp, max_iterations=3, max_reflections=2)

    with patch("app.agents.react.acompletion", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            # 初始回答
            _make_choice(content="Python 很好。"),
            # Reflection: 不通过
            _make_choice(content='{"pass": false, "feedback": "回答太简略，缺少具体描述"}'),
            # 重试回答
            _make_choice(content="Python 是一种高级编程语言，广泛用于数据科学和 Web 开发。"),
            # Reflection: 通过
            _make_choice(content='{"pass": true, "feedback": ""}'),
        ]
        result = await agent.execute(
            query="什么是 Python？",
            skills=coding_skills,
            session_id="s1",
            enable_reflection=True,
        )

    assert "高级编程语言" in result["final_answer"]
    assert len(result["reflections"]) == 2
    assert result["reflections"][0]["passed"] is False
    assert result["reflections"][1]["passed"] is True


@pytest.mark.asyncio
async def test_reflection_max_limit(coding_skills):
    """Reflection 达到最大次数后停止。"""
    mcp = StubMCP()
    agent = ReActAgent(mcp_client=mcp, max_iterations=3, max_reflections=2)

    with patch("app.agents.react.acompletion", new_callable=AsyncMock) as mock:
        mock.side_effect = [
            _make_choice(content="回答1"),
            _make_choice(content='{"pass": false, "feedback": "不够好"}'),
            _make_choice(content="回答2"),
            _make_choice(content='{"pass": false, "feedback": "还是不够好"}'),
            _make_choice(content="回答3"),
        ]
        result = await agent.execute(
            query="测试",
            skills=coding_skills,
            session_id="s1",
            enable_reflection=True,
        )

    assert len(result["reflections"]) == 2
    assert all(not r["passed"] for r in result["reflections"])


@pytest.mark.asyncio
async def test_reflection_disabled(coding_skills):
    """Reflection 关闭时不执行评估。"""
    mcp = StubMCP()
    agent = ReActAgent(mcp_client=mcp, max_iterations=3, max_reflections=2)

    with patch("app.agents.react.acompletion", new_callable=AsyncMock) as mock:
        mock.return_value = _make_choice(content="直接回答")
        result = await agent.execute(
            query="测试",
            skills=coding_skills,
            session_id="s1",
            enable_reflection=False,
        )

    assert result["reflections"] == []


def test_parse_reflection_valid():
    result = _parse_reflection('{"pass": false, "feedback": "缺少细节"}')
    assert result["passed"] is False
    assert result["feedback"] == "缺少细节"


def test_parse_reflection_with_markdown():
    result = _parse_reflection('```json\n{"pass": true, "feedback": ""}\n```')
    assert result["passed"] is True


def test_parse_reflection_invalid_json():
    result = _parse_reflection("not json at all")
    assert result["passed"] is True  # defaults to pass on parse failure
