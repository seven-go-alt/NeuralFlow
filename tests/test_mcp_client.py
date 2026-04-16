from __future__ import annotations

import httpx
import pytest

from app.skills.mcp_client import MCPClient, MCPToolExecutionError


class HttpxClientFactory:
    def __init__(self, transport: httpx.BaseTransport | httpx.AsyncBaseTransport) -> None:
        self.transport = transport

    def __call__(self, timeout: httpx.Timeout) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, timeout=timeout)


@pytest.mark.asyncio
async def test_mcp_client_lists_tools_with_retry_on_transient_failure() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        assert request.url.path == "/tools"
        if attempts["count"] == 1:
            return httpx.Response(status_code=503, json={"detail": "busy"})
        return httpx.Response(
            status_code=200,
            json={
                "tools": [
                    {
                        "name": "memory",
                        "description": "查询长期记忆",
                        "read_only": True,
                    }
                ]
            },
        )

    client = MCPClient(
        base_url="http://mcp.test",
        client_factory=HttpxClientFactory(httpx.MockTransport(handler)),
        retry_attempts=2,
        retry_backoff_seconds=0,
    )

    tools = await client.list_tools()

    assert attempts["count"] == 2
    assert len(tools) == 1
    assert tools[0].name == "memory"
    assert tools[0].read_only is True


@pytest.mark.asyncio
async def test_mcp_client_blocks_mutating_tool_calls_by_default() -> None:
    client = MCPClient(base_url="http://mcp.test")

    with pytest.raises(MCPToolExecutionError) as exc_info:
        await client.call_tool(
            "filesystem",
            {"path": "/tmp/demo.txt", "content": "hello"},
            read_only=False,
        )

    assert exc_info.value.should_trigger_fallback is False
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_mcp_client_marks_transport_failures_as_offline_fallback_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = MCPClient(
        base_url="http://mcp.test",
        client_factory=HttpxClientFactory(httpx.MockTransport(handler)),
        retry_attempts=2,
        retry_backoff_seconds=0,
    )

    with pytest.raises(MCPToolExecutionError) as exc_info:
        await client.call_tool("memory", {"input": "hi"})

    assert exc_info.value.is_retryable is True
    assert exc_info.value.should_trigger_fallback is True
    assert exc_info.value.status_code == 503
