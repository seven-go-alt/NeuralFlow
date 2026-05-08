from __future__ import annotations

from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_fixed

from app.config import get_settings


class MCPToolDescriptor(BaseModel):
    name: str
    description: str
    read_only: bool = True
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPToolExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        is_retryable: bool = False,
        should_trigger_fallback: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.is_retryable = is_retryable
        self.should_trigger_fallback = should_trigger_fallback


class MCPClient:
    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        retry_attempts: int | None = None,
        retry_backoff_seconds: float | None = None,
        client_factory: Callable[[httpx.Timeout], httpx.AsyncClient] | None = None,
        tool_routes: dict[str, str] | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.mcp_base_url).rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds or settings.mcp_timeout_seconds)
        self.retry_attempts = retry_attempts or settings.mcp_retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds or settings.mcp_retry_backoff_seconds
        self.client_factory = client_factory or self._default_client_factory
        self.tool_routes: dict[str, str] = tool_routes or {}
        if settings.mcp_code_server_url:
            for tool in ("python_exec",):
                self.tool_routes.setdefault(tool, settings.mcp_code_server_url.rstrip("/"))
        if settings.mcp_filesystem_server_url:
            for tool in ("file_read", "file_write", "file_list"):
                self.tool_routes.setdefault(tool, settings.mcp_filesystem_server_url.rstrip("/"))

    async def list_tools(self) -> list[MCPToolDescriptor]:
        all_tools: list[MCPToolDescriptor] = []
        seen_bases: set[str] = set()

        # Main MCP server
        try:
            payload = await self._request_json("GET", "/tools")
            tools = payload.get("tools", []) if isinstance(payload, dict) else []
            all_tools.extend(MCPToolDescriptor.model_validate(item) for item in tools)
            seen_bases.add(self.base_url)
        except Exception:
            pass

        # Additional tool servers from routes
        for tool_name, base in self.tool_routes.items():
            if base in seen_bases:
                continue
            seen_bases.add(base)
            try:
                payload = await self._request_json("GET", "/tools", base_url=base)
                tools = payload.get("tools", []) if isinstance(payload, dict) else []
                all_tools.extend(MCPToolDescriptor.model_validate(item) for item in tools)
            except Exception:
                pass

        return all_tools

    async def call_tool(
        self, tool_name: str, payload: dict[str, Any], *, read_only: bool = True
    ) -> dict[str, Any]:
        if not read_only:
            raise MCPToolExecutionError(
                "Mutating MCP tools are disabled by default",
                status_code=403,
                is_retryable=False,
                should_trigger_fallback=False,
            )
        base = self.tool_routes.get(tool_name, self.base_url)
        result = await self._request_json(
            "POST", f"/tools/{tool_name}", json_body=payload, base_url=base
        )
        return result if isinstance(result, dict) else {"result": result}

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any] | list[Any] | str:
        target_base = base_url or self.base_url
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_fixed(self.retry_backoff_seconds),
            retry=retry_if_exception(self._is_retryable_error),
            reraise=True,
        )
        try:
            async for attempt in retryer:
                with attempt:
                    async with self.client_factory(self.timeout) as client:
                        response = await client.request(
                            method, f"{target_base}{path}", json=json_body
                        )
                    if response.status_code in self.RETRYABLE_STATUS_CODES:
                        raise MCPToolExecutionError(
                            f"MCP server returned retryable status {response.status_code}",
                            status_code=response.status_code,
                            is_retryable=True,
                            should_trigger_fallback=True,
                        )
                    if response.is_error:
                        raise MCPToolExecutionError(
                            f"MCP server returned status {response.status_code}",
                            status_code=response.status_code,
                            is_retryable=False,
                            should_trigger_fallback=response.status_code >= 500,
                        )
                    return response.json()
        except MCPToolExecutionError:
            raise
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            raise MCPToolExecutionError(
                f"MCP transport unavailable: {exc}",
                status_code=503,
                is_retryable=True,
                should_trigger_fallback=True,
            ) from exc
        except ValueError as exc:
            raise MCPToolExecutionError(
                "MCP server returned invalid JSON",
                status_code=502,
                is_retryable=False,
                should_trigger_fallback=True,
            ) from exc
        raise MCPToolExecutionError("MCP request failed unexpectedly", status_code=500)

    def _default_client_factory(self, timeout: httpx.Timeout) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout)

    @staticmethod
    def _is_retryable_error(exc: BaseException) -> bool:
        return isinstance(exc, MCPToolExecutionError) and exc.is_retryable


__all__ = ["MCPClient", "MCPToolDescriptor", "MCPToolExecutionError"]
