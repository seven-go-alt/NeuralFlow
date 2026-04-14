import httpx

from app.config import get_settings


class MCPClient:
    def __init__(self, base_url: str | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.mcp_base_url).rstrip("/")

    async def call_tool(self, tool_name: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/tools/{tool_name}",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
