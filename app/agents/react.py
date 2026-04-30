from __future__ import annotations

import json
import logging
from typing import Any

from litellm import acompletion

from app.config import get_settings
from app.skills.mcp_client import MCPClient
from app.skills.registry import SkillDefinition

logger = logging.getLogger("neuralflow.agents")

SYSTEM_PROMPT = (
    "你是一个具备工具调用能力的智能助手。"
    "根据用户问题，选择最合适的工具来获取信息，然后给出准确的回答。"
    "如果不需要工具就能回答，直接回复即可。"
)


def _skills_to_tools(skills: list[SkillDefinition]) -> list[dict[str, Any]]:
    """将 SkillDefinition 列表转换为 OpenAI function calling tools 格式。"""
    tools = []
    for skill in skills:
        tools.append({
            "type": "function",
            "function": {
                "name": skill.name,
                "description": skill.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": f"传递给 {skill.name} 工具的查询或指令",
                        }
                    },
                    "required": ["input"],
                },
            },
        })
    return tools


class ReActAgent:
    def __init__(
        self,
        mcp_client: MCPClient,
        max_iterations: int = 5,
    ) -> None:
        settings = get_settings()
        self.model = settings.litellm_model
        self.api_base = settings.llm_api_base
        self.api_key = settings.llm_api_key or settings.openai_api_key
        self.mcp = mcp_client
        self.max_iterations = max_iterations

    async def execute(
        self,
        query: str,
        skills: list[SkillDefinition],
        session_id: str,
        tenant_context: Any | None = None,
    ) -> dict[str, Any]:
        tools = _skills_to_tools(skills)
        skill_map = {s.name: s for s in skills}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        steps: list[dict[str, Any]] = []
        final_answer = ""

        for iteration in range(self.max_iterations):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = await acompletion(**kwargs)
            choice = response.choices[0]
            message = choice.message

            # 收集 assistant 消息（包含可能的 tool_calls）
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            tool_calls = getattr(message, "tool_calls", None)

            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            # 无 tool_calls → 视为最终回答
            if not tool_calls:
                final_answer = message.content or ""
                steps.append({
                    "iteration": iteration + 1,
                    "type": "final_answer",
                    "content": final_answer,
                })
                break

            # 逐个执行 tool_calls
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"input": tc.function.arguments}

                tool_input = args.get("input", "")

                skill = skill_map.get(tool_name)
                if not skill:
                    observation = f"Error: 未知工具 {tool_name}"
                else:
                    payload = {
                        "session_id": session_id,
                        "input": tool_input,
                    }
                    if tenant_context:
                        payload.update({
                            "tenant_id": tenant_context.tenant_id,
                            "tenant_roles": tenant_context.roles,
                        })
                    try:
                        obs_result = await self.mcp.call_tool(
                            skill.tool_name,
                            payload,
                            read_only=skill.read_only,
                        )
                        observation = json.dumps(obs_result, ensure_ascii=False)
                    except Exception as e:
                        observation = f"Error executing tool: {e}"

                steps.append({
                    "iteration": iteration + 1,
                    "type": "tool_call",
                    "tool": tool_name,
                    "input": tool_input,
                    "observation": observation,
                })

                # 将工具结果以 tool role 回传
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

        if not final_answer:
            final_answer = "未能在最大迭代次数内得出结论。"

        return {
            "query": query,
            "final_answer": final_answer,
            "steps": steps,
            "iterations": len(steps),
        }
