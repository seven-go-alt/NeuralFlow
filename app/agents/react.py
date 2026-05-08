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

REFLECTION_PROMPT = """你是一个回答质量评估员。请评估以下回答是否充分解决了用户的问题。

用户问题: {query}
智能体回答: {answer}

评估标准:
1. 是否直接回答了用户的问题
2. 是否有明显的事实错误或逻辑漏洞
3. 是否遗漏了关键信息

请只返回 JSON 格式:
{{"pass": true/false, "feedback": "如果不通过，说明需要改进的具体方面"}}

如果回答质量可接受，返回 {{"pass": true, "feedback": ""}}
"""


def _skills_to_tools(skills: list[SkillDefinition]) -> list[dict[str, Any]]:
    """将 SkillDefinition 列表转换为 OpenAI function calling tools 格式。"""
    tools = []
    for skill in skills:
        tools.append(
            {
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
            }
        )
    return tools


def _parse_reflection(raw: str) -> dict[str, Any]:
    """解析 Reflection LLM 返回的 JSON。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "pass" in data:
            return {"passed": bool(data["pass"]), "feedback": str(data.get("feedback", ""))}
    except (json.JSONDecodeError, ValueError):
        pass
    return {"passed": True, "feedback": ""}


class ReActAgent:
    def __init__(
        self,
        mcp_client: MCPClient,
        max_iterations: int = 5,
        max_reflections: int = 2,
    ) -> None:
        settings = get_settings()
        self.model = settings.litellm_model
        self.api_base = settings.llm_api_base
        self.api_key = settings.llm_api_key or settings.openai_api_key
        self.mcp = mcp_client
        self.max_iterations = max_iterations
        self.max_reflections = max_reflections

    async def _call_llm(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Any:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        response = await acompletion(**kwargs)
        return response.choices[0]

    async def _run_react_loop(
        self,
        query: str,
        tools: list[dict[str, Any]],
        skill_map: dict[str, SkillDefinition],
        session_id: str,
        tenant_context: Any | None = None,
        extra_context: str = "",
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """执行 ReAct function calling 循环，返回 (final_answer, messages, steps)。"""
        system_content = SYSTEM_PROMPT
        if extra_context:
            system_content += f"\n\n{extra_context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        steps: list[dict[str, Any]] = []
        final_answer = ""

        for iteration in range(self.max_iterations):
            choice = await self._call_llm(messages, tools or None)
            message = choice.message

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

            if not tool_calls:
                final_answer = message.content or ""
                steps.append(
                    {
                        "iteration": iteration + 1,
                        "type": "final_answer",
                        "content": final_answer,
                    }
                )
                break

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"input": tc.function.arguments}

                tool_input = args.get("input", args.get("code", str(args)))

                skill = skill_map.get(tool_name)
                if not skill:
                    observation = f"Error: 未知工具 {tool_name}"
                else:
                    payload = {"session_id": session_id, "input": tool_input}
                    if tenant_context:
                        payload.update(
                            {
                                "tenant_id": tenant_context.tenant_id,
                                "tenant_roles": tenant_context.roles,
                            }
                        )
                    try:
                        obs_result = await self.mcp.call_tool(
                            skill.tool_name,
                            payload,
                            read_only=skill.read_only,
                        )
                        observation = json.dumps(obs_result, ensure_ascii=False)
                    except Exception as e:
                        observation = f"Error executing tool: {e}"

                steps.append(
                    {
                        "iteration": iteration + 1,
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": tool_input,
                        "observation": observation,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    }
                )

        if not final_answer:
            final_answer = "未能在最大迭代次数内得出结论。"

        return final_answer, messages, steps

    async def execute(
        self,
        query: str,
        skills: list[SkillDefinition],
        session_id: str,
        tenant_context: Any | None = None,
        enable_reflection: bool = True,
    ) -> dict[str, Any]:
        tools = _skills_to_tools(skills)
        skill_map = {s.name: s for s in skills}

        # Phase 1: Initial ReAct loop
        final_answer, _, steps = await self._run_react_loop(
            query, tools, skill_map, session_id, tenant_context
        )

        # Phase 2: Reflection (self-correction)
        reflections: list[dict[str, Any]] = []
        if enable_reflection and final_answer and final_answer != "未能在最大迭代次数内得出结论。":
            for _ in range(self.max_reflections):
                reflection = await self._reflect(query, final_answer)
                reflections.append(reflection)
                if reflection["passed"]:
                    break
                # Re-run with reflection feedback
                extra_ctx = f"上次回答存在以下问题，请在本次回答中修正:\n{reflection['feedback']}"
                final_answer, _, new_steps = await self._run_react_loop(
                    query, tools, skill_map, session_id, tenant_context, extra_context=extra_ctx
                )
                steps.extend(new_steps)

        return {
            "query": query,
            "final_answer": final_answer,
            "steps": steps,
            "iterations": len(steps),
            "reflections": reflections,
        }

    async def _reflect(self, query: str, answer: str) -> dict[str, Any]:
        """评估回答质量，返回 {"passed": bool, "feedback": str}。"""
        prompt = REFLECTION_PROMPT.format(query=query, answer=answer)
        messages = [{"role": "user", "content": prompt}]
        try:
            choice = await self._call_llm(messages)
            return _parse_reflection(choice.message.content or "")
        except Exception as exc:
            logger.warning("Reflection failed: %s", exc)
            return {"passed": True, "feedback": ""}
