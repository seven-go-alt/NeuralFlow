"""
Multi-Agent Orchestrator
接收用户问题，分类路由到 Specialist Agent，汇总结果。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from litellm import acompletion

from app.agents.react import ReActAgent
from app.config import get_settings
from app.skills.mcp_client import MCPClient
from app.skills.registry import SkillDefinition, skill_registry

logger = logging.getLogger("neuralflow.orchestrator")

ROUTE_PROMPT = """根据用户问题，将其分类为以下三种类型之一，只返回 JSON：

- "coding": 需要写代码、调试、技术实现
- "planning": 需要制定方案、架构设计、项目规划
- "general": 日常对话、知识问答、其他

用户问题: {query}

只返回: {{"route": "coding"/"planning"/"general", "reason": "简短理由"}}
"""

CODER_SYSTEM = (
    "你是一个专业的编程助手。擅长写高质量、可运行的代码。"
    "回答时给出完整代码示例，附带简要说明。"
    "如果需要验证代码正确性，使用 python_exec 工具执行。"
)

PLANNER_SYSTEM = (
    "你是一个系统架构和项目规划专家。"
    "回答时给出结构化的方案，包含目标、步骤、风险和回滚策略。"
    "使用 Markdown 格式，层次清晰。"
)

GENERAL_SYSTEM = (
    "你是一个知识渊博的智能助手。"
    "给出准确、简洁、有帮助的回答。"
)

ROUTE_SKILL_MAP: dict[str, list[str]] = {
    "coding": ["python_exec", "file_read", "file_write", "file_list"],
    "planning": ["planner"],
    "general": [],
}


def _get_skills_for_route(route: str) -> list[SkillDefinition]:
    skill_names = ROUTE_SKILL_MAP.get(route, [])
    return [s for s in skill_registry.list_skills() if s.name in skill_names]


class AgentOrchestrator:
    def __init__(self, mcp_client: MCPClient, max_iterations: int = 5) -> None:
        settings = get_settings()
        self.model = settings.litellm_model
        self.api_base = settings.llm_api_base
        self.api_key = settings.llm_api_key or settings.openai_api_key
        self.mcp = mcp_client
        self.max_iterations = max_iterations

    async def _classify_route(self, query: str) -> dict[str, str]:
        prompt = ROUTE_PROMPT.format(query=query)
        kwargs: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            response = await acompletion(**kwargs)
            raw = response.choices[0].message.content or ""
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").replace("json", "", 1).strip()
            data = json.loads(cleaned)
            route = data.get("route", "general")
            if route not in ("coding", "planning", "general"):
                route = "general"
            return {"route": route, "reason": data.get("reason", "")}
        except Exception as exc:
            logger.warning("Route classification failed: %s", exc)
            return {"route": "general", "reason": "classification failed"}

    async def execute(
        self,
        query: str,
        session_id: str,
        tenant_context: Any | None = None,
    ) -> dict[str, Any]:
        # 1. 分类路由
        routing = await self._classify_route(query)
        route = routing["route"]

        # 2. 选择 Specialist 配置
        system_prompts = {
            "coding": CODER_SYSTEM,
            "planning": PLANNER_SYSTEM,
            "general": GENERAL_SYSTEM,
        }
        system_prompt = system_prompts.get(route, GENERAL_SYSTEM)
        skills = _get_skills_for_route(route)

        # 3. 创建 Specialist Agent 并执行
        agent = ReActAgent(
            mcp_client=self.mcp,
            max_iterations=self.max_iterations,
            max_reflections=1,
        )

        # 注入自定义 system prompt（通过 extra_context）
        result = await agent._run_react_loop(
            query=query,
            tools=ReActAgent._skills_to_tools(skills) if skills else [],
            skill_map={s.name: s for s in skills},
            session_id=session_id,
            tenant_context=tenant_context,
            extra_context=system_prompt,
        )

        final_answer, _, steps = result

        return {
            "query": query,
            "route": route,
            "route_reason": routing["reason"],
            "final_answer": final_answer,
            "steps": steps,
            "iterations": len(steps),
        }
