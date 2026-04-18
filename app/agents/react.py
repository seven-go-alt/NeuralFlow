from __future__ import annotations

import json
import logging
from typing import Any

from app.core.llm import LLMClient
from app.skills.mcp_client import MCPClient
from app.skills.registry import SkillDefinition

logger = logging.getLogger("neuralflow.agents")

REACT_SYSTEM_PROMPT = """你是一个具备自主决策能力的智能体 (AI Agent)。
你必须遵循 REASON -> ACT -> OBSERVATION 的循环来解决问题。

可用工具列表:
{tool_descriptions}

回答格式规范:
Thought: 你对当前步骤的思考过程。
Action: 必须是工具列表中工具的 name。
Action Input: 执行该工具所需的 JSON 格式参数。
Observation: 工具执行后的结果（由系统提供）。
... (重复上述步骤)
Final Answer: 最终总结出的答案。

注意：
1. 每次只输出一个 Action。
2. 如果你认为已经拿到了足够的信息，请输出 Final Answer。
3. 请确保 Action Input 是合法的 JSON 格式。
"""

class ReActAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        mcp_client: MCPClient,
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm_client
        self.mcp = mcp_client
        self.max_iterations = max_iterations

    async def execute(
        self,
        query: str,
        skills: list[SkillDefinition],
        session_id: str,
        tenant_context: Any | None = None,
    ) -> dict[str, Any]:
        tool_desc = "\n".join([f"- {s.name}: {s.description} (tool_name: {s.tool_name})" for s in skills])
        system_msg = REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)
        
        history = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]
        
        steps = []
        final_answer = ""
        
        for i in range(self.max_iterations):
            # 虽然 LLMClient.generate 只接收 prompt，但我们可以构造一个带上下文的 prompt
            prompt = self._build_react_prompt(history)
            response = await self.llm.generate(prompt)
            
            logger.info(f"ReAct Iteration {i+1} response: {response}")
            
            # 解析 Action
            action, action_input, thought = self._parse_response(response)
            
            steps.append({
                "iteration": i + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "response": response
            })
            
            if not action or action == "Final Answer":
                final_answer = response.split("Final Answer:")[-1].strip() if "Final Answer:" in response else response
                break
            
            # 找到对应的 skill
            target_skill = next((s for s in skills if s.name == action), None)
            if not target_skill:
                observation = f"Error: 找不到工具 {action}"
            else:
                try:
                    # 构造 MCP Payload
                    payload = {
                        "session_id": session_id,
                        "input": action_input,
                    }
                    if tenant_context:
                        payload.update({
                            "tenant_id": tenant_context.tenant_id,
                            "tenant_roles": tenant_context.roles,
                        })
                    
                    obs_result = await self.mcp.call_tool(target_skill.tool_name, payload, read_only=target_skill.read_only)
                    observation = json.dumps(obs_result, ensure_ascii=False)
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
            
            logger.info(f"Observation: {observation}")
            steps[-1]["observation"] = observation
            
            # 更新上下文
            history.append({"role": "assistant", "content": response})
            history.append({"role": "user", "content": f"Observation: {observation}"})

        return {
            "query": query,
            "final_answer": final_answer or "未能达成结论",
            "steps": steps,
            "iterations": len(steps)
        }

    def _build_react_prompt(self, history: list[dict[str, str]]) -> str:
        # 由于 LLMClient 内部会包装系统提示词，这里我们将历史拼接成一个大的上下文
        lines = []
        for msg in history:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            if msg["role"] == "system":
                prefix = "System Instructions: "
            lines.append(f"{prefix}{msg['content']}")
        return "\n".join(lines) + "\nAssistant: "

    def _parse_response(self, response: str) -> tuple[str | None, Any, str]:
        thought = ""
        action = None
        action_input = {}
        
        if "Thought:" in response:
            thought = response.split("Thought:")[1].split("Action:")[0].strip()
        
        if "Action:" in response and "Action Input:" in response:
            action = response.split("Action:")[1].split("Action Input:")[0].strip()
            input_str = response.split("Action Input:")[1].split("Observation:")[0].split("Final Answer:")[0].strip()
            try:
                action_input = json.loads(input_str)
            except:
                action_input = input_str
                
        return action, action_input, thought
