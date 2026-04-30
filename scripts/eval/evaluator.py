"""
NeuralFlow Agent Evaluator
LLM-as-a-judge 评测 + 关键词自动评分。
"""

import json
from typing import Any

from app.core.llm import LLMClient

EVAL_PROMPT_TEMPLATE = """你是一个专业的 AI Agent 评测员。请根据以下信息对智能体的表现进行评分（0-10分）。

用户问题: {query}
智能体执行步骤:
{steps}
最终回答: {final_answer}

评测维度:
1. 工具使用准确性 (Tool Accuracy): 是否使用了最合适的工具？参数是否正确？
2. 逻辑严密性 (Logic Reasoning): Thought 过程是否合乎逻辑？是否解决了用户的意图？
3. 最终质量 (Final Quality): 答案是否准确、简洁且对用户有帮助？

请返回 JSON 格式结果:
{{
  "scores": {{
    "tool_accuracy": 0,
    "logic_reasoning": 0,
    "final_quality": 0
  }},
  "total_score": 0,
  "feedback": "详细的评价反馈"
}}
"""


class AgentEvaluator:
    def __init__(self, judge_model: str | None = None) -> None:
        self.judge_llm = LLMClient(model=judge_model)

    async def evaluate_run(self, run_result: dict[str, Any]) -> dict[str, Any]:
        """评估单次 Agent 运行结果。"""
        steps_str = ""
        for step in run_result.get("steps", []):
            step_type = step.get("type", "")
            if step_type == "tool_call":
                steps_str += f"- Tool Call: {step.get('tool')}({step.get('input')})\n"
                steps_str += f"  Observation: {step.get('observation')}\n"
            elif step_type == "final_answer":
                steps_str += f"- Final Answer: {step.get('content', '')[:200]}\n"
            else:
                steps_str += f"- Thought: {step.get('thought', '')}\n"
                steps_str += f"  Action: {step.get('action', '')}\n"
                steps_str += f"  Observation: {step.get('observation', '')}\n"

        eval_prompt = EVAL_PROMPT_TEMPLATE.format(
            query=run_result.get("query", ""),
            steps=steps_str or "(无步骤，直接回答)",
            final_answer=run_result.get("final_answer", ""),
        )

        raw_eval = await self.judge_llm.generate(eval_prompt)
        try:
            json_str = raw_eval.strip()
            if json_str.startswith("```"):
                json_str = json_str.strip("`").replace("json", "", 1).strip()
            return json.loads(json_str)
        except Exception:
            return {"error": "Failed to parse evaluation", "raw": raw_eval}

    @staticmethod
    def keyword_score(answer: str, expected_keywords: list[str]) -> float:
        """基于关键词的自动评分（0-10）。"""
        if not expected_keywords:
            return 10.0
        hits = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
        return round(hits / len(expected_keywords) * 10, 1)
