"""
NeuralFlow Agent Evaluator
实现 LLM-as-a-judge 评估模型。
针对 Agent 的工具调用准确性、逻辑严密性和最终回答质量进行评分。
"""

import json
import asyncio
from typing import Any
from app.core.llm import LLMClient

# 评估 Prompt 模版
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
    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_llm = LLMClient(model=judge_model)

    async def evaluate_run(self, run_result: dict[str, Any]) -> dict[str, Any]:
        """评估单次 Agent 运行结果"""
        steps_str = ""
        for step in run_result.get("steps", []):
            steps_str += f"- Thought: {step.get('thought')}\n"
            steps_str += f"  Action: {step.get('action')}({step.get('action_input')})\n"
            steps_str += f"  Observation: {step.get('observation')}\n"

        eval_prompt = EVAL_PROMPT_TEMPLATE.format(
            query=run_result.get("query"),
            steps=steps_str,
            final_answer=run_result.get("final_answer")
        )

        raw_eval = await self.judge_llm.generate(eval_prompt)
        try:
            # 简单清理 LLM 输出中的 Markdown 标记
            json_str = raw_eval.strip("`").replace("json", "", 1).strip()
            return json.loads(json_str)
        except Exception as e:
            return {"error": f"Failed to parse evaluation: {str(e)}", "raw": raw_eval}

    async def run_benchmark(self, dataset: list[dict[str, Any]], agent_executor: Any):
        """运行完整基准测试集"""
        results = []
        print(f"--- Starting Benchmark: {len(dataset)} cases ---")
        
        for case in dataset:
            print(f"Testing Case: {case['query'][:50]}...")
            # 模拟 Agent 执行
            run_result = await agent_executor.execute(
                query=case["query"],
                skills=case.get("skills", []),
                session_id="eval-session"
            )
            # 进行评估
            eval_score = await self.evaluate_run(run_result)
            results.append({
                "query": case["query"],
                "run": run_result,
                "evaluation": eval_score
            })
            
        return results

async def demo_eval():
    # 模拟数据
    mock_run = {
        "query": "帮我查询最近 3 天的系统 CPU 负载并总结趋势",
        "steps": [
            {"thought": "需要查询监控数据", "action": "monitor_tool", "action_input": {"days": 3}, "observation": "CPU Load: 20%, 35%, 50%"},
        ],
        "final_answer": "过去 3 天 CPU 负载呈上升趋势，从 20% 增加到了 50%。"
    }
    
    evaluator = AgentEvaluator()
    report = await evaluator.evaluate_run(mock_run)
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # asyncio.run(demo_eval())
    print("Evaluator ready. Run within NeuralFlow env to test performance.")
