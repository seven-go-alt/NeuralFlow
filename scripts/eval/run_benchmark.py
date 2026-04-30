"""
NeuralFlow Agent Benchmark Runner
逐条执行测试用例，输出评分报告。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from app.agents.react import ReActAgent
from app.core.intent_router import IntentRouter
from app.skills.mcp_client import MCPClient
from app.skills.registry import skill_registry
from scripts.eval.evaluator import AgentEvaluator


async def run_single_case(
    case: dict,
    agent: ReActAgent,
    intent_router: IntentRouter,
    evaluator: AgentEvaluator,
    max_retries: int = 3,
) -> dict:
    query = case["query"]
    expected_keywords = case.get("expected_answer_keywords", [])
    expected_tools = case.get("expected_tools", [])
    expected_answer = case.get("expected_answer", "")

    # 意图识别
    routed = await intent_router.detect(query)
    primary_policy = routed.policies[routed.primary_intent]
    selected_skills = skill_registry.get_allowed_skills(primary_policy.skill_whitelist)

    # 执行 Agent（带重试）
    run_result: dict | None = None
    latency: float = 0.0
    for attempt in range(max_retries):
        start = time.perf_counter()
        try:
            run_result = await agent.execute(
                query=query,
                skills=selected_skills,
                session_id=f"bench-{case['id']}",
            )
            latency = round(time.perf_counter() - start, 2)
            break
        except Exception as exc:
            latency = round(time.perf_counter() - start, 2)
            if attempt < max_retries - 1:
                print(f"\n  [retry {attempt+1}/{max_retries}] {exc}")
                await asyncio.sleep(2)
            else:
                print(f"\n  [FAILED after {max_retries} retries] {exc}")
                run_result = {
                    "query": query,
                    "final_answer": f"ERROR: {exc}",
                    "steps": [],
                    "iterations": 0,
                }

    assert run_result is not None

    # 关键词评分
    kw_score = AgentEvaluator.keyword_score(run_result["final_answer"], expected_keywords)

    # Ground Truth 评分（ROUGE-L）
    gt_score = AgentEvaluator.rouge_l_score(run_result["final_answer"], expected_answer)

    # 工具使用检查
    used_tools = [s.get("tool", "") for s in run_result.get("steps", []) if s.get("type") == "tool_call"]
    tool_match = True
    if expected_tools:
        tool_match = any(t in used_tools for t in expected_tools)

    # LLM 评分（可选，较慢）
    llm_eval = {}
    try:
        llm_eval = await evaluator.evaluate_run(run_result)
    except Exception as e:
        llm_eval = {"error": str(e)}

    return {
        "id": case["id"],
        "category": case["category"],
        "query": query,
        "intent": routed.primary_intent,
        "final_answer": run_result["final_answer"][:200],
        "iterations": run_result["iterations"],
        "latency_s": latency,
        "used_tools": used_tools,
        "expected_tools": expected_tools,
        "tool_match": tool_match,
        "keyword_score": kw_score,
        "ground_truth_score": gt_score,
        "llm_eval": llm_eval,
    }


async def main() -> None:
    # 加载测试用例
    cases_path = Path(__file__).parent / "test_cases.json"
    with open(cases_path) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} test cases\n")

    # 初始化组件
    mcp_client = MCPClient()
    intent_router = IntentRouter()
    agent = ReActAgent(mcp_client=mcp_client)
    evaluator = AgentEvaluator()

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']}: {case['query'][:50]}...", end=" ", flush=True)
        result = await run_single_case(case, agent, intent_router, evaluator)
        results.append(result)
        kw = result["keyword_score"]
        tool = "OK" if result["tool_match"] else "MISS"
        gt = result["ground_truth_score"]
        print(f"kw={kw} gt={gt} tool={tool} {result['latency_s']}s")

    # 汇总
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    total = len(results)
    avg_kw = sum(r["keyword_score"] for r in results) / total if total else 0
    avg_gt = sum(r["ground_truth_score"] for r in results) / total if total else 0
    tool_hits = sum(1 for r in results if r["tool_match"])
    avg_latency = sum(r["latency_s"] for r in results) / total if total else 0
    avg_iter = sum(r["iterations"] for r in results) / total if total else 0

    # 按类别分组
    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    print(f"\nTotal cases: {total}")
    print(f"Avg keyword score: {avg_kw:.1f}/10")
    print(f"Avg ground truth (ROUGE-L): {avg_gt:.1f}/10")
    print(f"Tool match rate: {tool_hits}/{total}")
    print(f"Avg latency: {avg_latency:.2f}s")
    print(f"Avg iterations: {avg_iter:.1f}")

    print("\n--- By Category ---")
    for cat, items in categories.items():
        cat_kw = sum(r["keyword_score"] for r in items) / len(items)
        cat_gt = sum(r["ground_truth_score"] for r in items) / len(items)
        cat_tool = sum(1 for r in items if r["tool_match"])
        print(f"  {cat:20s}  cases={len(items)}  kw={cat_kw:.1f}  gt={cat_gt:.1f}  tool={cat_tool}/{len(items)}")

    # 保存报告
    report_path = Path(__file__).parent / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "summary": {
                    "total_cases": total,
                    "avg_keyword_score": round(avg_kw, 2),
                    "avg_ground_truth_score": round(avg_gt, 2),
                    "tool_match_rate": f"{tool_hits}/{total}",
                    "avg_latency_s": round(avg_latency, 2),
                    "avg_iterations": round(avg_iter, 1),
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
