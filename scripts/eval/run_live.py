"""Live eval runner — full RAG eval baseline.

Usage (from repo root):
    uv run python -m scripts.eval.run_live
    uv run python -m scripts.eval.run_live --top-k 5 --datasets-dir data/eval/datasets --output-dir docs/eval-reports
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.evals.datasets import load_cases
from app.evals.factories import (
    make_live_answer_eval_fn,
    make_live_answer_fn,
    make_live_retrieve_fn,
)
from app.evals.metrics import aggregate_metrics
from app.evals.runner import build_eval_report, run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live RAG eval baseline runner")
    parser.add_argument(
        "--datasets-dir",
        default="data/eval/datasets",
        help="Directory containing .jsonl datasets",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="docs/eval-reports",
        help="Directory to write baseline report",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-Judge to save cost",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = sorted(datasets_dir.glob("*.jsonl"))
    if not dataset_files:
        print(f"error: no .jsonl files in {datasets_dir}", file=sys.stderr)
        return 1

    all_cases = []
    for path in dataset_files:
        all_cases.extend(load_cases(path))

    print(f"Running live eval on {len(all_cases)} cases from {len(dataset_files)} dataset(s)")
    print(f"top_k={args.top_k}, judge={'off' if args.no_judge else 'on'}")

    retrieve_fn = make_live_retrieve_fn()
    answer_fn = make_live_answer_fn()
    answer_eval_fn = None if args.no_judge else make_live_answer_eval_fn()

    all_results = []
    for path in dataset_files:
        print(f"  Running {path.name}...")
        results = await run_eval(
            str(path),
            retrieve_fn,
            answer_fn,
            top_k=args.top_k,
            answer_eval_fn=answer_eval_fn,
        )
        all_results.extend(results)

    metrics = aggregate_metrics(all_results)
    report = build_eval_report(all_results, metrics)

    total_tokens = sum((r.token_usage_json or {}).get("total_tokens", 0) for r in all_results)
    total_cost = sum((r.token_usage_json or {}).get("cost_usd", 0.0) for r in all_results)
    if total_tokens > 0:
        report += f"\n\n## Token Usage\n\n- Total tokens: {total_tokens}\n- Estimated cost: ${total_cost:.4f}\n"

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"baseline-{timestamp}.md"
    report_path.write_text(report)

    json_path = output_dir / f"baseline-{timestamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "top_k": args.top_k,
                "total_cases": metrics.total_cases,
                "metrics": {
                    "retrieval_hit_rate": metrics.retrieval_hit_rate,
                    "citation_accuracy": metrics.citation_accuracy,
                    "keyword_coverage": metrics.keyword_coverage,
                    "no_answer_accuracy": metrics.no_answer_accuracy,
                    "average_latency_ms": metrics.average_latency_ms,
                    "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
                    "average_precision_at_k": metrics.average_precision_at_k,
                    "average_recall_at_k": metrics.average_recall_at_k,
                    "average_answer_relevance": metrics.average_answer_relevance,
                    "average_answer_faithfulness": metrics.average_answer_faithfulness,
                    "average_answer_completeness": metrics.average_answer_completeness,
                },
                "token_usage": {
                    "total_tokens": total_tokens,
                    "cost_usd": total_cost,
                },
            },
            indent=2,
        )
    )

    print(f"\nReport saved to {report_path}")
    print(f"JSON data saved to {json_path}")

    print(f"\nhit@{args.top_k}: {metrics.retrieval_hit_rate:.1%}")
    print(f"keyword coverage: {metrics.keyword_coverage:.1%}")
    return 0 if metrics.retrieval_hit_rate >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
