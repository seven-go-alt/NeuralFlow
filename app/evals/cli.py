"""CLI entry point for running RAG evaluations.

Usage:
    uv run python -m app.evals.cli run <dataset_path> [--top-k 5]
    uv run python -m app.evals.cli compare <dataset_path> [--top-k 5]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.evals.metrics import aggregate_metrics
from app.evals.runner import build_eval_report, run_eval


def _make_mock_retrieve(
    doc_id: str = "mock_doc", content: str = "Mock content about the query."
) -> Callable[[str, int], list[dict[str, Any]]]:
    def retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
        return [
            {"document_id": doc_id, "content": content, "score": 0.95},
            {"document_id": "extra_doc", "content": "Additional context.", "score": 0.80},
        ]

    return retrieve


def _make_mock_answer(answer_prefix: str = "Based on the documents:") -> Callable[[str, str], str | None]:
    def answer(query: str, context: str) -> str | None:
        return f"{answer_prefix} {query}"

    return answer


def cmd_run(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    import asyncio

    retrieve_fn = args.retrieve if args.retrieve else _make_mock_retrieve()
    answer_fn = args.answer if args.answer else _make_mock_answer()

    results = asyncio.run(run_eval(str(dataset_path), retrieve_fn, answer_fn, top_k=args.top_k))
    metrics = aggregate_metrics(results)
    report = build_eval_report(results, metrics)
    print(report)


def cmd_compare(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    import asyncio

    from app.evals.comparison import compare_runs, format_comparison_table

    result = asyncio.run(
        compare_runs(
            str(dataset_path),
            baseline_retrieve_fn=args.baseline_retrieve or _make_mock_retrieve("baseline_doc"),
            baseline_answer_fn=args.baseline_answer or _make_mock_answer("Baseline answer:"),
            experiment_retrieve_fn=args.experiment_retrieve or _make_mock_retrieve("experiment_doc"),
            experiment_answer_fn=args.experiment_answer or _make_mock_answer("Experiment answer:"),
            top_k=args.top_k,
        )
    )
    print(format_comparison_table(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run evaluation on a dataset")
    run_parser.add_argument("dataset_path", type=str, help="Path to JSONL eval dataset")
    run_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve (default: 5)")
    run_parser.set_defaults(func=cmd_run)

    compare_parser = subparsers.add_parser("compare", help="Compare baseline vs experiment eval runs")
    compare_parser.add_argument("dataset_path", type=str, help="Path to JSONL eval dataset")
    compare_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve (default: 5)")
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
