from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["case_id"]: item for item in report.get("results", [])}


def compare_summary(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    current_summary = current.get("summary", {})
    baseline_summary = baseline.get("summary", {})

    if current_summary.get("pass_rate", 0) < baseline_summary.get("pass_rate", 0):
        failures.append(
            f"pass_rate regressed: {baseline_summary.get('pass_rate')} -> {current_summary.get('pass_rate')}"
        )

    base_citation = baseline_summary.get("citation_coverage")
    curr_citation = current_summary.get("citation_coverage")
    if (
        base_citation is not None
        and curr_citation is not None
        and curr_citation < base_citation - 0.05
    ):
        failures.append(f"citation_coverage regressed: {base_citation} -> {curr_citation}")

    base_keyword = baseline_summary.get("avg_keyword_score")
    curr_keyword = current_summary.get("avg_keyword_score")
    if base_keyword is not None and curr_keyword is not None and curr_keyword < base_keyword - 0.05:
        failures.append(f"avg_keyword_score regressed: {base_keyword} -> {curr_keyword}")

    base_latency = baseline_summary.get("avg_latency_ms")
    curr_latency = current_summary.get("avg_latency_ms")
    if base_latency and curr_latency and curr_latency > base_latency * 1.3:
        failures.append(f"avg_latency_ms regressed: {base_latency} -> {curr_latency}")

    return failures


def compare_cases(current: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    current_cases = index_results(current)
    baseline_cases = index_results(baseline)

    for case_id, baseline_case in baseline_cases.items():
        current_case = current_cases.get(case_id)
        if current_case is None:
            failures.append(f"case missing from current report: {case_id}")
            continue
        if baseline_case.get("status") == "passed" and current_case.get("status") == "failed":
            failures.append(f"case regressed: {case_id} passed -> failed")

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NeuralFlow eval report to baseline")
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current = load_report(args.current)
    baseline = load_report(args.baseline)
    failures = compare_summary(current, baseline) + compare_cases(current, baseline)

    output = {
        "current": args.current,
        "baseline": args.baseline,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
