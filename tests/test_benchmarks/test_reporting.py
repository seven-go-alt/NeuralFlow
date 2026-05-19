from __future__ import annotations

import json

from benchmarks.models import BenchmarkResult, LatencyStats
from benchmarks.reporting.compare import compare_with_baseline
from benchmarks.reporting.html_reporter import html_report
from benchmarks.reporting.json_reporter import to_json_report


def _make_result(name: str, mean_ms: float) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        suite="retrieval",
        latency=LatencyStats(mean_ms=mean_ms, min_ms=mean_ms, max_ms=mean_ms, samples=1),
        recall_at_k=0.5,
    )


def test_json_report_serialization() -> None:
    r = _make_result("test", 10.0)
    report = to_json_report([r])
    data = json.loads(report)
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "test"


def test_html_report_generation() -> None:
    results = [_make_result("bench1", 10.0), _make_result("bench2", 20.0)]
    html = html_report(results)
    assert "<!DOCTYPE html>" in html
    assert "bench1" in html
    assert "bench2" in html


def test_compare_no_baseline() -> None:
    cur = [_make_result("test", 10.0)]
    result = compare_with_baseline(cur, [])
    assert len(result["regressions"]) == 0
    assert len(result["improvements"]) == 0


def test_compare_detects_regression() -> None:
    cur = [_make_result("test", 20.0)]
    base = [_make_result("test", 10.0)]
    result = compare_with_baseline(cur, base, threshold=0.15)
    assert len(result["regressions"]) == 1
    assert len(result["improvements"]) == 0


def test_compare_detects_improvement() -> None:
    cur = [_make_result("test", 8.0)]
    base = [_make_result("test", 10.0)]
    result = compare_with_baseline(cur, base, threshold=0.15)
    assert len(result["improvements"]) == 1
    assert len(result["regressions"]) == 0
