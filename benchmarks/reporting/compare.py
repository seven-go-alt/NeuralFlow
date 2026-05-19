from __future__ import annotations

from benchmarks.models import BenchmarkResult


def compare_with_baseline(
    current: list[BenchmarkResult],
    baseline: list[BenchmarkResult],
    threshold: float = 0.15,
) -> dict[str, list[str]]:
    """Compare current results with baseline, flagging regressions."""
    regressions: list[str] = []
    improvements: list[str] = []

    baseline_map = {r.name: r for r in baseline}

    for cur in current:
        base = baseline_map.get(cur.name)
        if base is None:
            continue

        delta = (cur.latency.mean_ms - base.latency.mean_ms) / max(base.latency.mean_ms, 0.001)
        if delta > threshold:
            regressions.append(
                f"{cur.name}: mean latency increased by {delta * 100:.1f}% "
                f"({base.latency.mean_ms:.1f}ms → {cur.latency.mean_ms:.1f}ms)"
            )
        elif delta < -threshold:
            improvements.append(
                f"{cur.name}: mean latency decreased by {abs(delta) * 100:.1f}% "
                f"({base.latency.mean_ms:.1f}ms → {cur.latency.mean_ms:.1f}ms)"
            )

    return {"regressions": regressions, "improvements": improvements}
