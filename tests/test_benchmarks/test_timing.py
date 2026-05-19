from __future__ import annotations

from benchmarks.utils.timing import compute_stats


def test_compute_stats_empty() -> None:
    stats = compute_stats([])
    assert stats.samples == 0


def test_compute_stats_single_sample() -> None:
    stats = compute_stats([10.0])
    assert stats.samples == 1
    assert stats.mean_ms == 10.0
    assert stats.min_ms == 10.0
    assert stats.max_ms == 10.0


def test_compute_stats_multiple() -> None:
    stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.samples == 5
    assert stats.mean_ms == 3.0
    assert stats.p50_ms == 3.0
    assert stats.p90_ms == 5.0
