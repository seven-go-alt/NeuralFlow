from __future__ import annotations

from benchmarks.utils.metrics import recall_at_k


def test_recall_at_k_empty_no_results() -> None:
    assert recall_at_k([], []) == 0.0


def test_recall_at_k_empty_no_relevant() -> None:
    assert recall_at_k([], ["d1"]) == 0.0
