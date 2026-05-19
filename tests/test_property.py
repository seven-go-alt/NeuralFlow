from __future__ import annotations

from hypothesis import assume, given
from hypothesis import strategies as st

from app.evals.metrics import (
    EvalMetrics,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_retrieval_hit,
)


@given(
    st.lists(st.text(min_size=1, max_size=50)),
    st.lists(st.text(min_size=1, max_size=50)),
)
def test_retrieval_hit_is_boolean(retrieved_ids, expected_ids):
    """retrieval_hit should always be a boolean."""
    result = compute_retrieval_hit(tuple(retrieved_ids), tuple(expected_ids))
    assert isinstance(result, bool)


@given(
    st.lists(st.text(min_size=1, max_size=20)),
    st.lists(st.text(min_size=1, max_size=20)),
    st.integers(min_value=1, max_value=20),
)
def test_precision_at_k_is_between_0_and_1(retrieved, relevant, k):
    assume(len(retrieved) > 0)
    result = compute_precision_at_k(tuple(retrieved), tuple(relevant), k)
    assert 0.0 <= result <= 1.0


@given(
    st.lists(st.text(min_size=1, max_size=20)),
    st.lists(st.text(min_size=1, max_size=20)),
    st.integers(min_value=1, max_value=20),
)
def test_recall_at_k_is_between_0_and_1(retrieved, relevant, k):
    assume(len(retrieved) > 0)
    result = compute_recall_at_k(tuple(retrieved), tuple(relevant), k)
    assert 0.0 <= result <= 1.0


@given(st.integers(min_value=1, max_value=1000))
def test_metrics_total_cases_positive(total):
    m = EvalMetrics(
        total_cases=total,
        retrieval_hits=0,
        mrr_sum=0.0,
        precision_at_k_sum=0.0,
        recall_at_k_sum=0.0,
        total_latency_ms=0.0,
    )
    assert m.total_cases > 0
    assert m.retrieval_hit_rate >= 0.0


@given(
    st.lists(st.text(min_size=1, max_size=10)),
    st.lists(st.text(min_size=1, max_size=10)),
    st.integers(min_value=1, max_value=10),
)
def test_precision_with_empty_relevant(retrieved, relevant, k):
    """When both retrieved and relevant are empty, precision should be 1.0."""
    assume(len(retrieved) > 0)
    result = compute_precision_at_k(tuple(retrieved), tuple(relevant), k)
    if not relevant:
        assert result == 1.0


@given(
    st.lists(st.text(min_size=1, max_size=10)),
    st.lists(st.text(min_size=1, max_size=10)),
    st.integers(min_value=1, max_value=10),
)
def test_recall_with_empty_relevant(retrieved, relevant, k):
    """When both retrieved and relevant are empty, recall should be 1.0."""
    assume(len(retrieved) > 0)
    result = compute_recall_at_k(tuple(retrieved), tuple(relevant), k)
    if not relevant:
        assert result == 1.0


@given(
    st.lists(st.text(min_size=1, max_size=50)),
    st.lists(st.text(min_size=1, max_size=50)),
)
def test_retrieval_hit_empty_expected(retrieved_ids, expected_ids):
    """When expected is empty, retrieval_hit should be True."""
    result = compute_retrieval_hit(tuple(retrieved_ids), tuple(expected_ids))
    if not expected_ids:
        assert result is True
