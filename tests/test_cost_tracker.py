from __future__ import annotations

import pytest

from app.observability.cost_tracker import CostTracker, ModelCostConfig, UsageRecord


class TestModelCostConfig:
    def test_estimate_input_cost(self) -> None:
        cfg = ModelCostConfig("test", input_cost_per_1k=0.01, output_cost_per_1k=0.02)
        assert cfg.estimate_input_cost(1000) == 0.01

    def test_estimate_output_cost(self) -> None:
        cfg = ModelCostConfig("test", input_cost_per_1k=0.01, output_cost_per_1k=0.02)
        assert cfg.estimate_output_cost(500) == 0.01

    def test_zero_tokens(self) -> None:
        cfg = ModelCostConfig("test", input_cost_per_1k=0.01, output_cost_per_1k=0.02)
        assert cfg.estimate_input_cost(0) == 0.0
        assert cfg.estimate_output_cost(0) == 0.0


class TestCostTracker:
    def test_record_and_summary(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        s = tracker.summary()
        assert s["total_input_tokens"] == 1000
        assert s["total_output_tokens"] == 500
        assert s["total_estimated_cost"] > 0
        assert s["calls"] == 1

    def test_unknown_model_cost_is_zero(self) -> None:
        tracker = CostTracker()
        tracker.record("unknown-model", input_tokens=1000)
        s = tracker.summary()
        assert s["total_estimated_cost"] == 0.0

    def test_multiple_records(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=1000, output_tokens=0)
        tracker.record("gpt-4o-mini", input_tokens=2000, output_tokens=0)
        s = tracker.summary()
        assert s["total_input_tokens"] == 3000
        assert s["calls"] == 2

    def test_by_model_breakdown(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1000)
        tracker.record("gpt-4o-mini", input_tokens=2000)
        s = tracker.summary()
        assert set(s["by_model"].keys()) == {"gpt-4o", "gpt-4o-mini"}

    def test_clear(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=1000)
        assert len(tracker.records) == 1
        tracker.clear()
        assert len(tracker.records) == 0

    def test_record_metadata(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, session_id="s1")
        assert tracker.records[0].metadata.get("session_id") == "s1"

    def test_custom_cost_config(self) -> None:
        custom = {
            "my-model": ModelCostConfig("my-model", input_cost_per_1k=0.1, output_cost_per_1k=0.2)
        }
        tracker = CostTracker(model_costs=custom)
        tracker.record("my-model", input_tokens=1000, output_tokens=1000)
        assert tracker.total_estimated_cost == pytest.approx(0.3)

    def test_records_property(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100)
        assert len(tracker.records) == 1
        assert isinstance(tracker.records[0], UsageRecord)

    def test_total_properties(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", input_tokens=100, output_tokens=50)
        assert tracker.total_input_tokens == 100
        assert tracker.total_output_tokens == 50
