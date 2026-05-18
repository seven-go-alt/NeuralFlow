from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelCostConfig:
    model_id: str
    input_cost_per_1k: float
    output_cost_per_1k: float

    def estimate_input_cost(self, tokens: int) -> float:
        return (tokens / 1000) * self.input_cost_per_1k

    def estimate_output_cost(self, tokens: int) -> float:
        return (tokens / 1000) * self.output_cost_per_1k


DEFAULT_MODEL_COSTS: dict[str, ModelCostConfig] = {
    "gpt-4o": ModelCostConfig("gpt-4o", input_cost_per_1k=0.0025, output_cost_per_1k=0.01),
    "gpt-4o-mini": ModelCostConfig("gpt-4o-mini", input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    "gpt-4": ModelCostConfig("gpt-4", input_cost_per_1k=0.03, output_cost_per_1k=0.06),
    "gpt-3.5-turbo": ModelCostConfig("gpt-3.5-turbo", input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
    "claude-sonnet-4-20250514": ModelCostConfig(
        "claude-sonnet-4-20250514", input_cost_per_1k=0.003, output_cost_per_1k=0.015
    ),
    "claude-haiku-3-5": ModelCostConfig(
        "claude-haiku-3-5", input_cost_per_1k=0.0008, output_cost_per_1k=0.004
    ),
}


@dataclass(slots=True)
class UsageRecord:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Track LLM token usage and estimate costs across operations."""

    def __init__(self, model_costs: dict[str, ModelCostConfig] | None = None) -> None:
        self._model_costs = model_costs or dict(DEFAULT_MODEL_COSTS)
        self._records: list[UsageRecord] = []

    def record(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        **metadata: Any,
    ) -> UsageRecord:
        cost_config = self._model_costs.get(model)
        if cost_config:
            cost = cost_config.estimate_input_cost(input_tokens) + cost_config.estimate_output_cost(
                output_tokens
            )
        else:
            cost = 0.0
        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=round(cost, 6),
            metadata=metadata,
        )
        self._records.append(record)
        return record

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self._records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self._records)

    @property
    def total_estimated_cost(self) -> float:
        return round(sum(r.estimated_cost for r in self._records), 6)

    @property
    def records(self) -> list[UsageRecord]:
        return list(self._records)

    def summary(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_estimated_cost": self.total_estimated_cost,
            "calls": len(self._records),
            "by_model": {
                model: {
                    "input_tokens": sum(r.input_tokens for r in self._records if r.model == model),
                    "output_tokens": sum(r.output_tokens for r in self._records if r.model == model),
                    "estimated_cost": round(
                        sum(r.estimated_cost for r in self._records if r.model == model), 6
                    ),
                    "calls": sum(1 for r in self._records if r.model == model),
                }
                for model in {r.model for r in self._records}
            },
        }

    def clear(self) -> None:
        self._records.clear()
