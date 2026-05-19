from __future__ import annotations

from dataclasses import dataclass, field

from app.evals.metrics import EvalMetrics, aggregate_metrics
from app.evals.runner import AnswerEvalFn, AnswerFn, RetrieveFn, run_eval


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    baseline: EvalMetrics
    experiment: EvalMetrics
    deltas: dict[str, float] = field(default_factory=dict)
    winners: dict[str, str] = field(default_factory=dict)
    total_cases: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_cases", self.baseline.total_cases)
        computed_deltas: dict[str, float] = {}
        computed_winners: dict[str, str] = {}

        metric_names = [
            "retrieval_hit_rate",
            "citation_accuracy",
            "keyword_coverage",
            "no_answer_accuracy",
            "average_latency_ms",
            "mean_reciprocal_rank",
            "average_precision_at_k",
            "average_recall_at_k",
        ]
        if self.baseline.answer_count > 0 or self.experiment.answer_count > 0:
            metric_names.extend(
                [
                    "average_answer_relevance",
                    "average_answer_faithfulness",
                    "average_answer_completeness",
                ]
            )

        for name in metric_names:
            base_val = getattr(self.baseline, name, 0.0)
            exp_val = getattr(self.experiment, name, 0.0)
            delta = exp_val - base_val
            computed_deltas[name] = round(delta, 6)

            if name == "average_latency_ms":
                computed_winners[name] = "experiment" if delta < 0 else "baseline" if delta > 0 else "tie"
            else:
                computed_winners[name] = "experiment" if delta > 0 else "baseline" if delta < 0 else "tie"

        object.__setattr__(self, "deltas", computed_deltas)
        object.__setattr__(self, "winners", computed_winners)


async def compare_runs(
    cases_path: str,
    baseline_retrieve_fn: RetrieveFn,
    baseline_answer_fn: AnswerFn,
    experiment_retrieve_fn: RetrieveFn,
    experiment_answer_fn: AnswerFn,
    top_k: int = 5,
    baseline_answer_eval_fn: AnswerEvalFn | None = None,
    experiment_answer_eval_fn: AnswerEvalFn | None = None,
) -> ComparisonResult:
    """Run two eval configurations side-by-side and produce a ComparisonResult."""
    baseline_results = await run_eval(
        cases_path,
        baseline_retrieve_fn,
        baseline_answer_fn,
        top_k=top_k,
        answer_eval_fn=baseline_answer_eval_fn,
    )
    baseline_metrics = aggregate_metrics(baseline_results)

    experiment_results = await run_eval(
        cases_path,
        experiment_retrieve_fn,
        experiment_answer_fn,
        top_k=top_k,
        answer_eval_fn=experiment_answer_eval_fn,
    )
    experiment_metrics = aggregate_metrics(experiment_results)

    return ComparisonResult(
        baseline=baseline_metrics,
        experiment=experiment_metrics,
    )


def format_comparison_table(result: ComparisonResult) -> str:
    """Format a comparison result as a markdown table."""
    lines: list[str] = []
    lines.append("# A/B Comparison Results")
    lines.append("")
    lines.append(f"- **Total cases**: {result.total_cases}")
    lines.append("")

    lines.append("| Metric | Baseline | Experiment | Delta | Winner |")
    lines.append("|--------|----------|------------|-------|--------|")

    metric_display: list[tuple[str, str]] = [
        ("retrieval_hit_rate", "Retrieval Hit Rate"),
        ("citation_accuracy", "Citation Accuracy"),
        ("keyword_coverage", "Keyword Coverage"),
        ("no_answer_accuracy", "No-Answer Accuracy"),
        ("mean_reciprocal_rank", "Mean Reciprocal Rank"),
        ("average_precision_at_k", "Avg Precision@k"),
        ("average_recall_at_k", "Avg Recall@k"),
        ("average_latency_ms", "Avg Latency (ms)"),
    ]

    if result.baseline.answer_count > 0 or result.experiment.answer_count > 0:
        metric_display.extend(
            [
                ("average_answer_relevance", "Avg Answer Relevance"),
                ("average_answer_faithfulness", "Avg Answer Faithfulness"),
                ("average_answer_completeness", "Avg Answer Completeness"),
            ]
        )

    for metric_key, display_name in metric_display:
        base_val = getattr(result.baseline, metric_key)
        exp_val = getattr(result.experiment, metric_key)
        delta = result.deltas.get(metric_key, 0.0)
        winner = result.winners.get(metric_key, "tie")

        if metric_key == "average_latency_ms":
            base_str = f"{base_val:.1f}"
            exp_str = f"{exp_val:.1f}"
            delta_str = f"{delta:+.1f}"
        elif metric_key == "mean_reciprocal_rank":
            base_str = f"{base_val:.4f}"
            exp_str = f"{exp_val:.4f}"
            delta_str = f"{delta:+.4f}"
        else:
            base_str = f"{base_val:.1%}"
            exp_str = f"{exp_val:.1%}"
            delta_str = f"{delta:+.1%}"

        winner_symbol = {"baseline": "baseline", "experiment": "experiment", "tie": "tie"}.get(winner, "—")
        lines.append(f"| {display_name} | {base_str} | {exp_str} | {delta_str} | {winner_symbol} |")

    lines.append("")
    return "\n".join(lines)
