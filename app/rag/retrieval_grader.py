from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.llm import LLMClient
from app.rag.no_answer_policy import decide_no_answer
from app.retrieval.schemas import RetrievalResult


@dataclass(slots=True)
class GradeResult:
    score: float
    sufficient: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


_LLM_GRADE_PROMPT = """请评估以下检索结果与问题的相关性（1-5分）。只返回一个数字。

问题：{query}

检索结果：
{chunks}

请给出1-5分的相关性评分（只返回数字）："""


async def grade_retrieval(
    query: str,
    results: list[RetrievalResult],
    llm: LLMClient | None = None,
) -> GradeResult:
    """Grade retrieval quality using no_answer_policy and optionally LLM."""
    no_answer = decide_no_answer(query, results)
    llm_score: float | None = None

    if llm is not None and results:
        llm_score = await _llm_grade_chunks(query, results[:3], llm)

    return _combine_grade(no_answer, llm_score)


async def _llm_grade_chunks(
    query: str,
    results: list[RetrievalResult],
    llm: LLMClient,
) -> float:
    """Ask LLM to rate relevance of top chunks."""
    chunks_text = "\n".join(f"{i + 1}. {r.content[:200]}" for i, r in enumerate(results))
    raw = await llm.generate(_LLM_GRADE_PROMPT.format(query=query, chunks=chunks_text))
    match = re.search(r"[1-5]", raw.strip())
    if match:
        return int(match.group()) / 5.0
    return 0.5


def _combine_grade(
    no_answer_decision: Any,
    llm_score: float | None,
) -> GradeResult:
    """Combines no_answer_policy and LLM scores into GradeResult."""
    confidence = no_answer_decision.confidence if hasattr(no_answer_decision, "confidence") else 0.0
    details: dict[str, Any] = {
        "no_answer_should_answer": no_answer_decision.should_answer,
        "no_answer_reason": no_answer_decision.reason,
        "llm_score": llm_score,
    }

    combined = (confidence + llm_score) / 2.0 if llm_score is not None else confidence

    if not no_answer_decision.should_answer:
        return GradeResult(
            score=combined,
            sufficient=False,
            reason=no_answer_decision.reason,
            details=details,
        )

    if llm_score is not None and llm_score < 0.4:
        return GradeResult(
            score=combined,
            sufficient=False,
            reason=f"LLM relevance score too low: {llm_score:.2f}",
            details=details,
        )

    return GradeResult(
        score=combined,
        sufficient=True,
        reason=f"Retrieval quality sufficient (score={combined:.2f})",
        details=details,
    )
