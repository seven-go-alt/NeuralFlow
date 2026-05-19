from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.llm import LLMClient

_EVAL_PROMPT = """你是一个 RAG 答案质量评估员。请根据问题和检索上下文评估给出的答案。

问题：{query}

检索上下文：
{context}

答案：{answer}

请从以下三个维度评分（每个维度 0.0-1.0）：
1. relevance（相关性）：答案是否直接回答了问题？
2. faithfulness（忠实性）：答案是否基于检索上下文、没有幻觉？
3. completeness（完整性）：答案是否覆盖了问题的所有方面？

请返回 JSON 格式：
{{"relevance": 0.0-1.0, "faithfulness": 0.0-1.0, "completeness": 0.0-1.0, "reason": "简短评价"}}"""


@dataclass(slots=True)
class AnswerEvalResult:
    relevance: float
    faithfulness: float
    completeness: float
    overall: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def zero(cls, reason: str = "no answer") -> AnswerEvalResult:
        return cls(relevance=0.0, faithfulness=0.0, completeness=0.0, overall=0.0, reason=reason)

    @classmethod
    def from_llm_response(cls, raw: str) -> AnswerEvalResult:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return cls.zero(reason="parse error: no JSON found")
            data = json.loads(match.group())
            relevance = max(0.0, min(1.0, float(data.get("relevance", 0.0))))
            faithfulness = max(0.0, min(1.0, float(data.get("faithfulness", 0.0))))
            completeness = max(0.0, min(1.0, float(data.get("completeness", 0.0))))
            reason = str(data.get("reason", ""))
            overall = (relevance + faithfulness + completeness) / 3.0
            return cls(
                relevance=relevance,
                faithfulness=faithfulness,
                completeness=completeness,
                overall=overall,
                reason=reason,
                details=data,
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return cls.zero(reason="parse error: invalid JSON")


async def evaluate_answer(
    query: str,
    answer: str,
    context_chunks: list[str],
    llm: LLMClient,
) -> AnswerEvalResult:
    if not answer:
        return AnswerEvalResult.zero(reason="empty answer")
    context = "\n".join(f"- {c[:500]}" for c in context_chunks)
    prompt = _EVAL_PROMPT.format(query=query, answer=answer, context=context)
    try:
        raw = await llm.generate(prompt)
        return AnswerEvalResult.from_llm_response(raw)
    except Exception as exc:
        return AnswerEvalResult.zero(reason=f"llm error: {exc}")
