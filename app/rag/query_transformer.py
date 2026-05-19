from __future__ import annotations

from dataclasses import dataclass, field

from app.core.llm import LLMClient
from app.retrieval.schemas import RetrievalResult


@dataclass(slots=True)
class TransformResult:
    query: str
    strategy: str  # rewrite | multi_query | hyde
    variants: list[str] = field(default_factory=list)
    original_query: str = ""


_REWRITE_PROMPT = "请将以下用户问题改写成更适合检索的形式，保持原意但更清晰具体：{query}"
_MULTI_QUERY_PROMPT = (
    "请从不同角度将以下问题改写成{n}个检索查询，每行一个（只输出查询文本）：\n{query}"
)
_HYDE_PROMPT = "请针对以下问题写一段假设性的回答（假设你已掌握相关知识）：\n{query}"


async def rewrite_query(query: str, llm: LLMClient) -> str:
    """Rewrite a vague or short query into a clearer retrieval query."""
    rewritten = await llm.generate(_REWRITE_PROMPT.format(query=query))
    result = rewritten.strip().strip("\"'")
    return result if result else query


async def expand_multi_query(query: str, llm: LLMClient, n: int = 3) -> TransformResult:
    """Generate multiple query variants for broader retrieval."""
    raw = await llm.generate(_MULTI_QUERY_PROMPT.format(query=query, n=n))
    variants = _parse_variants(raw)
    return TransformResult(
        query=query,
        strategy="multi_query",
        variants=variants if variants else [query],
        original_query=query,
    )


async def hyde_transform(query: str, llm: LLMClient) -> str:
    """Generate a hypothetical answer and use it as the retrieval query."""
    hypothesis = await llm.generate(_HYDE_PROMPT.format(query=query))
    result = hypothesis.strip()
    return result if len(result) > len(query) else query


def _parse_variants(text: str) -> list[str]:
    """Parse line-separated variants from LLM output."""
    lines = [line.strip().strip("-*1234567890. ") for line in text.split("\n") if line.strip()]
    return [line for line in lines if len(line) > 3][:5]


def merge_deduplicated(
    results: list[list[RetrievalResult]],
) -> list[RetrievalResult]:
    """Merge multiple result lists with deduplication by doc_id:chunk_id."""
    seen: set[str] = set()
    merged: list[RetrievalResult] = []
    for batch in results:
        for r in batch:
            key = f"{r.document_id}:{r.chunk_id}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
    return merged
