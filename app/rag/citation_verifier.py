from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CitationVerificationResult:
    total_citations_in_answer: int
    valid_citations: int
    invalid_citations: int
    missing_indices: list[int]
    used_indices: list[int]
    details: list[dict[str, Any]] = field(default_factory=list)


def extract_citations(text: str | None) -> list[int]:
    """Extract citation indices like [1], [2,3], [1-3] from text."""
    if not text:
        return []
    indices: set[int] = set()
    for match in re.finditer(r"\[([^\]]+)\]", text):
        content = match.group(1).strip()
        parts = content.split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                try:
                    start, end = part.split("-", 1)
                    indices.update(range(int(start.strip()), int(end.strip()) + 1))
                except ValueError:
                    pass
            else:
                with contextlib.suppress(ValueError):
                    indices.add(int(part))
    return sorted(indices)


def verify_citations(
    answer: str | None,
    citations: list[dict[str, Any]],
) -> CitationVerificationResult:
    """Verify that citations in the answer match the provided citation metadata."""
    used_indices = extract_citations(answer)
    valid_indices = {c["index"] for c in citations}
    invalid = [i for i in used_indices if i not in valid_indices]
    missing = [c["index"] for c in citations if c["index"] not in used_indices]
    return CitationVerificationResult(
        total_citations_in_answer=len(used_indices),
        valid_citations=len(used_indices) - len(invalid),
        invalid_citations=len(invalid),
        missing_indices=missing,
        used_indices=used_indices,
        details=[
            {
                "index": i,
                "valid": i in valid_indices,
                "used": i in used_indices,
                "label": next(
                    (c.get("label", "") for c in citations if c["index"] == i),
                    None,
                ),
            }
            for i in sorted(set(used_indices) | valid_indices)
        ],
    )
