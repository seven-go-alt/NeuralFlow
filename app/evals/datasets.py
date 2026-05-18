from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class EvalCase:
    id: str
    question: str
    expected_keywords: tuple[str, ...]
    expected_doc_ids: tuple[str, ...]
    should_answer: bool


def load_cases(path: str | Path) -> list[EvalCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval cases file not found: {path}")
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cases.append(
                EvalCase(
                    id=record["id"],
                    question=record["question"],
                    expected_keywords=tuple(record.get("expected_keywords", [])),
                    expected_doc_ids=tuple(record.get("expected_doc_ids", [])),
                    should_answer=record.get("should_answer", True),
                )
            )
    return cases
