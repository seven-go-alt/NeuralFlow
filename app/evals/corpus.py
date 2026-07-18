from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.evals.datasets import EvalCase


@dataclass(slots=True, frozen=True)
class CorpusIssue:
    case_id: str
    doc_id: str
    kind: str  # "missing_doc" | "missing_keyword" | "negative_case_has_refs" | "unreadable_doc"
    detail: str


@dataclass(slots=True)
class CorpusReport:
    total_cases: int
    checked_docs: int = 0
    issues: list[CorpusIssue] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def collect_expected_doc_ids(cases: Sequence[EvalCase]) -> set[str]:
    ids: set[str] = set()
    for case in cases:
        ids.update(case.expected_doc_ids)
    return ids


def validate_corpus(cases: Sequence[EvalCase], corpus_dir: Path) -> CorpusReport:
    """Check that every positive case has an aligned corpus document.

    Rules:
      * each expected_doc_id must exist as <corpus_dir>/<doc_id>.md
        and be readable UTF-8 text
      * each expected keyword must appear (case-insensitively) in the
        combined text of the case's documents
      * negative cases must not reference docs or keywords
    Orphan corpus files (not referenced by any case) are reported as
    warnings, not issues.
    """
    report = CorpusReport(total_cases=len(cases))
    cache: dict[str, str | None] = {}
    unreadable: dict[str, str] = {}
    referenced: set[str] = set()

    def _load(doc_id: str) -> str | None:
        if doc_id not in cache:
            path = corpus_dir / f"{doc_id}.md"
            if path.is_file():
                try:
                    cache[doc_id] = path.read_text(encoding="utf-8").lower()
                except (UnicodeDecodeError, OSError) as exc:
                    unreadable[doc_id] = str(exc)
                    cache[doc_id] = None
            else:
                cache[doc_id] = None
        return cache[doc_id]

    for case in cases:
        referenced.update(case.expected_doc_ids)
        if not case.should_answer:
            if case.expected_doc_ids or case.expected_keywords:
                report.issues.append(
                    CorpusIssue(
                        case_id=case.id,
                        doc_id=",".join(case.expected_doc_ids) or "-",
                        kind="negative_case_has_refs",
                        detail=(
                            "negative case must have empty expected_doc_ids/"
                            f"expected_keywords (doc_ids={list(case.expected_doc_ids)}, "
                            f"keywords={list(case.expected_keywords)})"
                        ),
                    )
                )
            continue
        loaded: list[str] = []
        for doc_id in case.expected_doc_ids:
            text = _load(doc_id)
            if text is None:
                if doc_id in unreadable:
                    report.issues.append(
                        CorpusIssue(
                            case_id=case.id,
                            doc_id=doc_id,
                            kind="unreadable_doc",
                            detail=f"{doc_id}.md is not readable: {unreadable[doc_id]}",
                        )
                    )
                else:
                    report.issues.append(
                        CorpusIssue(
                            case_id=case.id,
                            doc_id=doc_id,
                            kind="missing_doc",
                            detail=f"{doc_id}.md not found in {corpus_dir}",
                        )
                    )
            else:
                loaded.append(text)
        if not loaded:
            continue
        combined = "\n".join(loaded)
        for keyword in case.expected_keywords:
            if keyword.lower() not in combined:
                report.issues.append(
                    CorpusIssue(
                        case_id=case.id,
                        doc_id=",".join(case.expected_doc_ids),
                        kind="missing_keyword",
                        detail=f"keyword '{keyword}' not found",
                    )
                )
    report.checked_docs = sum(1 for text in cache.values() if text is not None)
    if corpus_dir.is_dir():
        report.orphan_files = sorted(
            path.name for path in corpus_dir.glob("*.md") if path.stem not in referenced
        )
    return report
