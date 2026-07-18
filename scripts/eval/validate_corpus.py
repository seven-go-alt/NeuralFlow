"""Validate that the synthetic eval corpus aligns with the eval datasets.

For every positive eval case, checks that data/eval/corpus/<doc_id>.md
exists and contains all expected keywords (case-insensitively).

Usage (from repo root):
    uv run python -m scripts.eval.validate_corpus
    uv run python -m scripts.eval.validate_corpus --list-missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.evals.corpus import validate_corpus
from app.evals.datasets import EvalCase, load_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate eval corpus/dataset alignment")
    parser.add_argument("--datasets-dir", default="data/eval/datasets",
                        help="Directory containing .jsonl dataset files (default: data/eval/datasets)")
    parser.add_argument("--corpus-dir", default="data/eval/corpus",
                        help="Directory containing corpus .md files (default: data/eval/corpus)")
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="only print doc_ids that still have no corpus file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases: list[EvalCase] = []
    for path in sorted(Path(args.datasets_dir).glob("*.jsonl")):
        cases.extend(load_cases(path))
    if not cases:
        print(f"error: no .jsonl files found in {args.datasets_dir}", file=sys.stderr)
        return 1
    report = validate_corpus(cases, Path(args.corpus_dir))
    if args.list_missing:
        missing = sorted({issue.doc_id for issue in report.issues if issue.kind == "missing_doc"})
        if missing:
            print("\n".join(missing))
        return 0
    for issue in report.issues:
        print(f"[{issue.kind}] case={issue.case_id} doc={issue.doc_id}: {issue.detail}")
    for orphan in report.orphan_files:
        print(f"[orphan] {orphan}")
    print(
        f"cases={report.total_cases} docs_checked={report.checked_docs} "
        f"issues={len(report.issues)}"
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
