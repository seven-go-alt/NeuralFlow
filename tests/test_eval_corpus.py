from __future__ import annotations

import json
from pathlib import Path

from app.evals.corpus import collect_expected_doc_ids, validate_corpus
from app.evals.datasets import load_cases


def _write_dataset(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _case(
    case_id: str,
    doc_id: str | None,
    keywords: list[str],
    should_answer: bool = True,
) -> dict:
    return {
        "id": case_id,
        "question": f"question for {case_id}?",
        "expected_keywords": keywords,
        "expected_doc_ids": [doc_id] if doc_id else [],
        "should_answer": should_answer,
    }


def test_validate_corpus_passes_when_docs_and_keywords_align(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["annual leave"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_hr_leave.md").write_text(
        "# Leave Policy\nAnnual Leave is 25 days.", encoding="utf-8"
    )

    report = validate_corpus(load_cases(dataset), corpus)

    assert report.ok
    assert report.issues == []
    assert report.orphan_files == []
    assert report.checked_docs == 1


def test_validate_corpus_flags_missing_doc(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["annual leave"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    report = validate_corpus(load_cases(dataset), corpus)

    assert not report.ok
    assert [issue.kind for issue in report.issues] == ["missing_doc"]
    assert report.issues[0].doc_id == "doc_hr_leave"


def test_validate_corpus_flags_missing_keyword(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["annual leave", "carry-over"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_hr_leave.md").write_text("Annual Leave is 25 days.", encoding="utf-8")

    report = validate_corpus(load_cases(dataset), corpus)

    assert [issue.kind for issue in report.issues] == ["missing_keyword"]
    assert "carry-over" in report.issues[0].detail


def test_validate_corpus_keyword_match_is_case_insensitive(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["ANNUAL LEAVE"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_hr_leave.md").write_text("annual leave: 25 days", encoding="utf-8")

    report = validate_corpus(load_cases(dataset), corpus)

    assert report.ok


def test_validate_corpus_flags_negative_case_with_refs(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(
        dataset, [_case("c1", "doc_hr_leave", ["annual leave"], should_answer=False)]
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    report = validate_corpus(load_cases(dataset), corpus)

    assert [issue.kind for issue in report.issues] == ["negative_case_has_refs"]


def test_validate_corpus_lists_orphan_files(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_hr_leave", ["annual leave"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_hr_leave.md").write_text("annual leave", encoding="utf-8")
    (corpus / "doc_unused.md").write_text("nobody references me", encoding="utf-8")

    report = validate_corpus(load_cases(dataset), corpus)

    assert report.ok  # orphan 只是警告,不算 issue
    assert report.orphan_files == ["doc_unused.md"]


def test_validate_corpus_multi_doc_keyword_in_one(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(
        dataset,
        [
            {
                "id": "c1",
                "question": "multi doc test",
                "expected_keywords": ["unique term"],
                "expected_doc_ids": ["doc_a", "doc_b"],
                "should_answer": True,
            }
        ],
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_a.md").write_text("some text", encoding="utf-8")
    (corpus / "doc_b.md").write_text("unique term appears here", encoding="utf-8")

    report = validate_corpus(load_cases(dataset), corpus)
    assert report.ok


def test_validate_corpus_multi_doc_keyword_missing_from_all(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(
        dataset,
        [
            {
                "id": "c2",
                "question": "multi doc test",
                "expected_keywords": ["ghost"],
                "expected_doc_ids": ["doc_a", "doc_b"],
                "should_answer": True,
            }
        ],
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_a.md").write_text("some text", encoding="utf-8")
    (corpus / "doc_b.md").write_text("more text", encoding="utf-8")

    report = validate_corpus(load_cases(dataset), corpus)
    assert [issue.kind for issue in report.issues] == ["missing_keyword"]
    assert report.issues[0].doc_id == "doc_a,doc_b"


def test_validate_corpus_unreadable_doc(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case("c1", "doc_bad", ["policy"])])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc_bad.md").write_bytes(b"\xff\xfe\x00policy")

    report = validate_corpus(load_cases(dataset), corpus)

    assert [issue.kind for issue in report.issues] == ["unreadable_doc"]


def test_collect_expected_doc_ids_dedupes(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(
        dataset,
        [
            _case("c1", "doc_hr_leave", ["annual leave"]),
            _case("c2", "doc_hr_leave", ["entitlement"]),
            _case("c3", None, [], should_answer=False),
        ],
    )

    ids = collect_expected_doc_ids(load_cases(dataset))

    assert ids == {"doc_hr_leave"}
