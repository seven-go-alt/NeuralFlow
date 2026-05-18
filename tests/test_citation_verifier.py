from __future__ import annotations

import pytest

from app.rag.citation_verifier import CitationVerificationResult, extract_citations, verify_citations


class TestExtractCitations:
    def test_single(self) -> None:
        assert extract_citations("see [1] for details") == [1]

    def test_multiple(self) -> None:
        assert extract_citations("see [1] and [2] and [3]") == [1, 2, 3]

    def test_comma_list(self) -> None:
        assert extract_citations("see [1, 3]") == [1, 3]

    def test_range(self) -> None:
        assert extract_citations("see [1-3]") == [1, 2, 3]

    def test_none_text(self) -> None:
        assert extract_citations(None) == []

    def test_empty_text(self) -> None:
        assert extract_citations("") == []

    def test_no_citations(self) -> None:
        assert extract_citations("no references here") == []

    def test_mixed(self) -> None:
        assert extract_citations("see [1, 3-5] and [7]") == [1, 3, 4, 5, 7]


class TestVerifyCitations:
    def test_all_valid(self) -> None:
        result = verify_citations("See [1] and [2]", [{"index": 1}, {"index": 2}])
        assert result.valid_citations == 2
        assert result.invalid_citations == 0
        assert result.missing_indices == []

    def test_invalid_citation(self) -> None:
        result = verify_citations("See [1] and [99]", [{"index": 1}, {"index": 2}])
        assert result.valid_citations == 1
        assert result.invalid_citations == 1
        assert result.missing_indices == [2]

    def test_unused_citation(self) -> None:
        result = verify_citations("See [1]", [{"index": 1}, {"index": 2}])
        assert result.missing_indices == [2]

    def test_no_answer(self) -> None:
        result = verify_citations(None, [{"index": 1}])
        assert result.total_citations_in_answer == 0
        assert result.valid_citations == 0

    def test_no_citations_provided(self) -> None:
        result = verify_citations("See [1]", [])
        assert result.valid_citations == 0
        assert result.invalid_citations == 1

    def test_details_contains_all_indices(self) -> None:
        result = verify_citations("See [1]", [{"index": 1, "label": "doc1"}, {"index": 2, "label": "doc2"}])
        indices_in_details = {d["index"] for d in result.details}
        assert indices_in_details == {1, 2}

    def test_type(self) -> None:
        result = verify_citations(None, [])
        assert isinstance(result, CitationVerificationResult)
