from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.dataset_resolver import EvalDatasetError, resolve_eval_dataset


def test_resolves_dataset_id_inside_root(tmp_path: Path) -> None:
    dataset = tmp_path / "smoke.jsonl"
    dataset.write_text('{"id":"1","question":"q"}\n', encoding="utf-8")

    assert resolve_eval_dataset("smoke", root=tmp_path) == dataset.resolve()
    assert resolve_eval_dataset("smoke.jsonl", root=tmp_path) == dataset.resolve()


@pytest.mark.parametrize("dataset_id", ["../smoke", "/etc/passwd", "smoke.txt", ""])
def test_rejects_unsafe_dataset_id(tmp_path: Path, dataset_id: str) -> None:
    (tmp_path / "smoke.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(EvalDatasetError):
        resolve_eval_dataset(dataset_id, root=tmp_path)


def test_rejects_dataset_over_size_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from app.config import get_settings

    dataset = tmp_path / "large.jsonl"
    dataset.write_bytes(b"x" * 16)
    settings = get_settings()
    monkeypatch.setattr(settings, "eval_max_dataset_mb", 0)

    with pytest.raises(EvalDatasetError, match="size limit"):
        resolve_eval_dataset("large", root=tmp_path)
