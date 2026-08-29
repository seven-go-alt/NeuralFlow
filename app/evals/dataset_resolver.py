from __future__ import annotations

from pathlib import Path

from app.config import get_settings


class EvalDatasetError(ValueError):
    """Raised when a requested evaluation dataset is invalid or unavailable."""


def resolve_eval_dataset(dataset_id: str, *, root: str | Path | None = None) -> Path:
    """Resolve a dataset id below the configured evaluation dataset directory.

    API callers may reference a dataset by filename or stem, but may not supply an
    arbitrary filesystem path. CLI callers that already have a local path continue
    to use ``run_eval`` directly.
    """
    value = dataset_id.strip()
    if not value or value in {".", ".."}:
        raise EvalDatasetError("dataset_id must not be empty")

    dataset_root = Path(root or get_settings().eval_dataset_dir).expanduser().resolve()
    candidate_name = value if value.endswith(".jsonl") else f"{value}.jsonl"
    candidate = (dataset_root / candidate_name).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise EvalDatasetError(
            "dataset_id must refer to a file inside the eval dataset directory"
        ) from exc

    if candidate.suffix.lower() != ".jsonl":
        raise EvalDatasetError("evaluation datasets must use the .jsonl extension")
    if not candidate.is_file():
        raise EvalDatasetError(f"evaluation dataset not found: {value}")

    settings = get_settings()
    max_bytes = settings.eval_max_dataset_mb * 1024 * 1024
    if candidate.stat().st_size > max_bytes:
        raise EvalDatasetError("evaluation dataset exceeds the configured size limit")
    return candidate
