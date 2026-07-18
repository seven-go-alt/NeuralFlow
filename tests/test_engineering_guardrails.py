from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_includes_quality_tooling_configs() -> None:
    assert (PROJECT_ROOT / "ruff.toml").exists()
    assert (PROJECT_ROOT / "mypy.ini").exists()
    assert (PROJECT_ROOT / ".pre-commit-config.yaml").exists()
    assert (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").exists()


def test_dev_dependency_group_includes_quality_tools() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev_dependencies = pyproject["dependency-groups"]["dev"]

    assert any(dep.startswith("ruff>=") for dep in dev_dependencies)
    assert any(dep.startswith("mypy>=") for dep in dev_dependencies)
    assert any(dep.startswith("pre-commit>=") for dep in dev_dependencies)


def test_ci_workflow_runs_lint_typecheck_and_tests() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    setup_backend = (
        PROJECT_ROOT / ".github" / "actions" / "setup-backend" / "action.yml"
    ).read_text(encoding="utf-8")

    assert "on:" in workflow
    assert "push:" in workflow
    assert "pull_request:" in workflow
    # Dependency sync is handled by the composite setup-backend action
    assert "uv sync --group dev --extra ml --frozen" in setup_backend
    # Lint and typecheck run via pre-commit hooks (which wraps ruff + mypy)
    assert "pre-commit run --all-files" in workflow
    # Tests run in the test job
    assert "uv run pytest" in workflow


def test_pre_commit_hooks_match_project_quality_pipeline() -> None:
    pre_commit = (PROJECT_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "https://github.com/astral-sh/ruff-pre-commit" in pre_commit
    assert "https://github.com/pre-commit/mirrors-mypy" in pre_commit
    assert "id: ruff-check" in pre_commit
    assert "id: ruff-format" in pre_commit
    assert "id: mypy" in pre_commit
    assert "worker.py" in pre_commit
    assert "app" in pre_commit
    assert "tests" in pre_commit
