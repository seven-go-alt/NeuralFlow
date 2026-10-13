from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

_EVAL_RUN_COLUMNS: dict[str, str] = {
    "status": "VARCHAR(32) NOT NULL DEFAULT 'queued'",
    "progress": "INTEGER NOT NULL DEFAULT 0",
    "error_message": "VARCHAR(2000)",
    "celery_task_id": "VARCHAR(255)",
}


def apply_compatibility_migrations(engine: Engine) -> None:
    """Apply additive schema changes for installations without Alembic yet."""
    inspector = inspect(engine)
    if "eval_runs" not in inspector.get_table_names():
        return

    existing = {column["name"] for column in inspector.get_columns("eval_runs")}
    with engine.begin() as connection:
        for name, definition in _EVAL_RUN_COLUMNS.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE eval_runs ADD COLUMN {name} {definition}"))
