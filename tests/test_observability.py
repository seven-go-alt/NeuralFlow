from __future__ import annotations

import json
import logging

from prometheus_client import CollectorRegistry

from app.utils.observability import JsonLogFormatter, create_observability, set_log_context


def test_create_observability_reuses_existing_collectors() -> None:
    registry = CollectorRegistry()

    first = create_observability(registry=registry)
    second = create_observability(registry=registry)

    assert first.request_duration is second.request_duration
    assert first.llm_token_usage is second.llm_token_usage
    assert first.error_total is second.error_total


def test_json_log_formatter_emits_required_context_fields() -> None:
    formatter = JsonLogFormatter()
    set_log_context(session_id="session-1", trace_id="trace-1", intent="general")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello observability",
        args=(),
        exc_info=None,
    )
    record.duration_ms = 12.5

    rendered = formatter.format(record)
    payload = json.loads(rendered)

    assert payload["message"] == "hello observability"
    assert payload["session_id"] == "session-1"
    assert payload["trace_id"] == "trace-1"
    assert payload["intent"] == "general"
    assert payload["duration_ms"] == 12.5
