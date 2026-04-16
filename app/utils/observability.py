from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, cast

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, REGISTRY, generate_latest

try:
    from prometheus_client import multiprocess
except ImportError:  # pragma: no cover
    multiprocess = None


_session_id_var: ContextVar[str] = ContextVar("session_id", default="anonymous")
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
_intent_var: ContextVar[str] = ContextVar("intent", default="unknown")
_registry_lock = Lock()
_logger_lock = Lock()
_cached_observability: dict[int, "Observability"] = {}
_configured_loggers: set[str] = set()


@dataclass(slots=True)
class Observability:
    registry: CollectorRegistry
    request_duration: Histogram
    llm_token_usage: Counter
    memory_cache_hit: Counter
    active_sessions: Gauge
    error_total: Counter

    def render_metrics(self) -> bytes:
        return generate_latest(self.registry)

    def metrics_response(self) -> Response:
        return Response(content=self.render_metrics(), media_type=CONTENT_TYPE_LATEST)

    def record_llm_token_usage(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        if input_tokens > 0:
            self.llm_token_usage.labels(model=model, type="input").inc(input_tokens)
        if output_tokens > 0:
            self.llm_token_usage.labels(model=model, type="output").inc(output_tokens)

    def record_memory_cache_hit(self, layer: str) -> None:
        self.memory_cache_hit.labels(layer=layer).inc()


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", _session_id_var.get()),
            "trace_id": getattr(record, "trace_id", _trace_id_var.get()),
            "intent": getattr(record, "intent", _intent_var.get()),
            "duration_ms": getattr(record, "duration_ms", None),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _RESERVED_LOG_KEYS or key in payload:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


_RESERVED_LOG_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
    "session_id",
    "trace_id",
    "intent",
    "duration_ms",
}


def configure_structured_logging(logger_name: str = "neuralflow", audit_log_path: str | None = None) -> logging.Logger:
    with _logger_lock:
        logger = logging.getLogger(logger_name)
        if logger_name not in _configured_loggers:
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            logger.propagate = False
            formatter = JsonLogFormatter()
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(formatter)
            logger.addHandler(stdout_handler)
            if audit_log_path:
                Path(audit_log_path).parent.mkdir(parents=True, exist_ok=True)
                audit_handler = logging.FileHandler(audit_log_path)
                audit_handler.setFormatter(formatter)
                audit_handler.setLevel(logging.INFO)
                logger.addHandler(audit_handler)
            _configured_loggers.add(logger_name)
        return logger


def create_observability(registry: CollectorRegistry | None = None) -> Observability:
    target_registry = registry or _build_default_registry()
    cache_key = id(target_registry)
    with _registry_lock:
        cached = _cached_observability.get(cache_key)
        if cached is not None:
            return cached
        observability = Observability(
            registry=target_registry,
            request_duration=_get_or_create_histogram(
                target_registry,
                "neuralflow_request_duration_seconds",
                "Request duration in seconds.",
                ("endpoint", "intent"),
            ),
            llm_token_usage=_get_or_create_counter(
                target_registry,
                "neuralflow_llm_token_usage_total",
                "LLM token usage total.",
                ("model", "type"),
            ),
            memory_cache_hit=_get_or_create_counter(
                target_registry,
                "neuralflow_memory_cache_hit_total",
                "Memory cache hit total.",
                ("layer",),
            ),
            active_sessions=_get_or_create_gauge(
                target_registry,
                "neuralflow_active_sessions",
                "Active in-flight sessions.",
            ),
            error_total=_get_or_create_counter(
                target_registry,
                "neuralflow_errors_total",
                "Unhandled request errors total.",
                ("endpoint", "intent"),
            ),
        )
        _cached_observability[cache_key] = observability
        return observability


def set_log_context(session_id: str | None = None, trace_id: str | None = None, intent: str | None = None) -> None:
    if session_id is not None:
        _session_id_var.set(session_id)
    if trace_id is not None:
        _trace_id_var.set(trace_id)
    if intent is not None:
        _intent_var.set(intent)


def get_log_context() -> dict[str, str]:
    return {
        "session_id": _session_id_var.get(),
        "trace_id": _trace_id_var.get(),
        "intent": _intent_var.get(),
    }


def _build_default_registry() -> CollectorRegistry:
    multiproc_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_dir and multiprocess is not None:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return cast(CollectorRegistry, REGISTRY)


def _get_or_create_counter(
    registry: CollectorRegistry,
    name: str,
    documentation: str,
    label_names: tuple[str, ...],
) -> Counter:
    collector = _lookup_collector(registry, name)
    if collector is not None:
        return cast(Counter, collector)
    return Counter(name, documentation, labelnames=label_names, registry=registry)


def _get_or_create_histogram(
    registry: CollectorRegistry,
    name: str,
    documentation: str,
    label_names: tuple[str, ...],
) -> Histogram:
    collector = _lookup_collector(registry, name)
    if collector is not None:
        return cast(Histogram, collector)
    return Histogram(name, documentation, labelnames=label_names, registry=registry)


def _get_or_create_gauge(registry: CollectorRegistry, name: str, documentation: str) -> Gauge:
    collector = _lookup_collector(registry, name)
    if collector is not None:
        return cast(Gauge, collector)
    return Gauge(name, documentation, registry=registry)


def _lookup_collector(registry: CollectorRegistry, name: str) -> Any | None:
    names_to_collectors = getattr(registry, "_names_to_collectors", None)
    if isinstance(names_to_collectors, dict):
        return names_to_collectors.get(name)
    return None
