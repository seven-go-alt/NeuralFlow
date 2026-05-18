from __future__ import annotations

import pytest

from app.observability.trace_manager import TraceManager, TraceSpan


class TestTraceManager:
    def test_creates_root_span(self) -> None:
        mgr = TraceManager("test")
        assert mgr.root.name == "test"
        assert mgr.root.span_id != ""
        assert mgr.root.trace_id != ""

    def test_start_and_end_span(self) -> None:
        mgr = TraceManager()
        span = mgr.start_span("child")
        assert span.name == "child"
        assert span.parent_id == mgr.root.span_id
        mgr.end_span()
        assert span.duration_ms > 0

    def test_nested_spans(self) -> None:
        mgr = TraceManager("root")
        with mgr.span("parent"):
            with mgr.span("child"):
                pass
        assert len(mgr.root.children) == 1
        assert mgr.root.children[0].name == "parent"
        assert len(mgr.root.children[0].children) == 1
        assert mgr.root.children[0].children[0].name == "child"

    def test_close_closes_all(self) -> None:
        mgr = TraceManager()
        mgr.start_span("child")
        mgr.close()
        assert mgr.root.duration_ms > 0

    def test_span_context_manager(self) -> None:
        mgr = TraceManager()
        with mgr.span("op", key="value") as span:
            assert span.metadata.get("key") == "value"
        assert span.duration_ms > 0

    def test_to_dict(self) -> None:
        mgr = TraceManager("root")
        with mgr.span("op1"):
            pass
        d = mgr.to_dict()
        assert d["name"] == "root"
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "op1"
        assert "duration_ms" in d

    def test_current_property(self) -> None:
        mgr = TraceManager()
        assert mgr.current is mgr.root
        mgr.start_span("child")
        assert mgr.current.name == "child"

    def test_to_dict_trace_id_consistency(self) -> None:
        mgr = TraceManager()
        with mgr.span("child"):
            pass
        d = mgr.to_dict()
        assert d["trace_id"] == mgr.root.trace_id
        assert d["children"][0]["trace_id"] == mgr.root.trace_id


class TestTraceSpan:
    def test_close_sets_duration(self) -> None:
        span = TraceSpan(name="test")
        span.start_time = 0.0
        span.close()
        assert span.duration_ms > 0

    def test_to_dict(self) -> None:
        span = TraceSpan(name="test", span_id="s1", trace_id="t1", duration_ms=10.0)
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["span_id"] == "s1"
