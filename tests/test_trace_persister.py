from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models.rag_trace import RAGTraceORM
from app.observability.trace_manager import TraceManager
from app.observability.trace_persister import TracePersister


@pytest.fixture
def in_mem_db():
    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture
def persister(in_mem_db):
    return TracePersister(in_mem_db)


def test_persist_and_retrieve(persister, in_mem_db) -> None:
    trace = TraceManager("test_pipeline")
    with trace.span("retrieve"):
        with trace.span("embed"):
            pass

    trace_id = persister.persist(
        trace=trace,
        tenant_id="tenant1",
        session_id="session1",
        query="test query",
        answer="test answer",
    )

    assert trace_id is not None
    assert len(trace_id) > 0

    record = in_mem_db.get(RAGTraceORM, trace_id)
    assert record is not None
    assert record.tenant_id == "tenant1"
    assert record.query == "test query"
    assert record.answer == "test answer"
    assert record.total_duration_ms > 0
    assert "name" in record.span_tree_json


def test_get_trace(persister) -> None:
    trace = TraceManager("test")
    trace.start_span("op")
    trace.end_span()
    trace_id = persister.persist(trace, "t1", "s1", "q", "a")

    retrieved = persister.get_trace(trace_id)
    assert retrieved is not None
    assert retrieved["trace_id"] == trace_id
    assert retrieved["query"] == "q"
    assert retrieved["answer"] == "a"


def test_get_trace_not_found(persister) -> None:
    retrieved = persister.get_trace("nonexistent")
    assert retrieved is None


def test_list_traces(persister) -> None:
    for i in range(3):
        trace = TraceManager("test")
        persister.persist(trace, "t1", f"s{i}", f"q{i}", f"a{i}")

    traces = persister.list_traces("t1")
    assert len(traces) == 3
    assert traces[0]["query"] == "q2"


def test_list_traces_empty(persister) -> None:
    traces = persister.list_traces("nonexistent")
    assert traces == []


def test_persist_none_answer(persister, in_mem_db) -> None:
    trace = TraceManager("test")
    trace_id = persister.persist(trace, "t1", "s1", "q", None)

    record = in_mem_db.get(RAGTraceORM, trace_id)
    assert record is not None
    assert record.answer is None
