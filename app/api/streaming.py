from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


class StreamTaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def register(self, session_id: str, task: asyncio.Task[Any]) -> None:
        previous = self._tasks.get(session_id)
        if previous is not None and previous is not task and not previous.done():
            previous.cancel()
        self._tasks[session_id] = task

    def clear(self, session_id: str, task: asyncio.Task[Any] | None = None) -> None:
        current = self._tasks.get(session_id)
        if task is None or current is task:
            self._tasks.pop(session_id, None)


async def create_sse_response(
    session_id: str,
    event_source: Callable[[], AsyncIterator[dict[str, Any]]],
    registry: StreamTaskRegistry,
) -> StreamingResponse:
    async def stream() -> AsyncIterator[str]:
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("streaming task not available")

        started_at = time.perf_counter()
        registry.register(session_id, current_task)
        try:
            async for item in event_source():
                yield _format_sse(item["event"], item["data"])
            latency = round(time.perf_counter() - started_at, 4)
            logger.info("stream completed", extra={"session_id": session_id, "stream_latency": latency})
            yield _format_sse("done", {"status": "completed", "stream_latency": latency})
        except asyncio.CancelledError:
            latency = round(time.perf_counter() - started_at, 4)
            logger.info("stream cancelled", extra={"session_id": session_id, "stream_latency": latency})
            raise
        except Exception as exc:
            latency = round(time.perf_counter() - started_at, 4)
            logger.exception("stream failed", extra={"session_id": session_id, "stream_latency": latency})
            yield _format_sse("error", {"error": str(exc), "stream_latency": latency})
        finally:
            registry.clear(session_id, current_task)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _format_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
