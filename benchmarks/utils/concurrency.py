from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def run_concurrent(
    tasks: list[Callable[[], Awaitable[T]]],
    max_concurrency: int = 5,
) -> list[T]:
    """Run async tasks with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run(task: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await task()

    return await asyncio.gather(*[_run(t) for t in tasks])
