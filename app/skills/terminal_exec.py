from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("neuralflow.skills.terminal")


@dataclass(slots=True, frozen=True)
class TerminalResult:
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False


async def execute_command(
    command: str,
    *,
    timeout: float = 30.0,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> TerminalResult:
    """Execute a shell command locally and return the result."""
    resolved_cwd = cwd if cwd else os.getcwd()
    merged_env = {**os.environ, **(env or {})}

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=resolved_cwd,
            env=merged_env,
        )
    except FileNotFoundError:
        return TerminalResult(
            stdout="",
            stderr="Shell not found in the current environment",
            return_code=127,
        )
    except PermissionError:
        return TerminalResult(
            stdout="",
            stderr="Permission denied when trying to execute the command",
            return_code=126,
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return_code = process.returncode or 0
        return TerminalResult(stdout=stdout, stderr=stderr, return_code=return_code)
    except TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        logger.warning("Terminal command timed out after %ss: %s", timeout, command[:120])
        return TerminalResult(
            stdout=stdout,
            stderr=stderr + f"\n[Command timed out after {timeout}s]",
            return_code=-1,
            timed_out=True,
        )


def format_terminal_result(result: TerminalResult) -> str:
    """Format a TerminalResult into a string suitable for LLM context."""
    parts: list[str] = []
    if result.stdout:
        parts.append(f"[stdout]\n{result.stdout.rstrip()}")
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr.rstrip()}")
    if result.return_code == -1 and result.timed_out:
        parts.append("[status] timed out")
    else:
        parts.append(f"[status] exit code {result.return_code}")
    if not result.stdout and not result.stderr:
        parts.append("(no output)")
    return "\n".join(parts)
