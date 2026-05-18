from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GuardResult:
    allowed: bool
    reason: str = ""
    sanitized_input: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


# Dangerous patterns for terminal commands
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"(^|\s)rm\s+(-rf?|--recursive)(\s|$)", "Recursive deletion detected"),
    (r"(^|\s)dd\s+", "Low-level disk write command"),
    (r"(^|\s)mkfs\.", "Filesystem format command"),
    (r"(^|\s)(poweroff|reboot|shutdown|halt)(\s|$)", "System shutdown command"),
    (r"(^|\s)chmod\s+777(\s|$)", "Overly permissive permission change"),
    (r"(^|\s)chown\s", "Ownership change (blocked by default)"),
    (r"(^|\s)passwd(\s|$)", "Password change command"),
    (r"(^|\s)su(\s|$)", "Switch user command"),
    (r"(^|\s)sudo(\s|$)", "Superuser execute"),
    (r"(^|\s)kill\s+-9(\s|$)", "Force kill signal"),
    (r"(^|\s)>(>\s)?\s*/dev/", "Write to block device"),
    (r"(^|\s):\(\)\s*\{", "Fork bomb pattern"),
    (r"(^|\s)wget\s+.*\|", "Unsafe pipe from download"),
    (r"(^|\s)curl\s+.*\|", "Unsafe pipe from download"),
]

MAX_COMMAND_LENGTH = 4096
MAX_OUTPUT_SIZE = 100_000  # characters


def validate_terminal_command(command: str) -> GuardResult:
    """Validate a terminal command against dangerous patterns."""
    if not command.strip():
        return GuardResult(allowed=False, reason="Empty command")

    if len(command) > MAX_COMMAND_LENGTH:
        return GuardResult(
            allowed=False,
            reason=f"Command exceeds max length ({len(command)} > {MAX_COMMAND_LENGTH})",
            details={"command_length": len(command), "max_length": MAX_COMMAND_LENGTH},
        )

    for pattern, reason in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return GuardResult(allowed=False, reason=reason)

    return GuardResult(allowed=True, reason="Command passes validation")


def validate_skill_call(
    skill_name: str,
    params: dict[str, Any],
    *,
    read_only: bool = True,
    max_param_length: int = 10_000,
) -> GuardResult:
    """Validate a skill call against permission and parameter constraints."""
    if not read_only and skill_name == "terminal":
        cmd = params.get("command", "")
        return validate_terminal_command(cmd)

    for key, value in params.items():
        if isinstance(value, str) and len(value) > max_param_length:
            return GuardResult(
                allowed=False,
                reason=f"Parameter '{key}' exceeds max length ({len(value)} > {max_param_length})",
                details={"param": key, "length": len(value), "max_length": max_param_length},
            )

    return GuardResult(allowed=True, reason="Skill call passes validation")


def sanitize_output(result: str, max_size: int = MAX_OUTPUT_SIZE) -> str:
    """Truncate and sanitize output to safe size."""
    if not result:
        return result
    if len(result) > max_size:
        return result[:max_size] + f"\n\n...[truncated {len(result) - max_size} characters]"
    return result


def contains_sensitive_data(text: str) -> bool:
    """Check if text contains potential sensitive information patterns."""
    patterns = [
        (r"-----BEGIN (RSA |EC )?PRIVATE KEY-----", "Private key"),
        (
            r"(?:^|\s)(?:export|set)\s+\w*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD)\s*=",
            "Credential export",
        ),
        (r"(?:^|\s)(?:ghp|gho|github_pat)_[a-zA-Z0-9]{36}", "GitHub token"),
        (r"sk-[a-zA-Z0-9]{20,}", "OpenAI-style API key"),
        (r"(?:\b|_)AKIA[0-9A-Z]{16}\b", "AWS access key"),
    ]
    return any(re.search(pattern, text) for pattern, _ in patterns)


class RateLimiter:
    """Simple in-memory rate limiter per session."""

    def __init__(self, max_calls: int = 50, window_seconds: float = 60.0) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._records: dict[str, list[float]] = {}

    def check(self, session_id: str) -> GuardResult:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        calls = self._records.setdefault(session_id, [])
        calls[:] = [t for t in calls if t > cutoff]
        if len(calls) >= self.max_calls:
            return GuardResult(
                allowed=False,
                reason=f"Rate limit exceeded: {len(calls)} calls in {self.window_seconds}s window (max {self.max_calls})",
                details={
                    "calls_in_window": len(calls),
                    "max_calls": self.max_calls,
                    "window_seconds": self.window_seconds,
                },
            )
        calls.append(now)
        return GuardResult(allowed=True, reason="Rate limit OK")
