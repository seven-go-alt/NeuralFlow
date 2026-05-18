from __future__ import annotations

import pytest

from app.skills.terminal_exec import TerminalResult, format_terminal_result


class TestTerminalResult:
    def test_defaults(self) -> None:
        r = TerminalResult(stdout="out", stderr="err", return_code=0)
        assert r.timed_out is False


class TestFormatTerminalResult:
    def test_stdout_and_stderr(self) -> None:
        r = TerminalResult(stdout="hello", stderr="warning", return_code=0)
        result = format_terminal_result(r)
        assert "hello" in result
        assert "warning" in result
        assert "exit code 0" in result

    def test_stdout_only(self) -> None:
        r = TerminalResult(stdout="output", stderr="", return_code=0)
        result = format_terminal_result(r)
        assert "[stdout]" in result
        assert "[stderr]" not in result

    def test_stderr_only(self) -> None:
        r = TerminalResult(stdout="", stderr="error", return_code=1)
        result = format_terminal_result(r)
        assert "[stderr]" in result

    def test_timed_out(self) -> None:
        r = TerminalResult(stdout="", stderr="", return_code=-1, timed_out=True)
        result = format_terminal_result(r)
        assert "timed out" in result

    def test_no_output(self) -> None:
        r = TerminalResult(stdout="", stderr="", return_code=0)
        result = format_terminal_result(r)
        assert "(no output)" in result

    def test_empty_result_with_error_code(self) -> None:
        r = TerminalResult(stdout="", stderr="", return_code=127)
        result = format_terminal_result(r)
        assert "exit code 127" in result

    def test_normal_exit_code(self) -> None:
        r = TerminalResult(stdout="done", stderr="", return_code=0)
        result = format_terminal_result(r)
        assert "exit code 0" in result
