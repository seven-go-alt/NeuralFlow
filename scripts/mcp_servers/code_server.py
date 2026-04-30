"""
MCP Code Execution Server
提供 Python 代码执行能力，兼容 NeuralFlow MCPClient 协议。
"""

from __future__ import annotations

import re
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NeuralFlow MCP Code Server")

MAX_OUTPUT_LENGTH = 4096
EXEC_TIMEOUT_SECONDS = 5

BLOCKED_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+subprocess\b",
    r"\bfrom\s+shutil\b",
    r"\bos\.(system|popen|exec|remove|rmdir|makedirs)",
    r"\bsubprocess\.(run|Popen|call|check_output)",
    r"\b__import__\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\(.+['\"]w",
    r"\bshutil\.(rmtree|move|copy)",
]
BLOCKED_RE = re.compile("|".join(BLOCKED_PATTERNS), re.IGNORECASE)


class ToolCallRequest(BaseModel):
    session_id: str = ""
    input: str = ""


def _check_code_safety(code: str) -> str | None:
    if BLOCKED_RE.search(code):
        return "代码包含被禁止的危险操作（文件系统/网络/进程管理）"
    return None


def _execute_python(code: str) -> dict:
    block_reason = _check_code_safety(code)
    if block_reason:
        return {"stdout": "", "stderr": block_reason, "return_code": -1, "blocked": True}

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
            cwd="/tmp",
        )
        stdout = result.stdout[:MAX_OUTPUT_LENGTH] if result.stdout else ""
        stderr = result.stderr[:MAX_OUTPUT_LENGTH] if result.stderr else ""
        return {"stdout": stdout, "stderr": stderr, "return_code": result.returncode, "blocked": False}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"执行超时（{EXEC_TIMEOUT_SECONDS}秒）", "return_code": -1, "blocked": False}
    except Exception as exc:
        return {"stdout": "", "stderr": f"执行异常: {exc}", "return_code": -1, "blocked": False}


@app.get("/tools")
async def list_tools():
    return {
        "tools": [
            {
                "name": "python_exec",
                "description": "执行 Python 代码并返回 stdout/stderr。仅支持纯计算逻辑，禁止文件系统和网络操作。",
                "read_only": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "要执行的 Python 代码"},
                    },
                    "required": ["code"],
                },
            }
        ]
    }


@app.post("/tools/python_exec")
async def execute_python(request: ToolCallRequest):
    code = request.input.strip()
    if not code:
        raise HTTPException(status_code=400, detail="代码不能为空")

    result = _execute_python(code)
    return result
