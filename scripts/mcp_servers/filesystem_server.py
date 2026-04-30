"""
MCP Filesystem Server
提供沙箱化的文件读写能力，兼容 NeuralFlow MCPClient 协议。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NeuralFlow MCP Filesystem Server")

DEFAULT_SANDBOX = "/tmp/neuralflow_sandbox"
SANDBOX_ROOT = Path(os.getenv("MCP_FILESYSTEM_ROOT", DEFAULT_SANDBOX)).resolve()
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB


class FileReadRequest(BaseModel):
    session_id: str = ""
    path: str = ""


class FileWriteRequest(BaseModel):
    session_id: str = ""
    path: str = ""
    content: str = ""


class FileListRequest(BaseModel):
    session_id: str = ""
    path: str = ""


def _resolve_safe_path(requested_path: str) -> Path:
    resolved = (SANDBOX_ROOT / requested_path).resolve()
    if not str(resolved).startswith(str(SANDBOX_ROOT)):
        raise HTTPException(status_code=403, detail="路径遍历被拒绝：不允许访问沙箱外的文件")
    return resolved


@app.get("/tools")
async def list_tools():
    return {
        "tools": [
            {
                "name": "file_read",
                "description": "读取沙箱目录中的文件内容",
                "read_only": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对于沙箱根目录的文件路径"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "file_write",
                "description": "将内容写入沙箱目录中的文件（自动创建父目录）",
                "read_only": False,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对于沙箱根目录的文件路径"},
                        "content": {"type": "string", "description": "要写入的文件内容"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "file_list",
                "description": "列出沙箱目录中的文件和子目录",
                "read_only": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对于沙箱根目录的目录路径，默认为根目录"},
                    },
                },
            },
        ]
    }


@app.post("/tools/file_read")
async def read_file(request: FileReadRequest):
    file_path = _resolve_safe_path(request.path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {request.path}")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"不是文件: {request.path}")

    content = file_path.read_text(encoding="utf-8", errors="replace")
    if len(content) > MAX_FILE_SIZE:
        content = content[:MAX_FILE_SIZE] + "\n... [截断：文件超过 1MB]"
    return {"path": request.path, "content": content, "size": len(content)}


@app.post("/tools/file_write")
async def write_file(request: FileWriteRequest):
    file_path = _resolve_safe_path(request.path)
    if len(request.content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="内容超过 1MB 限制")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(request.content, encoding="utf-8")
    return {"path": request.path, "bytes_written": len(request.content)}


@app.post("/tools/file_list")
async def list_files(request: FileListRequest):
    dir_path = _resolve_safe_path(request.path or ".")
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"目录不存在: {request.path}")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"不是目录: {request.path}")

    entries = []
    for item in sorted(dir_path.iterdir()):
        entries.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None,
        })
    return {"path": request.path or ".", "entries": entries}
