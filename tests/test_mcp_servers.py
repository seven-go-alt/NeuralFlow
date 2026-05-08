"""Tests for MCP Code Execution and Filesystem servers."""

from __future__ import annotations

from fastapi.testclient import TestClient

from scripts.mcp_servers.code_server import app as code_app
from scripts.mcp_servers.filesystem_server import SANDBOX_ROOT
from scripts.mcp_servers.filesystem_server import app as fs_app

# ── Code Server Tests ──


class TestCodeServer:
    def setup_method(self):
        self.client = TestClient(code_app)

    def test_list_tools(self):
        resp = self.client.get("/tools")
        assert resp.status_code == 200
        tools = resp.json()["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "python_exec"
        assert tools[0]["read_only"] is True

    def test_execute_simple_code(self):
        resp = self.client.post(
            "/tools/python_exec", json={"session_id": "t", "input": "print(2 + 3)"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["stdout"].strip() == "5"
        assert data["return_code"] == 0
        assert data["blocked"] is False

    def test_execute_multi_line_code(self):
        code = "for i in range(3):\n    print(i)"
        resp = self.client.post("/tools/python_exec", json={"session_id": "t", "input": code})
        assert resp.status_code == 200
        assert "0\n1\n2" in resp.json()["stdout"]

    def test_execute_code_with_error(self):
        resp = self.client.post("/tools/python_exec", json={"session_id": "t", "input": "1/0"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["return_code"] != 0
        assert "ZeroDivisionError" in data["stderr"]

    def test_blocks_os_import(self):
        resp = self.client.post(
            "/tools/python_exec", json={"session_id": "t", "input": "import os; os.system('ls')"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["blocked"] is True
        assert "禁止" in data["stderr"]

    def test_blocks_subprocess_import(self):
        resp = self.client.post(
            "/tools/python_exec",
            json={"session_id": "t", "input": "import subprocess; subprocess.run(['ls'])"},
        )
        assert resp.status_code == 200
        assert resp.json()["blocked"] is True

    def test_blocks_eval_exec(self):
        resp = self.client.post(
            "/tools/python_exec", json={"session_id": "t", "input": "eval('1+1')"}
        )
        assert resp.status_code == 200
        assert resp.json()["blocked"] is True

    def test_blocks_file_write(self):
        resp = self.client.post(
            "/tools/python_exec",
            json={"session_id": "t", "input": "open('/tmp/x', 'w').write('hi')"},
        )
        assert resp.status_code == 200
        assert resp.json()["blocked"] is True

    def test_empty_code_rejected(self):
        resp = self.client.post("/tools/python_exec", json={"session_id": "t", "input": ""})
        assert resp.status_code == 400

    def test_safe_math_code_allowed(self):
        code = "import math; print(math.factorial(10))"
        resp = self.client.post("/tools/python_exec", json={"session_id": "t", "input": code})
        assert resp.status_code == 200
        assert resp.json()["stdout"].strip() == "3628800"
        assert resp.json()["blocked"] is False


# ── Filesystem Server Tests ──


class TestFilesystemServer:
    def setup_method(self):
        self.client = TestClient(fs_app)
        self._sandbox = SANDBOX_ROOT
        self._sandbox.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        import shutil

        if self._sandbox.exists():
            shutil.rmtree(self._sandbox)

    def test_list_tools(self):
        resp = self.client.get("/tools")
        assert resp.status_code == 200
        tools = resp.json()["tools"]
        names = [t["name"] for t in tools]
        assert "file_read" in names
        assert "file_write" in names
        assert "file_list" in names

    def test_write_and_read_file(self):
        write_resp = self.client.post(
            "/tools/file_write",
            json={"session_id": "t", "path": "test.txt", "content": "hello world"},
        )
        assert write_resp.status_code == 200
        assert write_resp.json()["bytes_written"] == 11

        read_resp = self.client.post(
            "/tools/file_read", json={"session_id": "t", "path": "test.txt"}
        )
        assert read_resp.status_code == 200
        assert read_resp.json()["content"] == "hello world"

    def test_write_creates_parent_dirs(self):
        resp = self.client.post(
            "/tools/file_write", json={"session_id": "t", "path": "a/b/c.txt", "content": "nested"}
        )
        assert resp.status_code == 200

        read_resp = self.client.post(
            "/tools/file_read", json={"session_id": "t", "path": "a/b/c.txt"}
        )
        assert read_resp.json()["content"] == "nested"

    def test_list_directory(self):
        self.client.post(
            "/tools/file_write", json={"session_id": "t", "path": "f1.txt", "content": "a"}
        )
        self.client.post(
            "/tools/file_write", json={"session_id": "t", "path": "f2.txt", "content": "b"}
        )

        resp = self.client.post("/tools/file_list", json={"session_id": "t", "path": "."})
        assert resp.status_code == 200
        names = [e["name"] for e in resp.json()["entries"]]
        assert "f1.txt" in names
        assert "f2.txt" in names

    def test_path_traversal_rejected(self):
        resp = self.client.post(
            "/tools/file_read", json={"session_id": "t", "path": "../../../etc/passwd"}
        )
        assert resp.status_code == 403
        assert "遍历" in resp.json()["detail"]

    def test_read_nonexistent_file(self):
        resp = self.client.post(
            "/tools/file_read", json={"session_id": "t", "path": "no_such_file.txt"}
        )
        assert resp.status_code == 404

    def test_list_nonexistent_dir(self):
        resp = self.client.post("/tools/file_list", json={"session_id": "t", "path": "no_such_dir"})
        assert resp.status_code == 404
