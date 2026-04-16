from __future__ import annotations

import os
import random
from uuid import uuid4

from locust import HttpUser, between, task


ADMIN_SECRET = os.getenv("ADMIN_SECRET_KEY", "")
ENABLE_CHAT = os.getenv("NEURALFLOW_LOAD_ENABLE_CHAT", "0").lower() in {"1", "true", "yes", "on"}
CHAT_MESSAGE = os.getenv("NEURALFLOW_LOAD_CHAT_MESSAGE", "你好，请简要总结当前会话。")
ADMIN_AUTH_MODE = os.getenv("NEURALFLOW_LOAD_ADMIN_AUTH", "bearer").lower()
MAX_CONTEXT_CHOICES = [2048, 3072, 4096, 6144, 8000]


def build_admin_headers() -> dict[str, str]:
    if not ADMIN_SECRET:
        return {}
    if ADMIN_AUTH_MODE == "x-admin-secret":
        return {"X-Admin-Secret": ADMIN_SECRET}
    return {"Authorization": f"Bearer {ADMIN_SECRET}"}


class NeuralFlowUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        headers = build_admin_headers()
        if not headers:
            return
        self.client.get("/admin/config", headers=headers, name="GET /admin/config")
        self.client.patch(
            "/admin/config",
            headers={**headers, "Content-Type": "application/json"},
            json={"max_context_tokens": random.choice(MAX_CONTEXT_CHOICES)},
            name="PATCH /admin/config",
        )

    @task(5)
    def healthz(self) -> None:
        self.client.get("/healthz", name="GET /healthz")

    @task(3)
    def metrics(self) -> None:
        self.client.get("/metrics", name="GET /metrics")

    @task(2)
    def admin_config_read(self) -> None:
        headers = build_admin_headers()
        if not headers:
            return
        self.client.get("/admin/config", headers=headers, name="GET /admin/config")

    @task(1)
    def admin_config_patch(self) -> None:
        headers = build_admin_headers()
        if not headers:
            return
        payload = {"max_context_tokens": random.choice(MAX_CONTEXT_CHOICES)}
        self.client.patch(
            "/admin/config",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            name="PATCH /admin/config",
        )

    @task(2)
    def chat(self) -> None:
        if not ENABLE_CHAT:
            return
        payload = {
            "session_id": f"load-{uuid4()}",
            "message": CHAT_MESSAGE,
        }
        self.client.post("/chat", json=payload, name="POST /chat")
