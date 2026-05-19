"""Locust load test for NeuralFlow API.

Usage: locust -f tests/load/locustfile.py --host http://localhost:8000
"""
from __future__ import annotations

from locust import HttpUser, between, task


class NeuralFlowUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        self.client.get("/healthz")

    @task(2)
    def list_documents(self):
        self.client.get("/api/v1/documents")

    @task(1)
    def list_eval_runs(self):
        self.client.get("/api/v1/eval/runs")

    @task(1)
    def list_traces(self):
        self.client.get("/api/v1/traces")

    @task(1)
    def list_skills(self):
        self.client.get("/api/v1/skills")

    @task(1)
    def list_models(self):
        self.client.get("/api/v1/models")
