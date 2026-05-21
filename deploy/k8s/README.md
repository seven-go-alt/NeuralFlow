# NeuralFlow Kubernetes Deployment

This directory contains Kubernetes manifests for deploying NeuralFlow in a production-like environment.

## Prerequisites

- Kubernetes cluster (v1.25+ recommended)
- kubectl configured with cluster access
- A storage class that supports dynamic provisioning (for PVCs)

## Quick Start

```bash
# Create the namespace and all resources
kubectl apply -k deploy/k8s/

# Or, step by step:
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/secret.yaml      # edit first with real secrets!
kubectl apply -f deploy/k8s/redis.yaml
kubectl apply -f deploy/k8s/chromadb.yaml
kubectl apply -f deploy/k8s/postgres.yaml
kubectl apply -f deploy/k8s/api.yaml
kubectl apply -f deploy/k8s/worker.yaml
kubectl apply -f deploy/k8s/frontend.yaml
kubectl apply -f deploy/k8s/ingress.yaml
```

## Required Secrets

Before deploying, edit `secret.yaml` and replace all `change-me` values with real secrets:

| Secret Key              | Description                       |
|-------------------------|-----------------------------------|
| `POSTGRES_PASSWORD`     | PostgreSQL password (base64)      |
| `DATABASE_URL`          | Full database connection string   |
| `OPENAI_API_KEY`        | OpenAI API key                    |
| `LLM_API_KEY`           | LLM provider API key              |
| `EMBEDDING_API_KEY`     | Embedding model API key           |
| `AUTH_JWT_SECRET`       | JWT signing secret                |
| `SENTRY_DSN`            | Sentry DSN (optional)             |

## Configuration

All non-sensitive configuration is in `configmap.yaml`. Adjust values as needed for your environment.

## Container Images

Set the image names via environment variables when building or edit the manifests directly:

- `NEURALFLOW_API_IMAGE` — FastAPI + Celery worker image
- `NEURALFLOW_FRONTEND_IMAGE` — Next.js frontend image

## Monitoring

The API exposes `/healthz` for liveness/readiness checks and `/metrics` for Prometheus metrics.

## Services & Ports

| Service    | Internal Port | Protocol |
|------------|---------------|----------|
| API        | 8000          | HTTP     |
| Frontend   | 3000          | HTTP     |
| Redis      | 6379          | TCP      |
| ChromaDB   | 8000          | HTTP     |
| PostgreSQL | 5432          | TCP      |
| Ingress    | 80 / 443      | HTTP     |

## PVC Storage

| PVC            | Size | Mount Path               |
|----------------|------|--------------------------|
| redis-data     | 1Gi  | /data                    |
| chroma-data    | 10Gi | /data                    |
| postgres-data  | 10Gi | /var/lib/postgresql/data |
| api-uploads    | 5Gi  | /data/uploads            |

## Helm Chart

A Helm chart is also available at `deploy/helm/`:

```bash
helm install neuralflow ./deploy/helm/ --namespace neuralflow --create-namespace
```
