# NeuralFlow Deployment Guide

## Prerequisites

- Docker & Docker Compose v2
- Python 3.11+ (local dev only)
- uv (package manager)
- OpenAI API key or compatible LLM endpoint
- (Optional) ChromaDB, Redis — provided via Docker Compose

## Quick Start (Local Development)

```bash
# 1. Clone and enter
git clone https://github.com/seven-go-alt/NeuralFlow.git
cd NeuralFlow

# 2. Copy environment config
cp .env.example .env
# Edit .env with your API keys

# 3. Start dependencies (Redis, ChromaDB)
docker compose up -d redis chroma

# 4. Install dependencies
uv sync --group dev

# 5. Run database migrations
uv run python -c "from app.db.session import init_db; init_db()"

# 6. Start the API
uv run uvicorn app.main:app --reload --port 8000

# 7. (Optional) Start Celery worker
uv run celery -A worker.celery_app worker --loglevel=info
```

## Docker Compose (Production)

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key |
| `LLM_API_BASE` | Yes* | — | Custom LLM API base URL |
| `LLM_API_KEY` | No | — | Custom LLM API key |
| `DATABASE_URL` | No | `sqlite:///./data/neuralflow.db` | SQLAlchemy database URL |
| `REDIS_HOST` | No | `localhost` | Redis host |
| `CHROMA_HOST` | No | `127.0.0.1` | ChromaDB host |
| `AUTH_ENABLED` | No | `false` | Enable JWT authentication |
| `ADMIN_SECRET_KEY` | No | — | Secret for `/admin/config` API |
| `MULTIMODAL_ENABLED` | No | `false` | Enable image/table extraction |
| `RAG_ADVANCED_ENABLED` | No | `false` | Enable advanced RAG pipeline |

*\* Either `OPENAI_API_KEY` or `LLM_API_BASE` + `LLM_API_KEY` is required.*

### Full Stack Deployment

```bash
# Start all services
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check status
docker compose ps

# View logs
docker compose logs -f api worker
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI application (gunicorn + uvicorn) |
| worker | — | Celery async task worker |
| redis | 6379 | Cache, Celery broker, session store |
| chroma | 8001 | Vector database |

## Monitoring Stack

```bash
# Start monitoring (Prometheus + Grafana + Loki + Promtail)
docker compose -f docker-compose.monitoring.yml up -d

# Access:
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

Pre-configured dashboards:
- Request latency & error rates
- LLM token usage
- Active sessions
- RAG evaluation metrics

## Health Checks

```bash
# Basic health
curl http://localhost:8000/healthz

# Response:
# {
#   "status": "ok",
#   "app": "NeuralFlow",
#   "database": {"status": "up"},
#   "chromadb": {"status": "up"},
#   "redis": {"status": "up"},
#   "duration_ms": 15.3
# }
```

## Production Considerations

### 1. PostgreSQL (replace SQLite)

```env
DATABASE_URL=postgresql://user:pass@host:5432/neuralflow
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

### 2. HTTPS & Domain

Use the provided `deploy/Caddyfile` for automatic TLS:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d caddy
```

### 3. Authentication

```env
AUTH_ENABLED=true
AUTH_JWT_SECRET=<generate-a-strong-random-secret>
```

### 4. Persistent Volumes

Docker Compose mounts volumes for:
- `chroma_data` — Vector database persistence
- Project `data/` directory — Uploaded documents, SQLite (if used)

### 5. Resource Requirements

| Service | CPU | Memory | Storage |
|---------|-----|--------|---------|
| API | 1 core | 512 MB | — |
| Worker | 1 core | 512 MB | — |
| Redis | 0.5 core | 256 MB | — |
| ChromaDB | 1 core | 1 GB | 10 GB+ |

## Proxy Configuration

See `deploy/proxy-guide.md` for Docker proxy settings in air-gapped environments.
