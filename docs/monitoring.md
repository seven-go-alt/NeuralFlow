# NeuralFlow Monitoring

This document describes how to configure and use the monitoring and alerting infrastructure for NeuralFlow.

## 1. Sentry Integration

Sentry is used for error tracking and performance monitoring.

### Configuration

Sentry is gated behind the `SENTRY_DSN` environment variable. It is only initialized when a DSN is provided.

| Environment Variable | Default | Description |
|---|---|---|
| `SENTRY_DSN` | (none) | Sentry project DSN. Omit to disable Sentry. |
| `SENTRY_TRACES_SAMPLE_RATE` | `0.1` | Traces sampling rate (0.0 to 1.0). |

### How to enable

```bash
export SENTRY_DSN="https://your-dsn@sentry.io/project-id"
export SENTRY_TRACES_SAMPLE_RATE="0.1"
uv run uvicorn app.main:app
```

### Integrations

- **FastApiIntegration** — automatically captures FastAPI request/response spans.
- **SqlalchemyIntegration** — automatically captures SQLAlchemy query spans.

The Sentry environment is set to the value of `settings.app_env` (default: `development`).

---

## 2. Prometheus Configuration

### Running Prometheus locally

Create a `prometheus.yml` configuration:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "neuralflow"
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: "/metrics"
```

Then run Prometheus:

```bash
prometheus --config.file=prometheus.yml
```

### Available Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `neuralflow_request_duration_seconds` | Histogram | `endpoint`, `intent` | Request duration in seconds |
| `neuralflow_http_requests_total` | Counter | `endpoint`, `method`, `status_code` | Total HTTP requests |
| `neuralflow_http_errors_total` | Counter | `endpoint`, `status_code` | HTTP error responses (4xx, 5xx) |
| `neuralflow_llm_token_usage_total` | Counter | `model`, `type` | LLM token usage |
| `neuralflow_memory_cache_hit_total` | Counter | `layer` | Memory cache hit count |
| `neuralflow_active_sessions` | Gauge | (none) | Active in-flight sessions |
| `neuralflow_errors_total` | Counter | `endpoint`, `intent` | Unhandled request errors |

### Alert Rules

Alert rules are defined in `ops/prometheus/alerts.yml`. To load them, include the file in your Prometheus configuration:

```yaml
rule_files:
  - "ops/prometheus/alerts.yml"
```

| Alert Name | Condition | Severity |
|---|---|---|
| `HighErrorRate` | Error ratio > 5% over 5 minutes | critical |
| `HighLatency` | p99 latency > 5s over 5 minutes | warning |
| `HealthCheckDown` | `/healthz` returns non-200 | critical |
| `LowApiThroughput` | Total requests < 10/min over 5 minutes | warning |

---

## 3. Grafana Dashboard

A pre-built Grafana dashboard JSON is available at `ops/grafana/dashboard.json`.

### How to import

1. Open Grafana (default: `http://localhost:3000`).
2. Navigate to **Dashboards > New > Import**.
3. Upload or paste the contents of `ops/grafana/dashboard.json`.
4. Select the **Prometheus** datasource (UID: `-- Grafana --`).
5. Click **Import**.

### Dashboard Panels

| Panel | Description |
|---|---|
| **Request Rate (QPS)** | HTTP request rate per second, broken down by endpoint |
| **Error Ratio** | Ratio of 4xx/5xx responses to total requests |
| **Request Latency (p50 / p95 / p99)** | Latency distribution percentiles over 5-minute windows |
| **Health Check Status** | Current health check status (1 = up, 0 = down) |
| **Active Sessions** | Number of currently active (in-flight) sessions |
| **Error Count by Status Code** | Error rate broken down by HTTP status code |
| **LLM Token Usage** | LLM token consumption rate by model and token type |

---

## 4. Local Development Setup

Start all services locally:

```bash
# 1. Start NeuralFlow
uv run uvicorn app.main:app --reload

# 2. Start Prometheus (with alert rules)
prometheus --config.file=prometheus.yml

# 3. Start Grafana
grafana-server
```

Make requests to NeuralFlow to generate metrics:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/metrics
```

Open Grafana at `http://localhost:3000`, import the dashboard, and observe the panels.
