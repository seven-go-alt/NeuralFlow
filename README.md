# NeuralFlow

NeuralFlow 是一个基于 FastAPI、Redis、ChromaDB 和 Celery 的 Agent 工程模板，当前方案聚焦轻量原型开发：

- L0 短期记忆：Redis 滑动窗口保存最近 10 轮对话
- L2 长期记忆：ChromaDB 存储压缩摘要与向量检索结果
- Token 优化：按意图决定是否查询长期记忆
- 异步归档：Celery 将过期对话压缩后写入长期记忆

这个版本不引入 MongoDB，避免额外的数据库运维负担。记忆层仍保留抽象基类，后续如果要扩展 SQLite、Milvus 或其他存储，可以继续往下接。

## 项目结构

```text
NeuralFlow/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── core/
│   │   ├── context.py
│   │   ├── llm.py
│   │   └── router.py
│   ├── memory/
│   │   ├── base.py
│   │   ├── long_term.py
│   │   ├── summarizer.py
│   │   └── working.py
│   ├── skills/
│   │   ├── mcp_client.py
│   │   └── registry.py
│   └── utils/
│       ├── redis_client.py
│       └── vector_client.py
├── src/neuralflow/__init__.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── worker.py
└── .env.example
```

## 用 uv 管理 Python 与依赖

```bash
cd ~/github/NeuralFlow
cp .env.example .env
uv sync
uv run pytest -q
uv run uvicorn app.main:app --reload
```

`.python-version` 已固定为 `3.11`。

## 环境变量

核心配置只围绕 Redis 和 ChromaDB：

- `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB`
- `CHROMA_HOST` / `CHROMA_PORT` / `CHROMA_COLLECTION`
- `OPENAI_API_KEY` 或其他兼容 LiteLLM 的凭证

Celery 默认直接复用 Redis：

- broker 默认使用 `redis://<host>:<port>/<db>`
- result backend 默认使用下一个 DB：`redis://<host>:<port>/<db+1>`

只有在你需要单独拆分时，才需要显式设置 `CELERY_BROKER_URL` 和 `CELERY_RESULT_BACKEND`。

## Docker Compose

```bash
docker compose up --build
```

启动后：

- API: http://localhost:8000
- ChromaDB: http://localhost:8001
- Redis: localhost:6379

## 基本接口

- `GET /healthz`：健康检查
- `GET /metrics`：导出 Prometheus 指标（请求耗时、错误数、活跃会话数）
- `GET /admin/config`：读取当前运行时配置快照，需要管理密钥（兼容 `X-Admin-Secret` 或 `Authorization: Bearer <token>`）
- `PATCH /admin/config`：热更新运行时配置并写入审计日志，需要管理密钥（兼容 `X-Admin-Secret` 或 `Authorization: Bearer <token>`）
- `GET /api/skills`：列出当前注册并可暴露给策略层的技能
- `POST /api/intent/detect`：返回意图识别结果、是否使用 fallback，以及每个意图对应的记忆/技能策略
- `POST /chat`：写入短期记忆，按意图构造上下文，按白名单调用 MCP 技能，并通过 LiteLLM 生成回复
- `POST /chat/stream`：输出 SSE 流式响应，并按运行时配置决定是否透出 thinking/reasoning 片段

`POST /chat` 当前会额外返回：

- `used_skills`：本轮实际执行的技能名列表
- `skill_results`：每个技能的执行结果，便于调试 MCP 调用链路

## 管理接口示例

先设置管理密钥：

```bash
export ADMIN_SECRET_KEY=test-admin-key
```

读取当前运行时配置：

```bash
curl -s http://localhost:8000/admin/config \
  -H "X-Admin-Secret: $ADMIN_SECRET_KEY" | jq
```

等价写法（Bearer 认证）:

```bash
curl -s http://localhost:8000/admin/config \
  -H "Authorization: Bearer $ADMIN_SECRET_KEY" | jq
```

示例响应：

```json
{
  "config": {
    "working_memory_max_turns": 10,
    "max_context_tokens_soft": 6000,
    "max_context_tokens": 8000,
    "token_budget_recent_messages": 4,
    "vector_search_cache_ttl_seconds": 300,
    "vector_search_default_top_k": 3,
    "stream_thinking_enabled": false,
    "intent_llm_fallback_enabled": true,
    "litellm_model": "gpt-4o-mini",
    "model_routing_strategy": "primary"
  },
  "audit_entry": null
}
```

热更新配置：

```bash
curl -s http://localhost:8000/admin/config \
  -X PATCH \
  -H "Content-Type: application/json" \
  -H "X-Admin-Secret: $ADMIN_SECRET_KEY" \
  -d '{
    "max_context_tokens": 4096,
    "stream_thinking_enabled": true
  }' | jq
```

示例响应：

```json
{
  "config": {
    "working_memory_max_turns": 10,
    "max_context_tokens_soft": 4096,
    "max_context_tokens": 4096,
    "token_budget_recent_messages": 4,
    "vector_search_cache_ttl_seconds": 300,
    "vector_search_default_top_k": 3,
    "stream_thinking_enabled": true,
    "intent_llm_fallback_enabled": true,
    "litellm_model": "gpt-4o-mini",
    "model_routing_strategy": "primary"
  },
  "audit_entry": {
    "timestamp": "2026-04-17T02:00:00Z",
    "actor": "admin_api",
    "source_ip": "127.0.0.1",
    "changes": {
      "max_context_tokens_soft": {"old": 6000, "new": 4096},
      "max_context_tokens": {"old": 8000, "new": 4096},
      "stream_thinking_enabled": {"old": false, "new": true}
    }
  }
}
```

说明：当你只下调 `max_context_tokens` 而未显式传入 `max_context_tokens_soft` 时，服务会自动把 soft limit 一并收缩，避免配置进入非法状态。

## Metrics 指标说明

默认通过 `GET /metrics` 暴露 Prometheus 文本格式指标，当前重点包括：

- `neuralflow_request_duration_seconds{endpoint, intent}`：请求耗时直方图
- `neuralflow_errors_total{endpoint, intent}`：未处理异常计数
- `neuralflow_active_sessions`：当前 in-flight 会话数
- `neuralflow_llm_token_usage_total{model, type}`：LLM 输入/输出 token 计数
- `neuralflow_memory_cache_hit_total{layer}`：记忆缓存命中计数

抓取示例：

```bash
curl -s http://localhost:8000/metrics | grep neuralflow_
```

如果以多进程方式部署 Prometheus client，可额外设置：

- `PROMETHEUS_MULTIPROC_DIR`：启用 multiprocess collector
- `NEURALFLOW_AUDIT_LOG_PATH`：指定结构化审计日志文件路径

## 测试与压测

安装开发依赖后，可以直接运行：

```bash
uv sync --group dev
uv run pytest tests/ -v --cov=app
```

如果你使用项目已有 `.venv`，也可执行：

```bash
source .venv/bin/activate
pytest tests/ -v --cov=app
```

项目根目录已提供 `load_test.py`，支持 `healthz`、`metrics`、`/admin/config` 以及可选的 `/chat` 压测：

```bash
export ADMIN_SECRET_KEY=***
locust -f load_test.py --host http://localhost:8000
```

说明：
- 默认压测 `GET /healthz`、`GET /metrics`、`GET/PATCH /admin/config`
- 默认使用 `Authorization: Bearer $ADMIN_SECRET_KEY` 调用管理接口
- 如需把 `/chat` 也加入压测，启动前设置 `NEURALFLOW_LOAD_ENABLE_CHAT=1`
- 若本地未配置外部 LLM 凭证，建议先只压 `healthz/metrics/admin` 链路

## 最近补充能力

- 可观测性：新增结构化请求日志、`TelemetryMiddleware` 与 `/metrics` 指标导出
- 运行时配置：新增 `ConfigManager`，支持通过管理接口热更新开关类配置并记录审计日志
- 流式响应：`/chat/stream` 支持 SSE 输出，并兼容 thinking 开关与中断注册表

## 开发路线调整

Phase 1：基础设施
- 启动 Redis 和 ChromaDB
- 跑通 FastAPI 骨架和健康检查

Phase 2：记忆模块
- 完成 `WorkingMemory` 的 Redis 滑动窗口逻辑
- 完成 `LongTermMemory` 的 Chroma 检索与写入逻辑

Phase 3：上下文优化
- 实现按意图查询长期记忆的 `ContextBuilder`
- 将短期记忆、长期记忆和 LLM 调用串起来

Phase 4：异步归档与技能调用
- 用 Celery 将历史对话压缩为摘要后异步写入 ChromaDB
- 接入 MCP/技能调用能力

Phase 5：运行时治理
- 增加请求链路可观测性与 Prometheus 指标
- 提供运行时配置热更新与审计能力

当前代码已完成 Phase 5 的主线能力，并补上了对应测试。
