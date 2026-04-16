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
- `GET /api/skills`：列出当前注册并可暴露给策略层的技能
- `POST /api/intent/detect`：返回意图识别结果、是否使用 fallback，以及每个意图对应的记忆/技能策略
- `POST /chat`：写入短期记忆，按意图构造上下文，按白名单调用 MCP 技能，并通过 LiteLLM 生成回复

`POST /chat` 当前会额外返回：

- `used_skills`：本轮实际执行的技能名列表
- `skill_results`：每个技能的执行结果，便于调试 MCP 调用链路

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

Phase 4：异步归档
- 用 Celery 将历史对话压缩为摘要后异步写入 ChromaDB
- 再接入 MCP/技能调用能力

当前代码已完成 Phase 4 的两项主线能力，并补上了对应测试。
