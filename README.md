# NeuralFlow

NeuralFlow 是一个基于 FastAPI、Redis、ChromaDB 和 Celery 的 Agent 工程模板，聚焦三件事：

- L0 短期记忆：Redis 滑动窗口保存最近 10 轮对话
- L2 长期记忆：ChromaDB 存储压缩摘要与向量检索结果
- Token 优化：按意图决定是否查询长期记忆

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
uv sync
uv run pytest -q
uv run uvicorn app.main:app --reload
```

`.python-version` 已固定为 `3.11`，适合本地直接用 uv 管理解释器版本。

## 环境变量

复制模板：

```bash
cp .env.example .env
```

至少配置：

- `OPENAI_API_KEY` 或兼容 LiteLLM 的模型凭证
- 如需远端 Redis/Chroma，可修改对应 host/port

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
- `POST /chat`：写入短期记忆，按意图构造上下文，并通过 LiteLLM 生成回复
