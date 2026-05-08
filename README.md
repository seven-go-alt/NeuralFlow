# NeuralFlow

NeuralFlow 是一个面向作品集与真实原型场景的 **企业 AI 知识库 Agent 平台**。

它把以下能力整合到一套可运行系统里：

- FastAPI 后端
- Agent Runtime
- Function Calling
- MCP Tool 集成
- Redis Working Memory
- ChromaDB Vector Retrieval
- Document RAG Pipeline
- Streaming Chat / SSE Runtime Events
- Next.js Runtime Console
- Multi-tenant Isolation
- Celery 异步摄入任务

这不是一个只会聊天的 demo，而是一套围绕 **文档知识库 + Agent + 检索增强生成** 的工程化骨架。

---

## 1. 项目定位

当前版本的 NeuralFlow 目标是：

> 将通用 Agent Runtime 升级为可扩展的企业知识库 AI Agent 平台。

它支持把 PDF、Markdown、TXT、DOCX 文档上传进系统，自动完成：

1. 文件存储
2. 文档解析
3. Chunk 切分
4. Embedding 生成
5. ChromaDB 建索引
6. Query Retrieval
7. RAG Context Assembly
8. 注入聊天 / Agent Prompt
9. 在前端展示引用来源与检索 chunks

---

## 2. 核心能力

### Agent Runtime
- `/chat` 同步对话
- `/chat/stream` SSE 流式响应
- `/chat/react` ReAct / Function Calling agent
- `/chat/orchestrate` 多 Agent 编排

### 文档知识库
- 支持上传：`PDF / Markdown / TXT / DOCX`
- 文档状态管理：`uploaded / queued / parsing / chunking / embedding / indexing / ready / failed`
- 多租户隔离
- 文档详情 / chunk 浏览

### RAG Pipeline
- 文档解析
- token-aware recursive chunk splitting
- configurable overlap
- OpenAI-compatible embedding provider
- embedding cache
- ChromaDB retrieval
- token-budget-aware context builder
- citations / source support

### 前端 Runtime Console
- documents 页面
- chunk 可视化
- runtime panel 检索展示
- 聊天消息下方 source citations 展示
- 上传后自动刷新文档列表

### 工程能力
- SQLAlchemy metadata storage
- Celery ingestion queue
- structured logging
- tenant middleware
- Prometheus metrics
- runtime config hot patch
- pytest regression coverage

完整变更记录见：

- [`CHANGELOG.md`](CHANGELOG.md)

---

## 3. 技术栈

### Backend
- FastAPI
- SQLAlchemy
- Redis
- ChromaDB
- Celery
- LiteLLM / OpenAI-compatible API
- PyMuPDF
- python-docx
- markdown-it-py
- tiktoken

### Frontend
- Next.js 15
- TypeScript
- Zustand
- React Query

---

## 4. 架构概览

完整 RAG 架构说明见：

- [`docs/rag-architecture.md`](docs/rag-architecture.md)

高层数据流：

```text
Upload File
  -> Document Record
  -> Async Ingestion Task
  -> Parse
  -> Chunk
  -> Embed
  -> Chroma Index

User Query
  -> Retrieval
  -> Context Builder
  -> Agent / Chat Prompt
  -> LLM Response
  -> Citations / Sources
```

---

## 5. 当前项目结构

```text
NeuralFlow/
├── app/
│   ├── agents/
│   ├── api/
│   │   ├── documents.py
│   │   ├── retrieval.py
│   │   └── streaming.py
│   ├── core/
│   ├── db/
│   ├── documents/
│   ├── embeddings/
│   ├── ingestion/
│   ├── memory/
│   ├── rag/
│   ├── retrieval/
│   └── main.py
├── frontend/
│   ├── app/
│   │   └── documents/
│   ├── components/
│   │   ├── documents/
│   │   └── rag/
│   ├── features/
│   ├── services/
│   └── types/
├── docs/
│   └── rag-architecture.md
├── tests/
└── worker.py
```

---

## 6. 文档知识库 API

### 上传文档
```http
POST /api/documents/upload
```

multipart/form-data：
- `file`
- `title` (optional)

### 文档列表
```http
GET /api/documents
```

### 文档详情
```http
GET /api/documents/{document_id}
```

### 查看 chunks
```http
GET /api/documents/{document_id}/chunks
```

### 删除文档
```http
DELETE /api/documents/{document_id}
```

### 重新建索引
```http
POST /api/documents/{document_id}/reindex
```

### 检索调试接口
```http
POST /api/retrieval/search
```

请求示例：

```json
{
  "query": "员工请假制度是什么？",
  "top_k": 5,
  "score_threshold": 0.2,
  "filters": {}
}
```

---

## 7. Chat / Agent API

### 同步聊天
```http
POST /chat
```

示例请求：

```json
{
  "session_id": "demo-1",
  "message": "员工请假制度是什么？",
  "use_retrieval": true
}
```

示例响应（节选）：

```json
{
  "session_id": "demo-1",
  "intent": "general",
  "reply": "...",
  "citations": [
    {
      "index": 1,
      "label": "Employee Handbook",
      "document_id": "doc_xxx",
      "chunk_id": "chk_xxx",
      "page_number": 3
    }
  ]
}
```

### 流式聊天
```http
POST /chat/stream
```

SSE events 包含：
- `retrieval`
- `chunk`
- `thinking`
- `message`
- `done`
- `error`

---

## 8. 本地启动

### 8.1 Python

```bash
cd ~/github/NeuralFlow
cp .env.example .env
uv sync
```

### 8.2 启动后端

```bash
uv run uvicorn app.main:app --reload
```

### 8.3 启动 Celery worker

```bash
uv run celery -A worker.celery_app worker --loglevel=info
```

### 8.4 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 8.5 可选：Docker Compose

```bash
docker compose up --build
```

---

## 9. 环境变量

核心变量：

- `REDIS_HOST`
- `REDIS_PORT`
- `REDIS_DB`
- `CHROMA_HOST`
- `CHROMA_PORT`
- `DATABASE_URL`
- `DOCUMENTS_STORAGE_DIR`
- `LLM_API_BASE`
- `LLM_API_KEY`
- `OPENAI_API_KEY`
- `LITELLM_MODEL`
- `MCP_BASE_URL`

RAG 相关：

- `EMBEDDING_MODEL`（代码默认 `text-embedding-3-small`）
- `rag_default_top_k`
- `rag_score_threshold`

---

## 10. 多租户隔离

系统通过 `TenantIsolationMiddleware` 基于 Header 传入租户上下文：

- `X-Tenant-ID`
- `X-Tenant-Scope`
- `X-Tenant-Roles`
- `X-Tenant-Subject`

文档 metadata、chunk metadata、retrieval filter 全链路带 `tenant_id`。

---

## 11. 作品集亮点

这版项目的亮点不在“接了一个向量库”，而在于它展示了完整工程链路：

- 文档上传系统
- 异步 ingestion pipeline
- 企业知识库检索
- RAG context assembly
- Agent integration
- 引用来源展示
- 多租户隔离
- streaming observability
- regression tests

如果你要把它写进简历或作品集，最适合强调的关键词是：

- **Enterprise RAG**
- **AI Knowledge Base Platform**
- **Agent Runtime**
- **Function Calling / MCP**
- **Multi-tenant Retrieval**
- **Async Ingestion Pipeline**
- **Source-aware Citation UX**

---

## 12. Demo Walkthrough (5-minute portfolio run)

如果你要快速演示这套系统，建议按下面顺序：

### Step 1: 启动服务

```bash
uv run uvicorn app.main:app --reload
uv run celery -A worker.celery_app worker --loglevel=info
cd frontend && npm run dev
```

### Step 2: 打开 Documents 页面

浏览器访问：

- `http://localhost:3000/documents`

上传一个示例文件，例如：
- 员工手册 PDF
- 团队规范 Markdown
- 产品需求 TXT
- 培训材料 DOCX

观察点：
- 文档是否出现在列表里
- 状态是否从 `queued` 进入后续阶段
- chunk 数量是否可见

### Step 3: 检查文档详情

点击文档进入详情页，确认：
- 文档 metadata 正常
- chunk 已生成
- page / token / section 信息可见

### Step 4: 回到 Chat Console 提问

示例问题：

- `员工请假制度是什么？`
- `这份规范里对代码评审有什么要求？`
- `根据知识库，总结一下 onboarding 流程。`

观察点：
- Runtime Panel 中是否出现 retrieved chunks
- 是否出现 source / score / page 信息
- assistant 回答下方是否展示 citations

### Step 5: 展示流式检索可观测性

使用 `/chat/stream` 路径时，可以重点展示：
- retrieval 事件
- chunk 事件
- streaming tokens
- 最终 answer + sources

这一步很适合在面试或作品集录屏中展示，因为它能直观看出系统不是“黑盒聊天框”，而是可观测的 RAG Agent Runtime。

---

## 13. 测试

运行新增的 RAG 关键测试：

```bash
uv run pytest -q \
  tests/test_rag_chat.py \
  tests/test_rag_context_builder.py \
  tests/test_documents_api.py \
  tests/test_ingestion_pipeline.py \
  tests/test_retrieval_api.py
```

运行全量测试：

```bash
uv run pytest tests/ -v
```

---

## 14. 已完成的 RAG 升级项

- [x] 文档上传系统
- [x] 文件元数据存储
- [x] 多用户 / 多租户隔离
- [x] PDF / Markdown / TXT / DOCX 解析
- [x] token-aware chunk pipeline
- [x] embedding provider abstraction
- [x] ChromaDB retrieval
- [x] RAG context builder
- [x] Agent / Chat RAG injection
- [x] 前端 documents 页面
- [x] source citations 展示
- [x] retrieval chunk 可视化
- [x] 异步 ingestion 任务
- [x] 测试覆盖主链路

---

## 15. 下一步可继续增强

- Redis-backed embedding cache
- richer metadata filter（tags / owner / document groups）
- markdown heading hierarchy parsing
- OCR pipeline
- reranker / hybrid retrieval
- signed file preview URLs
- Postgres production profile
- role-based knowledge base ACL

---

## 16. License / Notes

当前仓库更适合作为：

- AI 应用架构作品集
- 企业知识库 Agent 原型
- RAG 平台工程模板

如果你要进一步产品化，建议下一步优先补：

1. Postgres migration
2. object storage abstraction
3. auth / ACL
4. observability dashboard
5. deployment manifests
