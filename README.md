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
- **Hybrid Retrieval（keyword + vector）**
- **Advanced RAG Pipeline（query transformation → corrective loop）**
- **Multimodal RAG（image / table extraction with vision LLM）**
- **RAG Evaluation（LLM-as-judge answer quality scoring）**
- **Pipeline Observability（tracing with DB persistence + API）**
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
2. 文档解析（含图片/表格多模态提取）
3. Chunk 切分
4. Embedding 生成
5. ChromaDB 建索引
6. Query Retrieval（Hybrid keyword + vector search）
7. RAG Context Assembly（含 Advanced RAG 纠错循环）
8. 注入聊天 / Agent Prompt
9. 在前端展示引用来源与检索 chunks
10. LLM-as-judge 回答质量评估与 Pipeline 追踪

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
- 文档解析（PDF / Markdown / TXT / DOCX）
- token-aware recursive chunk splitting
- configurable overlap
- OpenAI-compatible embedding provider
- embedding cache（in-memory + Redis）
- ChromaDB retrieval
- **Hybrid retrieval（keyword + vector）**
- **Advanced RAG pipeline（query transformation → grading → corrective loop → context building）**
- **Multimodal RAG（image description extraction from PDF/DOCX, table extraction → markdown）**
- token-budget-aware context builder
- citations / source support

### RAG Evaluation & Observability
- **LLM-as-judge answer evaluation（relevance / faithfulness / completeness scoring）**
- **Pipeline tracing with DB persistence**
- **Prometheus eval metrics（Gauges / Histograms for answer scores）**
- **Eval API endpoints（run evals on datasets, list results, inspect detail）**
- **Trace API endpoints（filterable trace spans, full trace tree）**

### 前端 Runtime Console
- documents 页面
- chunk 可视化
- runtime panel 检索展示
- 聊天消息下方 source citations 展示
- 上传后自动刷新文档列表
- 文档处理中自动轮询状态
- 文档详情 / 列表页可直接发起单文档总结与 scoped chat

### 工程能力
- SQLAlchemy metadata storage
- Celery ingestion queue
- structured logging
- tenant middleware
- Prometheus metrics
- runtime config hot patch
- pytest regression coverage（**689 tests**）
- **Performance benchmarks（ingestion latency / retrieval latency / token usage）**
- **GitHub Actions CI/CD（auto-merge for Dependabot, eval gate, multi-stage workflows）**

完整变更记录见：

- [`CHANGELOG.md`](CHANGELOG.md)

---

## 本地验证（发布前）

发布前建议先做一轮本地验证：

```bash
bash scripts/verify-local.sh
```

注意：这个脚本使用 `bash` 特性（例如 `set -euo pipefail`），**不要用 `sh scripts/verify-local.sh`**，否则会直接报错。

默认会：
- 使用 `uv` 同步后端依赖
- 运行 `pytest`
- 执行前端 `npm ci` / `lint` / `typecheck` / `build`
- 在国内网络环境默认优先使用清华 PyPI 镜像并提高超时时间

本地验证未通过前，不应推送发布链路相关改动。

---

## 3. 技术栈

### Backend
- FastAPI
- SQLAlchemy
- Redis
- ChromaDB
- Celery
- LiteLLM / OpenAI-compatible API
- httpx（async HTTP client）
- PyMuPDF（含 PDF table extraction）
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
Upload File (PDF / DOCX / MD / TXT)
  -> Document Record (DB)
  -> Async Ingestion Task (Celery)
  -> Parse (text + images + tables for multimodal)
  -> Chunk (token-aware, configurable overlap)
  -> Embed (OpenAI-compatible API, in-memory + Redis cache)
  -> Chroma Index (vectors + metadata)

User Query
  -> [Optional] Query Transformation (rewrite / expand)
  -> Hybrid Retrieval (keyword BM25 + vector search)
  -> Relevance Grading (LLM-as-judge filter)
  -> [Optional] Corrective Loop (rewrite query → re-retrieve)
  -> Context Builder (token-budget-aware assembly)
  -> Agent / Chat Prompt
  -> LLM Response
  -> Answer Evaluation (LLM-as-judge score)
  -> Pipeline Trace (persisted to DB for observability)
  -> Citations / Sources (frontend display)
```

---

## 4.5 推荐演示路径（作品集 / 面试展示）

如果你要把这个项目作为求职作品集，最值得展示的是这条完整链路：

1. 启动 **FastAPI / Celery Worker / Next.js Frontend**
2. 在 `Documents` 页面上传一份 PDF / Markdown / TXT / DOCX（含图片和表格的 PDF 可以同时展示多模态提取）
3. 观察文档状态从 `queued -> parsing -> chunking -> embedding -> indexing -> ready`
4. 在文档列表页或详情页点击：
   - `Summarize this document`
   - `Chat with this document`
5. 系统会创建一个**绑定当前 document_id 的新会话**
6. 在首页 Runtime Console 中继续追问，检索会被限制在该文档范围内
7. 在右侧 runtime panel 查看：
   - retrieval 事件
   - matched chunks
   - source citations

这条流程能够完整展示：

- 异步文档摄入 pipeline（含多模态图片/表格提取）
- Hybrid 检索（keyword + vector）
- Advanced RAG 纠错循环
- 单文档 scoped retrieval
- Agent / Chat runtime observability
- 前后端联动的产品闭环

---

## 5. 当前项目结构

```text
NeuralFlow/
├── app/
│   ├── agents/
│   ├── api/
│   │   ├── documents.py
│   │   ├── eval.py
│   │   ├── retrieval.py
│   │   ├── streaming.py
│   │   └── traces.py
│   ├── core/
│   ├── db/
│   ├── documents/
│   ├── embeddings/
│   ├── evals/
│   ├── ingestion/
│   │   └── multimodal/     # image/table extraction
│   ├── memory/
│   ├── observability/
│   │   ├── trace_manager.py
│   │   ├── trace_persister.py
│   │   └── eval_metrics.py
│   ├── rag/
│   ├── retrieval/
│   └── main.py
├── frontend/
│   ├── app/
│   │   ├── chat/
│   │   └── documents/
│   ├── components/
│   │   ├── documents/
│   │   ├── rag/
│   │   └── ui/
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

> macOS / 本地演示环境建议使用 `--pool=solo`，避免 prefork 在文档摄入时出现子进程崩溃。

```bash
cd ~/github/NeuralFlow
. .venv/bin/activate
python -m celery -A worker.celery_app worker --loglevel=info --pool=solo
```

或用 uv run：

```bash
cd ~/github/NeuralFlow
uv run python -m celery -A worker.celery_app worker --loglevel=info --pool=solo
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

### 8.6 生产部署骨架

仓库已提供：

- `.env.production.example`
- `docker-compose.prod.yml`
- `deploy/Caddyfile`

推荐流程：

```bash
cp .env.production.example .env.production
# 填好域名、数据库、LLM、Embedding 配置

docker compose -f docker-compose.prod.yml up -d --build
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
- `EMBEDDING_API_BASE`（建议与 chat LLM 分开配置）
- `EMBEDDING_API_KEY`
- `rag_default_top_k`
- `rag_score_threshold`

多模态 RAG（可选）：
- `MULTIMODAL_ENABLED`（默认 `false`）
- `VISION_MODEL`（默认 `gpt-4o`）
- `MULTIMODAL_MAX_IMAGES`（默认 20）
- `MULTIMODAL_MAX_TABLES`（默认 50）

生产环境建议：

- 将 `LLM_API_BASE` / `LLM_API_KEY` 与 `EMBEDDING_API_BASE` / `EMBEDDING_API_KEY` 分开配置，避免某些代理只支持 chat 不支持 embeddings。
- 将 `OFFLINE_FALLBACK_ENABLED=false`，避免线上静默降级。
- `CORS_ALLOW_ORIGINS` 设置为明确域名，不要保留 `*`。
- 优先使用 PostgreSQL，而不是默认 SQLite。
- Linux 服务器如果稳定可尝试 `CELERY_WORKER_POOL=prefork`；首版上线建议先用 `solo`。

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

这版项目的亮点不在”接了一个向量库”，而在于它展示了从文档摄入到高级 RAG 再到评估观测的完整工程链路：

- 文档上传与多格式解析
- 多模态提取（图片描述 + 表格→markdown）
- 异步 ingestion pipeline（Celery）
- 向量检索 + Hybrid retrieval（keyword BM25）
- Advanced RAG（query transformation → grading → corrective loop）
- RAG context assembly（token-budget-aware）
- Agent / Chat integration
- LLM-as-judge 回答质量评估
- Pipeline tracing with DB persistence
- 引用来源展示
- 多租户隔离
- streaming observability
- 689 regression tests
- Performance benchmarks
- 自动化 CI/CD（path filtering、auto-merge、eval gate）

如果你要把它写进简历或作品集，最适合强调的关键词是：

- **Enterprise RAG**
- **AI Knowledge Base Platform**
- **Agent Runtime**
- **Function Calling / MCP**
- **Advanced RAG Pipeline（query transformation + corrective loop）**
- **Hybrid Retrieval（keyword + vector）**
- **Multimodal RAG（vision LLM image extraction）**
- **RAG Evaluation（LLM-as-judge）**
- **Pipeline Observability（tracing + metrics）**
- **Multi-tenant Retrieval**
- **Async Ingestion Pipeline**
- **Source-aware Citation UX**

---

## 12. Demo Walkthrough (5-minute portfolio run)

如果你要快速演示这套系统，建议按下面顺序：

### Step 1: 启动服务

在三个独立的终端中分别运行：

**终端 1 - 后端 API：**
```bash
cd ~/github/NeuralFlow
uv run uvicorn app.main:app --reload
```

**终端 2 - Celery Worker（文档处理必需）：**
```bash
cd ~/github/NeuralFlow
. .venv/bin/activate
python -m celery -A worker.celery_app worker --loglevel=info
```

⚠️ **注意：Celery Worker 必须运行，否则上传的文件无法处理**

**终端 3 - 前端：**
```bash
cd ~/github/NeuralFlow/frontend
npm run dev
```

### Step 2: 打开 Documents 页面

浏览器访问：

- `http://localhost:3000/documents`

上传一个示例文件，例如：
- 员工手册 PDF
- 团队规范 Markdown
- 产品需求 TXT
- 培训材料 DOCX

**上传前注意：** 一定要确保 Celery Worker 在后台运行（终端 2），否则文件会卡在 `queued` 状态。

观察点：
- 文档是否出现在列表里
- 状态是否从 `queued` → `parsing` → `chunking` → `embedding` → `indexing` → `ready` 进行
- chunk 数量是否可见

### Step 3: 检查文档详情

点击文档进入详情页，确认：
- 文档 metadata 正常
- chunk 已生成
- page / token / section 信息可见

### Step 4: 从文档直接发起总结或问答

**重要：** 只有当文档状态为 `ready` 时，Chat 才能检索到文档内容。上传后，文件会经过以下处理阶段：

1. `uploaded` - 文件已上传
2. `queued` - 等待 Celery worker 处理
3. `parsing` - 解析文件内容中
4. `chunking` - 分割成 chunks 中
5. `embedding` - 生成向量嵌入中
6. `indexing` - 建立 ChromaDB 索引中
7. `ready` ✓ - **现在可以检索了**

当状态变为 `ready` 后，你现在有两种推荐方式：

#### 方式 A：在 Documents 列表页直接操作
- 点击 `Summarize this document`
- 点击 `Chat with this document`

#### 方式 B：进入文档详情页再操作
- 查看 chunk
- 点击 `Summarize this document`
- 点击 `Chat with this document`

这两个入口都会：

1. 创建一个新的 chat session
2. 把当前 `document_id` 绑定到该 session
3. 回到首页 Runtime Console
4. 后续检索只在这份文档范围内进行

### Step 5: 回到 Chat Console 继续追问

在自动创建的新会话里，你可以继续追问：

- `员工请假制度是什么？`
- `这份规范里对代码评审有什么要求？`
- `把这份文档总结成 5 条重点`
- `根据知识库，总结一下 onboarding 流程。`

如果当前 session 是从文档页发起的，那么这些问题会自动限定在该文档范围内，而不是在整个知识库里做无约束检索。

观察点：
- Runtime Panel 中是否出现 retrieved chunks
- 是否出现 source / score / page 信息
- assistant 回答下方是否展示 citations

### Step 6: 展示流式检索可观测性

使用 `/chat/stream` 路径时，可以重点展示：
- retrieval 事件
- chunk 事件
- streaming tokens
- 最终 answer + sources

这一步很适合在面试或作品集录屏中展示，因为它能直观看出系统不是“黑盒聊天框”，而是可观测的 RAG Agent Runtime。

---

## 13. 测试

运行新增的 RAG 关键测试：

RAG Pipeline 关键测试：

```bash
uv run pytest -q \
  tests/test_rag_chat.py \
  tests/test_rag_context_builder.py \
  tests/test_advanced_pipeline.py \
  tests/test_hybrid_retrieval.py \
  tests/test_documents_api.py \
  tests/test_ingestion_pipeline.py \
  tests/test_retrieval_api.py
```

评估与可观测性测试：

```bash
uv run pytest -q \
  tests/test_answer_evaluator.py \
  tests/test_eval_api.py \
  tests/test_eval_regression.py \
  tests/test_trace_manager.py \
  tests/test_trace_persister.py
```

多模态 RAG 测试：

```bash
uv run pytest -q \
  tests/test_multimodal_extractor.py \
  tests/test_multimodal_config.py \
  tests/test_multimodal_processor.py
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
- [x] embedding provider abstraction（in-memory + Redis cache）
- [x] ChromaDB retrieval
- [x] RAG context builder
- [x] Agent / Chat RAG injection
- [x] 前端 documents 页面
- [x] source citations 展示
- [x] retrieval chunk 可视化
- [x] 异步 ingestion 任务
- [x] 测试覆盖主链路（**689 tests**）
- [x] **Hybrid retrieval（keyword + vector search）**
- [x] **Advanced RAG pipeline（query transformation + correctness grading + corrective loop）**
- [x] **Performance benchmarks（ingestion / retrieval / token usage, HTML reports）**
- [x] **LLM-as-judge answer evaluation（relevance / faithfulness / completeness）**
- [x] **Pipeline tracing with DB persistence + API**
- [x] **Prometheus eval metrics gauges**
- [x] **Multimodal RAG（image description from PDF/DOCX, table→markdown extraction）**
- [x] **CI/CD 自动合入（auto-merge for Dependabot, eval gate）**
- [x] **Security headers（CSP）**
- [x] **Redis-backed embedding cache with connection pooling**
- [x] **Rich metadata filters（tags / owner / document groups, multi-value $in support）**
- [x] **Markdown heading hierarchy parsing & chunking**
- [x] **OCR pipeline（pytesseract + pdf2image for scanned documents）**
- [x] **Cross-encoder reranker（lazy-loaded transformer model）**
- [x] **Signed file preview URLs（HMAC time-limited tokens）**
- [x] **Postgres production migration guide（MIGRATION.md + .env.production.example）**
- [x] **Deployment manifests（K8s + Helm charts）**

## 15. 下一步可继续增强

- **role-based knowledge base ACL** — 文档/知识库按角色粒度的访问控制，需要补全 auth 系统后再做
- **auth / SSO integration** — 用户认证与单点登录，已完成 JWT 基础框架，对接 OAuth/OIDC 待实现
- **evaluation dashboard** — 在 frontend 中加入 LLM-as-judge 评估结果可视化面板（当前仅可通过 API 查看）
- **advanced reranker model** — 替换当前 `MiniLM-L-6-v2` 为更大更强的 cross-encoder 模型（如 `electra-base`），提升排序精度
- **hybrid search index tuning** — BM25 参数（k1 / b）可配置化，适应不同文档类型的最佳检索权重
- **streaming eval monitoring** — 在 chat stream 中实时注入评估评分事件，供前端或监控系统消费

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
