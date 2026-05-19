# NeuralFlow Architecture

## Overview

NeuralFlow is an enterprise AI Knowledge Base Agent Platform that combines document ingestion, vector search, RAG pipelines, multi-agent orchestration, and LLM-based evaluation into a unified system.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (Next.js)                   │
│  Chat · Documents · Agent Console · Eval Dashboard      │
└──────────────┬──────────────────────────────────────────┘
               │ HTTP / SSE
┌──────────────▼──────────────────────────────────────────┐
│                   API Layer (FastAPI)                    │
│  /chat · /chat/stream · /chat/react · /chat/orchestrate │
│  /documents · /retrieval/search · /eval · /traces       │
│  /healthz · /metrics · /admin/config                    │
│  /auth · /api/v1/skills · /api/v1/models                │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────┐  ┌──────────────┐  ┌────────┐
│  Agents  │  │     RAG      │  │ Auth   │
│ ┌──────┐ │  │ ┌──────────┐ │  │ &      │
│ │ReAct │ │  │ │Advanced  │ │  │Tenant │
│ │Agent │ │  │ │Pipeline  │ │  │Isolat.│
│ ├──────┤ │  │ ├──────────┤ │  └────────┘
│ │Orch. │ │  │ │Hybrid    │ │
│ └──────┘ │  │ │Retrieval │ │
└──────────┘  │ ├──────────┤ │
              │ │Grading   │ │
              │ │ & Eval   │ │
              │ └──────────┘ │
              └──────────────┘
```

## Core Components

### 1. API Layer (`app/main.py`)

FastAPI application with middleware for CORS, tenant isolation, telemetry, and structured logging. Routes are organized into separate routers:

- **Chat**: `/chat`, `/chat/stream`, `/chat/react`, `/chat/orchestrate`
- **Documents**: CRUD + upload + reindex
- **Retrieval**: `/retrieval/search`
- **Evaluation**: `/eval/run`, `/eval/runs`
- **Traces**: `/traces`
- **System**: `/healthz`, `/metrics`, `/admin/config`

### 2. Ingestion Pipeline (`app/ingestion/`)

Transforms raw documents into searchable chunks:

```text
Upload → Parse → [Multimodal Extraction] → Chunk → Embed → Index (ChromaDB)
```

- **Parser**: Supports PDF, DOCX, MD, TXT via `ParserFactory`
- **Multimodal**: `ImageExtractor` + `TableExtractor` + `VisionDescriber` for images/tables
- **Chunking**: Token-aware recursive splitting with configurable size/overlap
- **Embedding**: OpenAI-compatible providers with cache layer
- **Orchestration**: Celery worker for async ingestion

### 3. RAG Pipeline (`app/rag/`)

Multi-stage retrieval-augmented generation:

- **AdvancedRAGPipeline**: Query transformation (multi-query, HyDE), retrieval grading, corrective loop, context assembly
- **HybridRetrievalService**: Keyword + vector search hybrid
- **RetrievalGrader**: LLM-as-judge for retrieved chunk relevance
- **ContextBuilder**: Token-budget-aware context assembly with citations
- **AnswerEvaluator**: LLM-as-judge scoring on relevance, faithfulness, completeness

### 4. Agent System (`app/agents/`)

Three agent interfaces:

- **ReAct Agent**: Multi-step thinking + tool calling loop
- **Agent Orchestrator**: Classifies queries and routes to Coder/Planner/General specialists
- **Skills/MCP**: Plugin-based tool system via Model Context Protocol

### 5. Observability (`app/observability/`)

- **TraceManager**: Nested span tracking with ContextVar propagation
- **TracePersister**: Span tree persistence to SQLite/PostgreSQL
- **Prometheus Metrics**: Request duration, LLM token usage, eval scores
- **Structured Logging**: JSON-format audit logging

### 6. Evaluation (`app/evals/`)

- **run_eval()**: Dataset-driven eval loop with configurable metrics
- **EvalMetrics**: Retrieval hit rate, citation accuracy, keyword coverage, answer quality
- **LLM-as-Judge**: Grading retrieval and answer quality

### 7. Memory (`app/memory/`)

- **WorkingMemory**: Session-scoped short-term memory with token budget control
- **Long-term memory**: History retrieval via vector store

## Data Flow

### Chat with RAG

```text
User Message
  → IntentRouter.detect()
  → Skill execution (MCP tools)
  → RetrievalService.search() / AdvancedRAGPipeline.execute()
  → ContextBuilder.build_prompt()
  → LLM.generate()
  → Response + citations
```

### Document Upload

```text
POST /documents/upload
  → Save file → Create document record (status: queued)
  → Celery task: parse → [multimodal extract] → chunk → embed → index
  → Status: ready
```

## Deployment Architecture

```text
┌────────────┐  ┌────────────┐  ┌────────────┐
│   Nginx /   │  │   API      │  │   Worker   │
│   Caddy     │──▶│  (gunicorn)│  │ (Celery)   │
└────────────┘  └─────┬──────┘  └──────┬─────┘
                      │                │
              ┌───────▼──────┐  ┌──────▼──────┐
              │   Redis      │  │  ChromaDB   │
              │ (cache+broker)│  │ (vector db) │
              └──────────────┘  └─────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector DB | ChromaDB | Embedded, simple ops, good enough for single-node |
| LLM Gateway | litellm | Universal adapter, vision support, streaming |
| Async | asyncio | I/O-bound workload (LLM calls, embedding, DB) |
| Task Queue | Celery + Redis | Reliable async ingestion, retry, monitoring |
| ORM | SQLAlchemy 2.0 | Mature, async capable, multi-DB support |
| Auth | JWT + Tenant isolation | Multi-tenant ready from day one |

## Configuration

All configuration lives in `app/config.py` via `pydantic-settings`, loaded from `.env` at startup. Key groups:

- **App**: `app_name`, `app_env`, `cors_allow_origins`
- **Database**: `database_url`, `db_pool_size`
- **Redis**: `redis_host`, `redis_port`, `redis_db`
- **LLM**: `litellm_model`, `llm_api_base`, `vision_model`
- **RAG**: `rag_advanced_enabled`, `rag_default_top_k`, `rag_score_threshold`
- **Multimodal**: `multimodal_enabled`, `multimodal_max_images`, `vision_prompt_template`
- **Timeouts**: `llm_request_timeout_seconds`, `chroma_request_timeout_seconds`
- **MCP**: `mcp_base_url`, `mcp_timeout_seconds`
