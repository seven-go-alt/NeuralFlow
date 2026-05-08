# NeuralFlow RAG Architecture

本文档描述 NeuralFlow 当前版本的 Document RAG Pipeline 设计。

## 1. 目标

把 NeuralFlow 从通用 Agent Runtime 升级为：

> 企业 AI 知识库 Agent 平台

核心要求：

- 文档可上传、可索引、可检索
- Retrieval 能注入到 Agent / Chat Prompt
- 前端可视化 chunk 与 source citations
- 全链路支持 tenant isolation

---

## 2. 分层设计

### Documents Layer
负责：
- 文档 metadata
- 文档状态流转
- 文档列表 / 详情 / 删除 / reindex

### Ingestion Layer
负责：
- 文件解析
- 文本标准化
- chunk splitting
- 触发 embedding 与 indexing

### Embeddings Layer
负责：
- embedding provider abstraction
- OpenAI-compatible embeddings
- embedding cache

### Retrieval Layer
负责：
- ChromaDB query
- top_k / threshold
- metadata filtering
- retrieval result normalization

### RAG Layer
负责：
- context assembly
- citation generation
- token budget control
- prompt injection

---

## 3. 数据模型

### documents
关键字段：
- document_id
- tenant_id
- owner_user_id
- file_type
- storage_path
- checksum_sha256
- status
- chunk_count
- token_count
- error_message

### document_chunks
关键字段：
- chunk_id
- document_id
- tenant_id
- chunk_index
- content
- token_count
- page_number
- section_title
- embedding_model

### Chroma metadata
每个 chunk 写入向量库时会带：
- tenant_id
- document_id
- chunk_id
- title
- filename
- file_type
- page_number
- embedding_model

---

## 4. 文档状态机

```text
uploaded
queued
parsing
chunking
embedding
indexing
ready
failed
deleting
deleted
```

---

## 5. Upload -> Ingestion 数据流

```text
Client Upload
  -> POST /api/documents/upload
  -> save file
  -> create document record
  -> set status=queued
  -> enqueue Celery task

Celery Worker
  -> parsing
  -> chunking
  -> embedding
  -> indexing
  -> ready
```

---

## 6. Query -> Retrieval 数据流

```text
User Query
  -> /chat or /chat/stream
  -> RetrievalService.search()
  -> Chroma query
  -> score normalization
  -> dedup
  -> RAGContextBuilder.build()
  -> prompt injection
  -> LLM response
  -> citations returned to frontend
```

---

## 7. Chunk 策略

当前采用：

- token-aware recursive splitting
- configurable `chunk_size`
- configurable `chunk_overlap`
- 保留：
  - `document_id`
  - `chunk_id`
  - `page_number`
  - `section_title`
  - `source_path`

---

## 8. Retrieval 输出

返回字段：

- `chunk_id`
- `document_id`
- `content`
- `score`
- `metadata`
- `source.title`
- `source.filename`
- `source.page_number`

---

## 9. Prompt 注入策略

NeuralFlow 当前把 RAG 结果作为独立上下文块拼进 prompt：

```text
企业知识库检索上下文（回答时优先参考并尽量给出引用）:
[1] Employee Handbook
...
```

这样能与：
- working memory
- long-term memory
- tool results

共同组成最终上下文。

---

## 10. 前端展示策略

### Documents 页面
- 上传文档
- 文档列表
- 状态查看
- chunk 数量
- 文档详情

### Chat 页面
- runtime panel 中展示 retrieved chunks
- assistant message 下方展示 citations
- 点击 source 跳转文档详情

---

## 11. 当前已完成 vs 后续增强

### 已完成
- document upload
- async ingestion
- parser support: pdf/md/txt/docx
- chunking
- embeddings
- retrieval
- prompt injection
- citations
- UI source visualization

### 可增强
- OCR
- reranker
- Redis embedding cache
- Postgres production profile
- ACL / permissions
- file preview service
