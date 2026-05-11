# NeuralFlow 生产部署（国内单机）

推荐：**单台 Linux 服务器 + Docker Compose + Caddy + Redis + Chroma + PostgreSQL**

## 1. 服务器准备

建议配置：
- 最低：2C4G / 50G SSD
- 推荐：4C8G / 80G SSD

系统建议：
- Ubuntu 22.04 / Debian 12

安装：
- Docker
- Docker Compose Plugin
- Git

## 2. 拉代码

```bash
git clone <your-repo-url>
cd NeuralFlow
```

## 3. 准备生产配置

```bash
cp .env.production.example .env.production
```

重点填写：
- `PUBLIC_BASE_URL=https://your-domain.com`
- `CORS_ALLOW_ORIGINS=https://your-domain.com`
- `DATABASE_URL=postgresql+psycopg://neuralflow:YOUR_PASSWORD@postgres:5432/neuralflow`
- `LLM_API_BASE`
- `LLM_API_KEY`
- `EMBEDDING_API_BASE`
- `EMBEDDING_API_KEY`
- `OFFLINE_FALLBACK_ENABLED=false`

如果 Chat 和 Embedding 用同一个服务，也建议保留独立字段，便于后续切换。

## 4. 配域名

编辑：
- `deploy/Caddyfile`

把：
- `your-domain.com`

改成你的真实域名。

## 5. 首次启动

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

查看状态：

```bash
docker compose -f docker-compose.prod.yml ps
```

查看日志：

```bash
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f worker
docker compose -f docker-compose.prod.yml logs -f frontend
docker compose -f docker-compose.prod.yml logs -f caddy
```

## 6. 验证

### 健康检查

```bash
curl http://127.0.0.1:8000/healthz
```

### 页面检查

浏览器打开：

- `https://your-domain.com`

### RAG 验证

1. 上传 TXT / PDF / DOCX
2. 看文档状态是否到 `ready`
3. 用首页聊天询问文档内容
4. 检查是否出现 citations / retrieval 事件

## 7. 更新部署

```bash
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

## 8. 回滚

如果你已经有历史 commit：

```bash
git log --oneline
git checkout <old-commit>
docker compose -f docker-compose.prod.yml up -d --build
```

## 9. 防火墙建议

只放行：
- 22
- 80
- 443

不要把这些直接暴露公网：
- 5432
- 6379
- 8000
- 8001

## 10. 上线建议

- 首版保持 `CELERY_WORKER_POOL=solo`
- `OFFLINE_FALLBACK_ENABLED=false`
- 先确认 Embedding 源真的支持 `/v1/embeddings`
- 生产优先 PostgreSQL，不建议长期用 SQLite
