# Docker 代理配置指南

适用场景：
- 服务器需要代理才能拉 Docker 基础镜像
- Docker 构建阶段里的 `npm ci` / `uv sync` / `pip install` 需要代理
- 运行中的 API / worker 访问外部模型服务需要代理

NeuralFlow 现在把代理分成了三层，别混着看：

1. **Docker daemon 代理**：解决 `FROM node:22-alpine` / `FROM python:...` 这类基础镜像拉取
2. **构建期 build args**：解决 Dockerfile 内部 `npm ci`、`uv sync` 等联网步骤
3. **运行时环境变量**：解决容器启动后访问 OpenAI / embedding / 其他外部 API

## 1. Docker daemon 代理（拉基础镜像）

创建或编辑 `/etc/systemd/system/docker.service.d/proxy.conf`：

```ini
[Service]
Environment="HTTP_PROXY=http://your-proxy:port"
Environment="HTTPS_PROXY=http://your-proxy:port"
Environment="NO_PROXY=localhost,127.0.0.1"
```

然后重启 Docker：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

如果这层没配，常见症状是：
- `docker build` 卡在拉基础镜像
- `docker pull` 很慢或直接超时

## 2. 构建期代理（Dockerfile 内联网步骤）

NeuralFlow 已经支持：
- 根目录 `Dockerfile`
- `frontend/Dockerfile`
- `deploy.sh`
- GitHub Actions 的 API / frontend 镜像构建

### 方式一：使用部署脚本

```bash
./deploy.sh --proxy http://your-proxy:port
```

脚本会自动把下面这些参数传入构建：
- `HTTP_PROXY`
- `HTTPS_PROXY`
- `NO_PROXY`

### 方式二：手动构建 API 镜像

```bash
docker build \
  --build-arg HTTP_PROXY=http://proxy:port \
  --build-arg HTTPS_PROXY=http://proxy:port \
  --build-arg NO_PROXY=localhost,127.0.0.1,redis,chroma,postgres,api,worker,frontend \
  -t neuralflow-api:latest .
```

### 方式三：手动构建 frontend 镜像

```bash
docker build \
  -f frontend/Dockerfile \
  frontend \
  --build-arg HTTP_PROXY=http://proxy:port \
  --build-arg HTTPS_PROXY=http://proxy:port \
  --build-arg NO_PROXY=localhost,127.0.0.1,redis,chroma,postgres,api,worker,frontend \
  -t neuralflow-frontend:latest
```

### 当前实现细节

- `frontend/.dockerignore` 已排除 `node_modules`、`.next` 等大目录，避免构建上下文过大
- `frontend/Dockerfile` 只会在传入代理变量时才写 npm proxy 配置，不再无脑写空值
- GitHub Actions 的 frontend build 已补上 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`

如果这层没配，常见症状是：
- 基础镜像能拉下来，但卡在 `npm ci`
- GitHub Actions 里 API 镜像能过，frontend 镜像超时
- 宿主机 `npm ci` 正常，但 Docker build 里的 `npm ci` 卡住

## 3. 运行时代理（容器访问外部 API）

`docker-compose.prod.yml` 已经给 API / worker 透传代理环境变量。

在 `.env.production` 中设置：

```bash
HTTP_PROXY=http://your-proxy:port
HTTPS_PROXY=http://your-proxy:port
NO_PROXY=localhost,127.0.0.1,redis,chroma,postgres,api,worker,frontend
```

建议 `NO_PROXY` 至少包含：
- `localhost`
- `127.0.0.1`
- `redis`
- `chroma`
- `postgres`
- `api`
- `worker`
- `frontend`

如果这层没配，常见症状是：
- 容器能启动，但调用模型 API 超时
- worker 无法访问 embedding / LLM 服务
- 容器间本地流量被错误送进代理

## 4. CI / GitHub Actions 代理说明

GitHub Actions 的镜像构建支持以下 secrets：

- `HTTP_PROXY`
- `HTTPS_PROXY`
- `NO_PROXY`

目前：
- API 镜像构建会传这些 build args
- frontend 镜像构建也会传这些 build args

如果只给 API 配了代理、没给 frontend 配，前端镜像构建时还是会卡。这个坑已经补上了。

## 5. 验证代理是否生效

### 验证 Docker daemon

```bash
docker info | grep -i proxy
```

### 验证构建阶段

```bash
docker build \
  -f frontend/Dockerfile \
  frontend \
  --build-arg HTTP_PROXY=http://proxy:port \
  --build-arg HTTPS_PROXY=http://proxy:port \
  --progress=plain
```

### 验证运行时容器

```bash
# 检查容器内代理
docker compose -f docker-compose.prod.yml exec api env | grep -i proxy

# 测试网络连通性
docker compose -f docker-compose.prod.yml exec api curl -I https://api.openai.com
```

## 6. 排障顺序

建议按这个顺序查，别一上来就怀疑代码：

1. **先看 Docker daemon 代理**
   - 基础镜像都拉不下来，别折腾 Dockerfile
2. **再看 build args 有没有传进去**
   - 特别是 frontend build
3. **最后看运行时 `.env.production`**
   - 这是容器启动后的事情，不解决构建期问题

## 常见问题

- **构建时超时**：先检查 Docker daemon 代理，再检查 build args
- **frontend 镜像单独卡住**：优先检查 CI/脚本是否给 frontend 也传了代理参数
- **容器内无法访问外部 API**：确保运行时代理已传递
- **容器间通信失败**：确保 `NO_PROXY` 包含所有内部服务名
- **Docker build 很慢但不是代理问题**：检查 `.dockerignore`，避免把 `node_modules` / `.next` 打进上下文
