# Docker 代理配置指南

## 1. Docker Daemon 代理（构建时拉取基础镜像）

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

## 2. 构建时传代理（已配置在 Dockerfile 和 deploy.sh 中）

```bash
# 方式一：使用部署脚本
./deploy.sh --proxy http://your-proxy:port

# 方式二：手动构建
docker build --build-arg HTTP_PROXY=http://proxy:port \
             --build-arg HTTPS_PROXY=http://proxy:port \
             -t neuralflow-api:latest .
```

## 3. 运行时代理（已配置在 docker-compose.prod.yml 中）

在 `.env.production` 中设置：

```bash
HTTP_PROXY=http://your-proxy:port
HTTPS_PROXY=http://your-proxy:port
NO_PROXY=localhost,127.0.0.1,redis,chroma,postgres,api,worker,frontend
```

## 4. 验证代理是否生效

```bash
# 检查容器内代理
docker exec neuralflow-api-1 env | grep -i proxy

# 测试网络连通性
docker exec neuralflow-api-1 curl -I https://api.openai.com
```

## 常见问题

- **构建时超时**：确保 Docker daemon 代理已配置（步骤 1）
- **容器内无法访问外部 API**：确保运行时代理已传递（步骤 3）
- **容器间通信失败**：确保 NO_PROXY 包含所有内部服务名
