#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# NeuralFlow 生产部署脚本
# 用法: ./deploy.sh [--no-cache] [--proxy http://host:port]
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NO_CACHE=""
PROXY=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --proxy)
            PROXY="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: ./deploy.sh [--no-cache] [--proxy http://host:port]"
            exit 1
            ;;
    esac
done

# 如果没指定代理，尝试从环境变量读取
if [[ -z "$PROXY" ]]; then
    PROXY="${HTTP_PROXY:-${http_proxy:-}}"
fi

PROXY_ARGS=""
if [[ -n "$PROXY" ]]; then
    echo "使用代理: $PROXY"
    PROXY_ARGS="--build-arg HTTP_PROXY=$PROXY --build-arg HTTPS_PROXY=$PROXY --build-arg NO_PROXY=localhost,127.0.0.1,redis,chroma,postgres,api,worker,frontend"
fi

echo "=== 构建后端镜像 ==="
docker build $NO_CACHE $PROXY_ARGS -t neuralflow-api:latest .

echo "=== 构建前端镜像 ==="
docker build $NO_CACHE $PROXY_ARGS -t neuralflow-frontend:latest ./frontend

echo "=== 启动服务 ==="
docker compose -f docker-compose.prod.yml up -d

echo "=== 等待服务就绪 ==="
echo "检查服务健康状态..."
for i in {1..30}; do
    if docker compose -f docker-compose.prod.yml ps | grep -q "healthy"; then
        echo "服务已就绪!"
        break
    fi
    echo "等待中... ($i/30)"
    sleep 2
done

echo "=== 部署完成 ==="
docker compose -f docker-compose.prod.yml ps
