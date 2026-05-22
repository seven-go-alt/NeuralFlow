# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- 阶段 1: 安装 uv 和依赖（仅在 pyproject.toml/uv.lock 变化时重建）---
FROM base AS deps

# 传递代理和包源配置（构建时生效）
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ARG UV_INDEX_URL
ARG UV_HTTP_TIMEOUT=180

ENV HTTP_PROXY=${HTTP_PROXY:-} \
    HTTPS_PROXY=${HTTPS_PROXY:-} \
    NO_PROXY=${NO_PROXY:-} \
    PIP_INDEX_URL=${PIP_INDEX_URL:-} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-} \
    UV_INDEX_URL=${UV_INDEX_URL:-} \
    UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制依赖定义和最小打包元数据，避免在安装依赖时因缺少本地包源码而构建失败
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir -U pip uv \
    && uv sync --frozen --no-dev --verbose \
    && find /app/.venv -name "*.pyc" -delete \
    && find /app/.venv -type d -name "__pycache__" -delete

# --- 阶段 2: 最终镜像 ---
FROM base AS runtime

# 运行时代理（容器启动后仍可能需要，如调用外部 LLM API）
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY:-} \
    HTTPS_PROXY=${HTTPS_PROXY:-} \
    NO_PROXY=${NO_PROXY:-}

WORKDIR /app

# 从 deps 阶段复制已安装的依赖
COPY --from=deps /app/.venv /app/.venv

# 确保 .venv/bin 在 PATH 中
ENV PATH="/app/.venv/bin:$PATH"

# 复制应用代码（代码变化只影响这一层）
COPY app ./app
COPY src ./src
COPY worker.py ./
COPY docker-entrypoint.sh /docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["gunicorn", "-k", "uvicorn.UvicornWorker", "-w", "4", "--bind", "0.0.0.0:8000", "app.main:app"]
