FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# --- 阶段 1: 安装 uv 和依赖（仅在 pyproject.toml/uv.lock 变化时重建）---
FROM base AS deps

# 传递代理环境变量（构建时生效）
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

WORKDIR /app

# 只复制依赖定义文件，利用层缓存
COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir -U pip uv \
    && uv sync --frozen --no-dev --verbose

# --- 阶段 2: 最终镜像 ---
FROM base AS runtime

# 运行时代理（容器启动后仍可能需要，如调用外部 LLM API）
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

WORKDIR /app

# 从 deps 阶段复制已安装的依赖
COPY --from=deps /app/.venv /app/.venv

# 确保 .venv/bin 在 PATH 中
ENV PATH="/app/.venv/bin:$PATH"

# 代理环境变量（运行时可通过 docker-compose 覆盖）
ENV HTTP_PROXY=${HTTP_PROXY:-}
ENV HTTPS_PROXY=${HTTPS_PROXY:-}
ENV NO_PROXY=${NO_PROXY:-}

# 复制应用代码（代码变化只影响这一层）
COPY app ./app
COPY src ./src
COPY worker.py ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
