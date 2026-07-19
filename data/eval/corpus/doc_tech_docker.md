# Meridian Analytics — Docker Multi-Stage Build for Python with Native Extensions

**Document ID:** doc_tech_docker
**Owner:** Platform Engineering
**Last updated:** 2026-07-16

## Overview

Meridian Analytics deploys several Python microservices that depend on native extension modules compiled from C and Cython sources. This document defines the **Docker multi-stage build** structure used to build these **native extensions** during the **build phase** while keeping the final runtime image lean.

## Build Architecture

The **Docker multi-stage build** is structured into three stages to support **compilation** of native dependencies:

```dockerfile
# Stage 1: Builder — compiles native extensions
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN CFLAGS="-O2 -march=x86-64-v3" \
    python setup.py build_ext --inplace

# Stage 2: Runtime — minimal Python image
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/src /app/src
COPY --from=builder /build/*.so /app/

ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The key design principle is that the **compilation** toolchain (gcc, g++, development headers) exists only in the builder stage and is not present in the final runtime image. This reduces the attack surface by approximately 70 packages.

## Stage 1 — Builder Stage

The builder stage handles the complete **build phase** for **native extensions**:

1. **System dependencies:** Installs gcc, g++, make, and development libraries required for **compilation** of packages like `cryptography`, `psutil`, `lxml`, and Meridian's custom Cython modules.
2. **Python dependencies:** Installs all packages from `requirements.txt` using `pip install --user` to isolate them in `/root/.local`.
3. **Custom **compilation**: Runs `python setup.py build_ext` with architecture-specific optimization flags to produce `.so` shared objects.

The builder stage uses `python:3.12-slim` rather than `python:3.12-alpine` because **compilation** of **native extensions** against musl libc frequently produces compatibility issues. The slim Debian variant provides a standard glibc environment.

## Stage 2 — Runtime Stage

The runtime stage is kept minimal by only installing runtime libraries needed by the compiled shared objects. For Meridian's processing service, the only additional system package is `libgomp1` (OpenMP runtime for parallel Cython extensions). The resulting image size is reduced from approximately 1.2 GB (single-stage build) to 290 MB.

## Build Phase Optimization

During the **build phase**, Meridian uses Docker BuildKit caching mount to accelerate repeated **compilation** of **native extensions**:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

RUN --mount=type=cache,target=/build/build \
    python setup.py build_ext --inplace
```

This ensures that pip downloads and intermediate **compilation** artifacts are cached across builds, reducing CI build time from 12 minutes to approximately 3 minutes.

## CI Integration

Meridian's CI pipeline runs the **Docker multi-stage build** on every pull request. The builder stage cache is stored in a GitHub Actions cache with a key based on `requirements.txt` hash. If the dependencies have not changed, the builder stage uses the cached layer and only rebuilds the runtime stage, which completes in under 30 seconds.

## Revision History

This guide was last updated on 16 July 2026 following the migration to BuildKit cache mounts for Cython compilation.
