#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7890}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,redis,chroma,postgres,api,worker,frontend}"

printf '%s\n' '==> verifying local toolchain'
python3 --version
uv --version
node --version
npm --version
printf '%s\n' '==> effective package/network settings'
printf 'UV_INDEX_URL=%s\n' "$UV_INDEX_URL"
printf 'UV_HTTP_TIMEOUT=%s\n' "$UV_HTTP_TIMEOUT"
printf 'UV_CONCURRENT_DOWNLOADS=%s\n' "$UV_CONCURRENT_DOWNLOADS"
printf 'HTTP_PROXY=%s\n' "$HTTP_PROXY"
printf 'HTTPS_PROXY=%s\n' "$HTTPS_PROXY"
printf 'ALL_PROXY=%s\n' "$ALL_PROXY"
printf 'NO_PROXY=%s\n' "$NO_PROXY"

printf '%s\n' '==> cleaning stale uv locks'
pkill -f 'uv sync --frozen --no-dev' 2>/dev/null || true
pkill -f 'uv run pytest' 2>/dev/null || true
rm -f "$ROOT/.venv/.lock"

printf '%s\n' '==> syncing backend dependencies'
uv sync --frozen --no-dev

printf '%s\n' '==> running backend tests'
uv run pytest

printf '%s\n' '==> installing frontend dependencies'
cd "$ROOT/frontend"
npm ci

printf '%s\n' '==> running frontend lint'
npm run lint

printf '%s\n' '==> running frontend typecheck'
npm run typecheck

printf '%s\n' '==> building frontend'
npm run build

printf '%s\n' '==> local verification passed'
