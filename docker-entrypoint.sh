#!/bin/sh
set -e

# 验证生产必需的环境变量
required_vars="POSTGRES_PASSWORD REDIS_HOST CHROMA_HOST"
missing=""

for var in $required_vars; do
  eval "value=\${$var:-}"
  if [ -z "$value" ]; then
    missing="$missing $var"
  fi
done

if [ -n "$missing" ]; then
  echo "ERROR: missing required environment variable(s):$missing" >&2
  echo "Check your .env.production file or docker-compose environment." >&2
  exit 1
fi

exec "$@"
