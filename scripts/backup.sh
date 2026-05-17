#!/bin/sh
set -e

BACKUP_DIR="${BACKUP_DIR:-./backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-neuralflow-postgres-1}"
POSTGRES_DB="${POSTGRES_DB:-neuralflow}"
POSTGRES_USER="${POSTGRES_USER:-neuralflow}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD}"

mkdir -p "$BACKUP_DIR"

# 备份 PostgreSQL
if docker ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER}$"; then
  echo "Backing up PostgreSQL..."
  docker exec "$POSTGRES_CONTAINER" pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" \
    | gzip > "$BACKUP_DIR/postgres_${TIMESTAMP}.sql.gz"
  echo "  -> $BACKUP_DIR/postgres_${TIMESTAMP}.sql.gz ($(wc -c < "$BACKUP_DIR/postgres_${TIMESTAMP}.sql.gz") bytes)"
else
  echo "WARNING: container $POSTGRES_CONTAINER not running, skipping PostgreSQL backup"
fi

# 清理过期备份
find "$BACKUP_DIR" -name "postgres_*.sql.gz" -mtime "+${RETENTION_DAYS}" -delete

echo "Done. Backups older than ${RETENTION_DAYS} days cleaned up."
