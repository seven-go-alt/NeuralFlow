# Migration Guide

## SQLite → PostgreSQL

NeuralFlow uses SQLite by default for local development and PostgreSQL for production.

### How it works

The app auto-detects the database backend from the `DATABASE_URL`:

- `sqlite:///...` → SQLite (development)
- `postgresql+psycopg://...` → PostgreSQL (production)

Schema is created automatically on startup via `init_db()` (`app/db/session.py`), which calls
`Base.metadata.create_all()`. No manual migration step is needed.

### Migrating existing data

If you have a SQLite database with production data and need to move to PostgreSQL:

1. Dump SQLite data:

   ```bash
   sqlite3 data/neuralflow.db .dump > dump.sql
   ```

2. Edit `dump.sql` — remove SQLite-specific directives (`PRAGMA`, `BEGIN TRANSACTION`/`COMMIT` wrappers)
   and adjust any type mappings (e.g., `BOOLEAN` → `SMALLINT`, `DATETIME` → `TIMESTAMP`).

3. Import into PostgreSQL:

   ```bash
   psql $DATABASE_URL < dump.sql
   ```

### Production PostgreSQL tuning

The following environment variables control the SQLAlchemy connection pool:

| Variable | Default | Description |
|---|---|---|
| `DB_POOL_SIZE` | `5` | Number of connections to keep in the pool |
| `DB_MAX_OVERFLOW` | `10` | Max connections beyond `pool_size` under load |

Set these in `.env.production` or your deployment environment.

### Adding Alembic (future)

If the schema grows complex enough to need versioned migrations, add Alembic:

```bash
uv add alembic
alembic init alembic
```

Point `alembic.ini` `sqlalchemy.url` at `DATABASE_URL` and set `target_metadata` in
`alembic/env.py`:

```python
from app.db.base import Base
target_metadata = Base.metadata
```
