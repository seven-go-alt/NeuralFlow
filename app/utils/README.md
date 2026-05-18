# Utils Module

Shared utilities: observability metrics, caching, retry, and configuration.

## Components

### `observability.py`
- Prometheus metrics: request duration, LLM token usage, memory cache hits, active sessions, error count
- JSON structured logging with `JsonLogFormatter`
- `create_observability()` singleton factory
- `set_log_context()` / `get_log_context()` for session/trace/intent propagation

### `cache.py`
- `TTLCache`: in-memory cache with TTL expiry and LRU eviction
- `CacheManager`: multi-namespace cache orchestration
- `build_key()` — deterministic key generation from parts

### `retry.py`
- `retry(fn, max_attempts, base_delay, backoff_factor)` — async retry with exponential backoff
- `CircuitBreaker`: failure threshold tracking with recovery timeout
  - States: CLOSED → OPEN → HALF_OPEN → CLOSED
  - `call()` for sync, `acall()` for async functions
