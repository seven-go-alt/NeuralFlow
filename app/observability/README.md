# Observability Module

Trace span management and LLM cost tracking.

## Components

### `TraceManager`
- Nested span tracking with parent-child relationships
- Context manager API: `with mgr.span("name") as span:`
- Export to dict via `to_dict()` for structured logging

```python
mgr = TraceManager("request")
with mgr.span("retrieval"):
    results = await retrieve()
with mgr.span("llm_call"):
    answer = await llm.generate(prompt)
mgr.close()
print(mgr.to_dict())
```

### `CostTracker`
- Records LLM token usage per model per operation
- Estimates cost using configurable per-model pricing
- Supports GPT-4o, GPT-4o-mini, Claude Sonnet 4, Claude Haiku out of the box
- `summary()` returns breakdown by model

```python
tracker = CostTracker()
tracker.record("gpt-4o", input_tokens=500, output_tokens=200)
print(tracker.total_estimated_cost)
```

## Integration

These modules integrate with the existing Prometheus metrics in `app/utils/observability.py` and structured logging in `app/middleware/telemetry.py`.
