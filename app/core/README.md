# Core Module

LLM interaction, intent routing, token budgeting, and model selection.

## Components

### `LLMClient`
- Primary LLM interface via `litellm`
- Supports `generate()` (single response) and `stream_generate()` (streaming)
- Fallback chain: primary model → offline model → rule-based summary
- Configurable via `Settings` (model, API base, API key)

### `IntentRouter`
- Three-tier intent detection: keyword rules → embedding similarity → LLM classification
- Each intent maps to a policy (`memory_strategy` + `skill_whitelist`)
- Intents: `general`, `query_history`, `coding`, `planning`

### `ModelRouter`
- Intent-based model selection with cost/latency constraints
- `select(intent, cost_max, latency_max)` returns optimal model ID
- `fallback_chain(intent)` returns ordered fallback candidates
- Pre-configured profiles: GPT-4o-mini, GPT-4o, Claude Sonnet 4

### `TokenBudgetManager`
- Manages context window token budgets
- Trim strategy: retain recent messages + high-priority segments
- Configurable soft/hard token limits

### `ContextManager`
- Maintains conversation history with token tracking
- Applies intent-specific memory strategies
