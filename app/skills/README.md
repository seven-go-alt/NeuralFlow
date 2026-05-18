# Skills Module

Tool calling infrastructure: skill registry, terminal execution, and safety guard.

## Components

### `SkillRegistry`
- Central registry of available skills with `SkillDefinition` (name, description, read_only)
- Skills: memory, planner, python_exec, file_read, file_write, file_list, terminal
- `get_allowed_skills(whitelist)` filters by permission

### `execute_command()`
- Async subprocess execution with timeout
- Returns `TerminalResult` (stdout, stderr, return_code, timed_out)
- Handles FileNotFoundError, PermissionError, TimeoutError

### `Guard` (safety layer)
- **`validate_terminal_command()`** — blocks dangerous patterns: `rm -rf`, `sudo`, `mkfs`, fork bombs, unsafe curl/wget pipes
- **`validate_skill_call()`** — permission check + parameter length validation
- **`sanitize_output()`** — truncates oversized output with truncation notice
- **`contains_sensitive_data()`** — detects API keys, private keys, tokens in output
- **`RateLimiter`** — in-memory rate limiting per session (configurable window + max calls)
