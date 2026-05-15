# Eval v1 checklist

## Planning
- [ ] Split existing eval cases into retrieval and chat suites
- [ ] Define case JSON schema and document required fields
- [ ] Define report JSON schema

## Implementation
- [ ] Implement `scripts/eval/runner.py`
- [ ] Implement retrieval suite execution against `/api/retrieval/search`
- [ ] Implement chat suite execution against `/chat`
- [ ] Reuse existing keyword scoring helpers where practical
- [ ] Emit per-case metrics and summary metrics
- [ ] Implement `scripts/eval/compare.py`
- [ ] Add optional baseline threshold flags

## Optional debug support
- [ ] Decide whether `/chat` exposes optional token-budget debug block
- [ ] If enabled, capture trim-related metrics in eval output

## Validation
- [ ] Add sample baseline files
- [ ] Run suite locally against seeded/dev environment
- [ ] Document how to run eval locally
- [ ] Optionally wire into CI as non-blocking first, then gating
