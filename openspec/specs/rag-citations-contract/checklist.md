# RAG + Citations contract checklist

## Contract definition
- [ ] Confirm current `/api/retrieval/search` request fields
- [ ] Confirm current `/api/retrieval/search` response fields
- [ ] Confirm current `/chat` citation shape
- [ ] Define minimum required citation fields
- [ ] Define no-result behavior expectations

## Code/test alignment
- [ ] Align backend tests with the documented contract
- [ ] Align frontend consumers with the documented contract
- [ ] Add or update eval cases that assert citation fields and retrieval behavior

## Optional debug support
- [ ] Decide whether to expose `debug.token_budget`
- [ ] Ensure debug is optional and non-breaking
