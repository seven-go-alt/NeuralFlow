# OpenSpec for NeuralFlow

This directory holds lightweight product/engineering specs for changes that need a clear contract before implementation.

## Structure

- `specs/<name>/spec.md` — human-readable spec
- `specs/<name>/checklist.md` — implementation checklist / acceptance criteria

## Initial specs

- `eval-v1` — minimum viable evaluation framework for retrieval/chat regression testing
- `rag-citations-contract` — retrieval and citation contract for `/api/retrieval/search` and `/chat`

## Usage

1. Write or update the relevant spec before larger feature changes.
2. Keep acceptance criteria concrete and testable.
3. Link implementation PRs/commits back to the spec folder when possible.

This is intentionally lightweight: enough structure to prevent drift, not enough ceremony to slow the project down.
