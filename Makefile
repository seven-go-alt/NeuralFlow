.PHONY: install test lint typecheck format clean run-frontend run-api dev db-setup

# Python
install:
	uv sync --group dev --frozen

test:
	uv run pytest -q

lint:
	uv run ruff check .

typecheck:
	uv run mypy app tests worker.py

format:
	uv run pre-commit run ruff-format --all-files

format-check:
	uv run ruff format --check .

clean:
	rm -rf .venv/ __pycache__/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Python API
run-api:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
run-frontend:
	cd frontend && npm run dev

# Both
dev:
	@echo "Starting API (port 8000) and Frontend (port 3000)..."
	@trap 'kill 0' EXIT; \
		$(MAKE) run-api & \
		$(MAKE) run-frontend & \
		wait

db-setup:
	uv run python -c "from app.db.session import init_db; init_db()"

# Docker
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Eval
eval-run:
	uv run python -m app.evals.cli run data/eval/datasets/rag_quality_50.jsonl --top-k 5

# Coverage
coverage:
	uv run pytest --cov=app --cov-report=term --cov-report=html
	@echo "HTML report: htmlcov/index.html"
