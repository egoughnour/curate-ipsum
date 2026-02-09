.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Development ───────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install project + dev dependencies via uv
	uv sync --all-extras

.PHONY: lock
lock: ## Regenerate uv.lock
	uv lock

.PHONY: update
update: ## Update all dependencies and regenerate lock
	uv lock --upgrade
	uv sync --all-extras

# ── Quality ───────────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter
	uv run ruff check .

.PHONY: fmt
fmt: ## Auto-format with ruff
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: typecheck
typecheck: ## Run mypy type checking
	uv run mypy --package curate_ipsum || uv run mypy .

.PHONY: check
check: lint typecheck ## Run all static checks

# ── Testing ───────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run fast test suite (no integration tests)
	uv run pytest -m "not integration" -v

.PHONY: test-all
test-all: ## Run all tests including integration
	uv run pytest -v

.PHONY: test-integration
test-integration: ## Run only integration tests (requires Docker + model)
	uv run pytest -m integration -v

.PHONY: test-docker
test-docker: ## Run only Docker-dependent tests
	uv run pytest -m docker -v

.PHONY: test-embedding
test-embedding: ## Run only embedding model tests
	uv run pytest -m embedding -v

.PHONY: coverage
coverage: ## Run tests with coverage report
	uv run pytest -m "not integration" --cov=. --cov-report=html --cov-report=term-missing

# ── Docker ────────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up: ## Start Chroma service (always-on)
	docker compose -f docker/docker-compose.yml up -d

.PHONY: docker-up-verify
docker-up-verify: ## Start Chroma + angr-runner
	docker compose -f docker/docker-compose.yml --profile verify up -d

.PHONY: docker-build
docker-build: ## Build all Docker images
	docker compose -f docker/docker-compose.yml --profile all build

.PHONY: docker-down
docker-down: ## Stop all services
	docker compose -f docker/docker-compose.yml --profile all down

.PHONY: docker-logs
docker-logs: ## Tail Docker service logs
	docker compose -f docker/docker-compose.yml logs -f

# ── Utilities ─────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

.PHONY: pre-commit
pre-commit: ## Install pre-commit hooks
	uv run pre-commit install

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
