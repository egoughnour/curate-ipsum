.DEFAULT_GOAL := help
SHELL := /bin/bash

# ── Development ───────────────────────────────────────────────────────────────

.PHONY: install
install: ## Install project + dev dependencies via uv
	uv sync --extra dev --extra rag --extra graph --extra synthesis --extra graphdb

.PHONY: lock
lock: ## Regenerate uv.lock
	uv lock

.PHONY: update
update: ## Update all dependencies and regenerate lock
	uv lock --upgrade
	uv sync --extra dev --extra rag --extra graph --extra synthesis --extra graphdb

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

# ── Release ───────────────────────────────────────────────────────────────

.PHONY: bump
bump: ## Bump version in pyproject.toml (usage: make bump VERSION=0.3.0)
ifndef VERSION
	$(error VERSION is required — usage: make bump VERSION=0.3.0)
endif
	@echo "Bumping version to $(VERSION)"
	sed -i 's/^version = ".*"/version = "$(VERSION)"/' pyproject.toml
	python3 -c "import json,pathlib; \
	[( \
	  p:=pathlib.Path(f), \
	  d:=json.loads(p.read_text()), \
	  d.__setitem__('version','$(VERSION)'), \
	  [pkg.__setitem__('version','$(VERSION)') for pkg in d.get('packages',[])], \
	  p.write_text(json.dumps(d,indent=2)+'\n') \
	) for f in ('server.json','manifest.json')]"
	uv lock
	@echo "Updated pyproject.toml, server.json, manifest.json → $(VERSION)"

.PHONY: release
release: bump ## Bump, commit, tag, push (usage: make release VERSION=0.3.0)
	git add pyproject.toml uv.lock server.json manifest.json
	git commit -m "release: v$(VERSION)"
	git tag -a "v$(VERSION)" -m "v$(VERSION)"
	git push origin main --follow-tags
	@echo "Pushed v$(VERSION) — GitHub Actions will handle PyPI + GHCR"

.PHONY: docker-mcp
docker-mcp: ## Build MCP server Docker image locally
	docker build -f docker/Dockerfile.mcp-server -t curate-ipsum:local .

# ── Documentation ─────────────────────────────────────────────────────────────

.PHONY: docs
docs: ## Build Sphinx HTML documentation
	uv run sphinx-build -b html docs/ docs/_build/html

.PHONY: docs-live
docs-live: ## Serve docs with live reload (requires sphinx-autobuild)
	uv run sphinx-autobuild docs/ docs/_build/html --open-browser

.PHONY: docs-clean
docs-clean: ## Remove built documentation
	rm -rf docs/_build

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
