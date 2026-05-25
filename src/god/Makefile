# ============================================================
# src/Makefile  --  convenience wrapper for working inside src/
#
# All targets delegate to the root Makefile with the correct
# working directory already set.  Run from repo root instead
# if you need targets that operate on the whole project
# (docker, services, clean, etc.).
# ============================================================
.DEFAULT_GOAL := help
MAKEFLAGS     += --no-print-directory
SHELL         := /bin/bash
ROOT          := ..

BOLD   := \033[1m
CYAN   := \033[36m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m

UV     := uv
PYTEST := $(UV) run pytest

# -- Help ------------------------------------------------------
.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'

# -- Install ---------------------------------------------------
.PHONY: install
install: ## Install all dependencies (uv sync --all-extras)
	@$(UV) sync --all-extras
	@printf "$(GREEN)Dependencies installed.$(RESET)\n"

# -- Code quality ----------------------------------------------
.PHONY: format
format: ## Format Python code (ruff)
	$(UV) run ruff format .

.PHONY: lint
lint: ## Lint and auto-fix Python code (ruff)
	$(UV) run ruff check --fix .

.PHONY: typecheck
typecheck: ## Run mypy type checks
	$(UV) run mypy .

# -- Tests -----------------------------------------------------
.PHONY: test
test: ## Run full test suite with coverage
	$(PYTEST) tests/ -v --tb=short --cov=. --cov-report=term-missing

.PHONY: test-unit
test-unit: ## Run unit tests only (excludes integration mark)
	$(PYTEST) tests/ -v --tb=short -m "not integration"

.PHONY: test-integration
test-integration: ## Run integration tests only (requires services)
	$(PYTEST) tests/ -v --tb=short -m integration

.PHONY: ci
ci: format lint typecheck test ## Run full CI pipeline
	@echo "CI passed"

# -- Evals -----------------------------------------------------
.PHONY: eval
eval: ## Run the full evaluation suite
	$(UV) run python -m evals.eval_all

.PHONY: eval-extractor
eval-extractor: ## Run extractor-stage evals only
	$(UV) run python -m evals.eval_extractor

.PHONY: eval-validator
eval-validator: ## Run validator-stage evals only
	$(UV) run python -m evals.eval_validator

.PHONY: eval-summarizer
eval-summarizer: ## Run summarizer-stage evals only
	$(UV) run python -m evals.eval_summarizer

# -- Dev workflow ----------------------------------------------
.PHONY: server
server: ## Start the API server with live reload
	$(UV) run uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

.PHONY: mock
mock: ## Start the API server in mock mode (no services or API keys needed)
	MOCK_MODE=true $(UV) run uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

.PHONY: ingest
ingest: ## Ingest sample documents (DOCS_DIR overridable)
	$(UV) run ingest $(CURDIR)/data/sample_docs

# -- Services (delegates to root) ------------------------------
.PHONY: services-up
services-up: ## Start Elasticsearch + Redis via docker compose
	$(MAKE) -C $(ROOT) services-up

.PHONY: services-down
services-down: ## Stop Elasticsearch + Redis containers
	$(MAKE) -C $(ROOT) services-down

.PHONY: es-reset
es-reset: ## Delete and recreate the Elasticsearch index
	$(MAKE) -C $(ROOT) es-reset
