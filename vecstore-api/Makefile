# ============================================================
# vecstore-api — Makefile
# ============================================================
.DEFAULT_GOAL := help
MAKEFLAGS     += --no-print-directory
SHELL         := /bin/bash

# -- Colors ---------------------------------------------------
BOLD   := \033[1m
RED    := \033[31m
GREEN  := \033[32m
CYAN   := \033[36m
YELLOW := \033[33m
RESET  := \033[0m

# -- Tooling --------------------------------------------------
UV     := uv
PYTEST := $(UV) run --extra dev pytest
RUFF   := $(UV) run --extra dev ruff
MYPY   := $(UV) run --extra dev mypy

# -- Paths ----------------------------------------------------
PKG_SRC := app

# System packages required
SYS_DEPS := git python3 uv docker

# -- Help ------------------------------------------------------
.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'

# -- System checks --------------------------------------------
.PHONY: check-sys-deps
check-sys-deps: ## Verify required system packages are installed (Linux only; skipped on macOS)
	@if [ "$$(uname)" = "Darwin" ]; then \
	  echo "  check-sys-deps skipped (macOS)"; \
	  exit 0; \
	fi; \
	missing=""; \
	for pkg in $(SYS_DEPS); do \
	  if ! command -v $$pkg >/dev/null 2>&1; then \
	    missing="$$missing $$pkg"; \
	  fi; \
	done; \
	if [ -n "$$missing" ]; then \
	  printf "$(RED)ERROR:$(RESET) missing system packages:$$missing\n"; \
	  exit 1; \
	fi; \
	echo "  system deps OK"

.PHONY: check-uv
check-uv: ## Verify uv is available
	@command -v $(UV) >/dev/null 2>&1 || \
	  (printf "$(RED)ERROR:$(RESET) uv not found. Install: https://docs.astral.sh/uv/\n"; exit 1)

.PHONY: env-check
env-check: ## Verify .env exists (or .env.example fallback) and warn on missing keys
	@if [ ! -f .env ]; then \
	  if [ -f .env.example ]; then \
	    printf "$(YELLOW)WARNING:$(RESET) .env missing -- copying from .env.example\n"; \
	    cp .env.example .env; \
	  else \
	    printf "$(RED)ERROR:$(RESET) No .env found\n"; \
	    exit 1; \
	  fi; \
	fi
	@missing=""; \
	for key in GEMINI_API_KEY REDIS_URL API_KEY; do \
	  grep -qE "^$$key=.+" .env 2>/dev/null || missing="$$missing $$key"; \
	done; \
	if [ -n "$$missing" ]; then \
	  printf "$(YELLOW)WARNING:$(RESET) unset keys in .env:$$missing\n"; \
	else \
	  printf "$(GREEN).env OK$(RESET)\n"; \
	fi

# -- Install --------------------------------------------------
.PHONY: install
install: check-sys-deps check-uv ## Install the package and all dev dependencies
	$(UV) sync --all-extras
	@printf "$(GREEN)Installed vecstore-api with dev extras.$(RESET)\n"

# -- Dev server -----------------------------------------------
.PHONY: dev
dev: check-uv ## Run the API server in reload mode (requires .env)
	$(UV) run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# -- Code quality ---------------------------------------------
.PHONY: format
format: check-uv ## Format Python code with ruff
	$(RUFF) format .

.PHONY: lint
lint: check-uv ## Lint and auto-fix with ruff
	$(RUFF) check --fix .

.PHONY: typecheck
typecheck: check-uv ## Static type checking with mypy
	$(MYPY) $(PKG_SRC)

.PHONY: ci
ci: format lint typecheck test ## Run full CI pipeline (format, lint, typecheck, test)
	@printf "$(GREEN)CI passed.$(RESET)\n"

# -- Tests ----------------------------------------------------
.PHONY: test
test: check-uv ## Run all tests with coverage
	$(PYTEST) --cov=$(PKG_SRC) --cov-report=term-missing

.PHONY: test-unit
test-unit: check-uv ## Run unit tests only (no external services)
	$(PYTEST) -m "not integration"

.PHONY: test-integration
test-integration: check-uv ## Run integration tests (requires Redis and OTel stack)
	$(PYTEST) -m integration

.PHONY: watch
watch: check-uv ## Re-run unit tests on file change (requires watchexec)
	@command -v watchexec >/dev/null 2>&1 || \
	  (printf "$(RED)ERROR:$(RESET) watchexec not found. Install: brew install watchexec\n"; exit 1)
	watchexec --exts py --restart -- $(PYTEST) -v --tb=short -m "not integration"

# -- OTel / infrastructure stack ------------------------------
.PHONY: stack-up
stack-up: ## Start the local stack (Redis, OTel Collector, etc.)
	docker compose up -d
	@printf "$(CYAN)Waiting for Redis...$(RESET)\n"
	@for i in $$(seq 1 20); do \
	  docker compose exec redis redis-cli ping >/dev/null 2>&1 && \
	    printf "$(GREEN)Redis ready$(RESET)\n" && break; \
	  printf "  ... waiting ($$i/20)\n"; sleep 2; \
	done
	@printf "$(GREEN)Stack ready:$(RESET)\n"
	@printf "  API:    http://localhost:8000\n"
	@printf "  Redis:  redis://localhost:6379\n"

.PHONY: stack-down
stack-down: ## Stop the local stack
	docker compose down

.PHONY: stack-logs
stack-logs: ## Tail logs from all stack services
	docker compose logs -f

.PHONY: stack-reset
stack-reset: ## Wipe stack volumes and restart fresh
	docker compose down -v
	$(MAKE) stack-up

# -- Build / image --------------------------------------------
.PHONY: build-image
build-image: ## Build the Docker image
	docker build -t vecstore-api:latest .

# -- Version bumps --------------------------------------------
.PHONY: bump-patch
bump-patch: ## Bump patch version in pyproject.toml (0.1.2 -> 0.1.3)
	@old=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	maj=$$(echo $$old | cut -d. -f1); \
	min=$$(echo $$old | cut -d. -f2); \
	pat=$$(echo $$old | cut -d. -f3); \
	new="$$maj.$$min.$$((pat + 1))"; \
	sed -i '' "s/^version = \"$$old\"/version = \"$$new\"/" pyproject.toml; \
	printf "  $(GREEN)$$old$(RESET) -> $(GREEN)$$new$(RESET)\n"

.PHONY: bump-minor
bump-minor: ## Bump minor version in pyproject.toml (0.1.2 -> 0.2.0)
	@old=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	maj=$$(echo $$old | cut -d. -f1); \
	min=$$(echo $$old | cut -d. -f2); \
	new="$$maj.$$((min + 1)).0"; \
	sed -i '' "s/^version = \"$$old\"/version = \"$$new\"/" pyproject.toml; \
	printf "  $(GREEN)$$old$(RESET) -> $(GREEN)$$new$(RESET)\n"

# -- Clean ----------------------------------------------------
.PHONY: clean-pyc
clean-pyc: ## Remove Python bytecode and tool caches
	@printf "$(CYAN)Cleaning$(RESET) Python caches...\n"
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .mypy_cache  -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name .ruff_cache  -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@printf "  $(GREEN)ok$(RESET)\n"

.PHONY: clean
clean: clean-pyc ## Remove all build artifacts and caches
	@printf "$(CYAN)Cleaning$(RESET) build artifacts...\n"
	@find . -type d -name dist  -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf coverage/ .coverage htmlcov/ 2>/dev/null || true
	@printf "  $(GREEN)ok$(RESET)\n"

.PHONY: clean-all
clean-all: clean ## Remove all artifacts and the virtualenv (full reset)
	@printf "$(CYAN)Removing$(RESET) .venv...\n"
	@rm -rf .venv
	@find . -type f -name 'uv.lock' -not -path './.venv/*' -exec rm -f {} + 2>/dev/null || true
	@printf "  $(GREEN)ok$(RESET)\n"
