# ============================================================
# genai-observability-data — Makefile
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
REPO_DIR   := .
PKG_DIR    := src/god
OUT_DIR    := ./output
DATA_DIR   := ./data

# -- Generation defaults (override on command line) -----------
TRACES    := 200
RATE      := 20.0
SCENARIO  := all
SEED      :=
FORMAT    := logfmt
BATCH     := 50
OTLP      := http://localhost:4318

# -- PyPI config ----------------------------------------------
IMAGE_NAME := genai-observability-data
IMAGE_TAG  := latest

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
	@if [ ! -f $(PKG_DIR)/.env ]; then \
	  if [ -f $(PKG_DIR)/.env.example ]; then \
	    printf "$(YELLOW)WARNING:$(RESET) .env missing — copying from .env.example\n"; \
	    cp $(PKG_DIR)/.env.example $(PKG_DIR)/.env; \
	  else \
	    printf "$(YELLOW)WARNING:$(RESET) No .env found — running in mock mode (no API calls)\n"; \
	  fi; \
	fi
	@missing=""; \
	for key in ANTHROPIC_API_KEY; do \
	  grep -qE "^$$key=.+" $(PKG_DIR)/.env 2>/dev/null || missing="$$missing $$key"; \
	done; \
	if [ -n "$$missing" ]; then \
	  printf "$(YELLOW)WARNING:$(RESET) unset keys in .env:$$missing (mock mode will be used)\n"; \
	else \
	  printf "$(GREEN).env OK$(RESET)\n"; \
	fi

# -- Install --------------------------------------------------
.PHONY: install
install: check-sys-deps ## Install the god package and all dev dependencies
	@cd $(PKG_DIR) && $(UV) sync --all-extras
	@printf "$(GREEN)Installed god package with dev extras.$(RESET)\n"

.PHONY: install-editable
install-editable: check-uv ## Install god in editable mode (pip install -e .)
	@cd $(PKG_DIR) && $(UV) pip install -e ".[dev]"
	@printf "$(GREEN)Editable install complete.$(RESET)\n"

# -- Code quality ---------------------------------------------
.PHONY: format
format: check-uv ## Format Python code with ruff
	cd $(PKG_DIR) && $(RUFF) format .

.PHONY: lint
lint: check-uv ## Lint and auto-fix with ruff
	cd $(PKG_DIR) && $(RUFF) check --fix .

.PHONY: typecheck
typecheck: check-uv ## Static type checking with mypy
	cd src && $(MYPY) god

.PHONY: ci
ci: format lint typecheck test ## Run full CI pipeline (format, lint, typecheck, test)
	@printf "$(GREEN)CI passed.$(RESET)\n"

# -- Tests ----------------------------------------------------
.PHONY: test
test: check-uv ## Run all tests with coverage
	cd $(PKG_DIR) && $(PYTEST) --cov=. --cov-report=term-missing

.PHONY: test-unit
test-unit: check-uv ## Run unit tests only (no services, no API keys)
	cd $(PKG_DIR) && $(PYTEST) -m "not integration"

.PHONY: test-integration
test-integration: check-uv ## Run integration tests (requires OTel stack running)
	cd $(PKG_DIR) && $(PYTEST) -m integration

.PHONY: watch
watch: check-uv ## Re-run unit tests on file change (requires watchexec)
	@command -v watchexec >/dev/null 2>&1 || \
	  (printf "$(RED)ERROR:$(RESET) watchexec not found. Install: brew install watchexec\n"; exit 1)
	watchexec --exts py --restart -- $(PYTEST) $(PKG_DIR)/tests/ -v --tb=short -m "not integration"

# -- Data generation ------------------------------------------
.PHONY: gen
gen: check-uv ## Generate traces to stdout (TRACES/RATE/SCENARIO/SEED/FORMAT overridable)
	@mkdir -p $(OUT_DIR)
	cd $(PKG_DIR) && GOD_MOCK_MODE=true $(UV) run python -m god \
	  --traces $(TRACES) \
	  --rate $(RATE) \
	  $(if $(SCENARIO),--scenario $(SCENARIO),) \
	  $(if $(SEED),--seed $(SEED),) \
	  --format $(FORMAT)

.PHONY: gen-files
gen-files: check-uv ## Generate OTLP JSON + logfmt files into output/ (TRACES/SCENARIO/SEED overridable)
	@mkdir -p $(OUT_DIR)
	cd $(PKG_DIR) && GOD_MOCK_MODE=true $(UV) run python -m god \
	  --traces $(TRACES) \
	  --rate $(RATE) \
	  $(if $(SCENARIO),--scenario $(SCENARIO),) \
	  $(if $(SEED),--seed $(SEED),) \
	  --format otlp_traces otlp_logs logfmt \
	  --output $(CURDIR)/$(OUT_DIR)
	@printf "$(GREEN)Output written to $(OUT_DIR)/$(RESET)\n"

.PHONY: gen-otlp
gen-otlp: check-uv ## Stream generated traces to OTel Collector (OTLP=http://... overridable)
	cd $(PKG_DIR) && $(UV) run python -m god \
	  --traces $(TRACES) \
	  --rate $(RATE) \
	  $(if $(SCENARIO),--scenario $(SCENARIO),) \
	  --otlp $(OTLP)
	@printf "$(GREEN)Streamed to $(OTLP)$(RESET)\n"

.PHONY: gen-fixtures
gen-fixtures: check-uv ## Generate deterministic test fixtures (seed=42, 200 traces, all formats)
	@mkdir -p $(DATA_DIR)/fixtures
	cd $(PKG_DIR) && GOD_MOCK_MODE=true $(UV) run python -m god \
	  --traces 200 \
	  --seed 42 \
	  --format otlp_traces otlp_logs logfmt loki \
	  --output $(CURDIR)/$(DATA_DIR)/fixtures \
	  --quiet
	@printf "$(GREEN)Fixtures written to $(DATA_DIR)/fixtures/$(RESET)\n"

ALL_SCENARIOS := db_incident llm_degradation events_degradation rag_meltdown payment_outage cascading_failure quiet

.PHONY: gen-scenario
gen-scenario: check-uv ## Generate a specific scenario (SCENARIO=<name>|all required)
	@[ -n "$(SCENARIO)" ] || (printf "$(RED)ERROR:$(RESET) Set SCENARIO= (e.g. SCENARIO=cascading_failure or SCENARIO=all)\n"; exit 1)
	@mkdir -p $(OUT_DIR)
	@if [ "$(SCENARIO)" = "all" ]; then \
	  for s in $(ALL_SCENARIOS); do \
	    printf "$(CYAN)Generating scenario:$(RESET) $$s\n"; \
	    (cd $(CURDIR)/$(PKG_DIR) && GOD_MOCK_MODE=true $(UV) run python -m god \
	      --traces $(TRACES) \
	      --scenario $$s \
	      --format otlp_traces otlp_logs logfmt \
	      --otlp $(OTLP) \
	      --output $(CURDIR)/$(OUT_DIR)/$$s) || exit 1; \
	  done; \
	  printf "$(GREEN)All scenarios written to $(OUT_DIR)/$(RESET)\n"; \
	else \
	  cd $(PKG_DIR) && GOD_MOCK_MODE=true $(UV) run python -m god \
	    --traces $(TRACES) \
	    --scenario $(SCENARIO) \
	    --format otlp_traces otlp_logs logfmt \
        --otlp $(OTLP) \
	    --output $(CURDIR)/$(OUT_DIR); \
	fi

.PHONY: list-scenarios
list-scenarios: check-uv ## List all available fault injection scenario presets
	cd $(PKG_DIR) && $(UV) run python -m god --list-scenarios

# -- OTel observability stack ---------------------------------
.PHONY: stack-up
stack-up: ## Start the full OTel stack (Collector, Jaeger, Loki, Grafana, Prometheus)
	docker compose up -d
	@printf "$(CYAN)Waiting for OTel Collector...$(RESET)\n"
	@for i in $$(seq 1 20); do \
	  curl -sf http://localhost:13133 >/dev/null 2>&1 && \
	    printf "$(GREEN)OTel Collector ready$(RESET)\n" && break; \
	  printf "  ... waiting ($$i/20)\n"; sleep 3; \
	done
	@printf "$(GREEN)Stack ready:$(RESET)\n"
	@printf "  Grafana:    http://grafana:3000\n"
	@printf "  Jaeger:     http://jaeger:16686\n"
	@printf "  Prometheus: http://prometheus:9090\n"
	@printf "  OTel OTLP:  http://otlp:4318\n"
	@printf "$(GREEN)Dashboards:$(RESET)\n"
	@printf "  Grafana:    http://grafana:3000/dashboards\n"
	@printf "$(GREEN)Admin:$(RESET)\n"
	@printf "  AdminApp:   http://grafana:3000/a/grafana-lokioperational-app\n"

.PHONY: stack-down
stack-down: ## Stop the OTel stack
	docker compose down

.PHONY: stack-logs
stack-logs: ## Tail logs from all stack services
	docker compose logs -f

.PHONY: stack-reset
stack-reset: ## Wipe stack volumes and restart fresh
	docker compose down -v
	$(MAKE) stack-up;

# -- Demo workflow --------------------------------------------
.PHONY: demo
demo: stack-up gen-otlp ## Start stack then stream 100 traces (good for first run)
	@printf "$(GREEN)Demo complete. Open Grafana: http://grafana:3000$(RESET)\n"

.PHONY: demo-scenario
demo-scenario: stack-up ## Stream a cascading failure scenario to the live stack
	cd $(PKG_DIR) && $(UV) run python -m god \
	  --traces 500 \
	  --scenario cascading_failure \
	  --rate 5 \
	  --otlp $(OTLP)
	@printf "$(GREEN)Scenario streamed. Check Jaeger: http://jaeger:16686$(RESET)\n"

# -- Build and publish ----------------------------------------
.PHONY: build
build: check-uv clean-pyc ## Build wheel + sdist for PyPI
	cd $(PKG_DIR) && $(UV) build --out-dir $(CURDIR)/dist

.PHONY: bump-patch
bump-patch: ## Bump patch version in pyproject.toml (0.1.2 -> 0.1.3)
	@old=$$(grep '^version = ' $(PKG_DIR)/pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	maj=$$(echo $$old | cut -d. -f1); \
	min=$$(echo $$old | cut -d. -f2); \
	pat=$$(echo $$old | cut -d. -f3); \
	new="$$maj.$$min.$$((pat + 1))"; \
	sed -i '' "s/^version = \"$$old\"/version = \"$$new\"/" $(PKG_DIR)/pyproject.toml; \
	sed -i '' "s/^__version__ = \"$$old\"/__version__ = \"$$new\"/" $(PKG_DIR)/god/__init__.py 2>/dev/null || true; \
	printf "  $(GREEN)$$old$(RESET) -> $(GREEN)$$new$(RESET)\n"

.PHONY: bump-minor
bump-minor: ## Bump minor version in pyproject.toml (0.1.2 -> 0.2.0)
	@old=$$(grep '^version = ' $(PKG_DIR)/pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	maj=$$(echo $$old | cut -d. -f1); \
	min=$$(echo $$old | cut -d. -f2); \
	new="$$maj.$$((min + 1)).0"; \
	sed -i '' "s/^version = \"$$old\"/version = \"$$new\"/" $(PKG_DIR)/pyproject.toml; \
	printf "  $(GREEN)$$old$(RESET) -> $(GREEN)$$new$(RESET)\n"

.PHONY: bump-major
bump-major: ## Bump major version in pyproject.toml (0.1.2 -> 1.0.0)
	@old=$$(grep '^version = ' $(PKG_DIR)/pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	maj=$$(echo $$old | cut -d. -f1); \
	new="$$((maj + 1)).0.0"; \
	sed -i '' "s/^version = \"$$old\"/version = \"$$new\"/" $(PKG_DIR)/pyproject.toml; \
	printf "  $(GREEN)$$old$(RESET) -> $(GREEN)$$new$(RESET)\n"

.PHONY: publish-test
publish-test: build ## Upload to TestPyPI
	$(UV) publish --index testpypi dist/*.whl dist/*.tar.gz

.PHONY: publish
publish: build ## Upload to PyPI (live) — prompts for confirmation
	@printf "$(RED)WARNING:$(RESET) publishing to live PyPI. Ctrl-C to abort, Enter to continue.\n" && read _
	$(UV) publish dist/*.whl dist/*.tar.gz

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
	@find . -type d -name dist    -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name build   -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf coverage/ .coverage htmlcov/ 2>/dev/null || true
	@printf "  $(GREEN)ok$(RESET)\n"

.PHONY: clean-output
clean-output: ## Remove generated data files from output/ and data/fixtures/
	@rm -rf $(OUT_DIR) $(DATA_DIR)/fixtures
	@printf "  $(GREEN)output cleared$(RESET)\n"

.PHONY: clean-all
clean-all: clean clean-output ## Remove all artifacts, caches, and the virtualenv (full reset)
	@printf "$(CYAN)Removing$(RESET) .venv...\n"
	@rm -rf $(PKG_DIR)/.venv $(REPO_DIR)/.venv
	@find $(REPO_DIR) -type f -name 'uv.lock' -exec rm -rf {} + 2>/dev/null || true
	@printf "  $(GREEN)ok$(RESET)\n"
