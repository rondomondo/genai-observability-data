.DEFAULT_GOAL := help
MAKEFLAGS     += --no-print-directory
SHELL         := /bin/bash

BOLD   := \033[1m
RED    := \033[31m
GREEN  := \033[32m
CYAN   := \033[36m
RESET  := \033[0m

UV     := uv
PYTEST := $(UV) run --extra dev pytest
RUFF   := $(UV) run --extra dev ruff
MYPY   := $(UV) run --extra dev mypy

IMAGE_NAME := embedder-sidecar
IMAGE_TAG  := dev
EMBED_MODEL ?= Qwen/Qwen3-Embedding-0.6B

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'

.PHONY: uv-lock
uv-lock: ## Generate uv.lock without installing (resolves model extra but does not download torch locally)
	$(UV) lock

.PHONY: install
install: uv-lock ## Install dev dependencies (no torch/sentence-transformers)
	$(UV) sync --extra dev
	@printf "$(GREEN)Installed embedder-sidecar dev dependencies.$(RESET)\n"

.PHONY: test
test: ## Run tests (no torch required)
	$(PYTEST) -v --tb=short

.PHONY: lint
lint: ## Lint and auto-fix with ruff
	$(RUFF) check --fix .

.PHONY: format
format: ## Format with ruff
	$(RUFF) format .

.PHONY: typecheck
typecheck: ## Static type checking with mypy
	$(MYPY) app

.PHONY: ci
ci: format lint typecheck test ## Run full CI pipeline
	@printf "$(GREEN)CI passed.$(RESET)\n"

.PHONY: build-image
build-image: uv-lock ## Build the Docker image (bakes model into image)
	docker build \
	  --build-arg EMBED_MODEL=$(EMBED_MODEL) \
	  -t $(IMAGE_NAME):$(IMAGE_TAG) \
	  .
	@printf "$(GREEN)Built $(IMAGE_NAME):$(IMAGE_TAG) with model $(EMBED_MODEL)$(RESET)\n"
