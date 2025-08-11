SHELL := /bin/bash
.PHONY: help bootstrap env clean

PY ?= python3
VENV_DIR := .venv

help:
	@echo "Targets:"
	@echo "  bootstrap   Create venv, upgrade pip, install requirements, copy .env.example -> .env (if missing)"
	@echo "  env         Print key environment vars"
	@echo "  clean       Remove virtualenv and caches"

bootstrap:
	@set -euo pipefail; \
	if ! command -v $(PY) >/dev/null 2>&1; then \
		echo "ERROR: '$(PY)' not found. Please install Python 3.10+."; exit 1; \
	fi; \
	if [ ! -d $(VENV_DIR) ]; then \
		$(PY) -m venv $(VENV_DIR); \
	fi; \
	source $(VENV_DIR)/bin/activate; \
	python -m pip install --upgrade pip; \
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi; \
	if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; fi; \
	echo "Bootstrap complete."

env:
	@set -a; \
	source ./.env 2>/dev/null || true; \
	set +a; \
	echo "DATA_ROOT=$${DATA_ROOT}"; \
	echo "MODELS_ROOT=$${MODELS_ROOT}"; \
	echo "CACHE_ROOT=$${CACHE_ROOT}"

clean:
	@rm -rf $(VENV_DIR) __pycache__ .pytest_cache .cache
