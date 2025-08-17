# ---- Makefile (Stage 1) ----------------------------------------------
# Default Python interpreter (override with: make PY=C:\path\to\python.exe <target>)
PY ?= C:\aqf311\.venv\Scripts\python.exe

# Convenience: print which python we use
.PHONY: check-python
check-python: ## Show the Python interpreter being used
	@echo Using PY=$(PY)
	@"$(PY)" -V

.PHONY: help
help: ## Show this help
	@echo Available targets:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(lastword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Environment / bootstrap ---
.PHONY: bootstrap
bootstrap: ## One-time: install deps (wheels-friendly), copy .env.example -> .env (if missing)
	@"$(PY)" -m pip install --upgrade pip setuptools wheel
	@"$(PY)" -m pip install -r requirements.txt
	@if not exist ".env" (copy config\env\.env.example .env >NUL)
	@echo bootstrap: done

.PHONY: smoke
smoke: ## Run local smoke test to validate env variables and imports
	@"$(PY)" scripts\smoke_test.py

.PHONY: lint
lint: ## Run linters/format checks (ruff/black if configured)
	@if exist ruff.toml @"$(PY)" -m ruff check .
	@if exist pyproject.toml @"$(PY)" -m black --check .
	@echo lint: done

.PHONY: fmt
fmt: ## Auto-format (black) if configured
	@if exist pyproject.toml @"$(PY)" -m black .
	@echo fmt: done

.PHONY: test
test: ## Run unit tests
	@"$(PY)" -m pytest -q

# --- Pipeline (names mirror your apps/* layout) ---
.PHONY: ingest-berlin
ingest-berlin: ## Ingest inputs for Berlin
	@"$(PY)" apps\etl\uba_obs.py --city berlin
	@"$(PY)" apps\etl\cams_stage1.py --city berlin
	@echo ingest-berlin: OK

.PHONY: ingest-munich
ingest-munich: ## Ingest inputs for Munich
	@"$(PY)" apps\etl\uba_obs.py --city munich
	@"$(PY)" apps\etl\cams_stage1.py --city munich
	@echo ingest-munich: OK

.PHONY: ingest-hamburg
ingest-hamburg: ## Ingest inputs for Hamburg
	@"$(PY)" apps\etl\uba_obs.py --city hamburg
	@"$(PY)" apps\etl\cams_stage1.py --city hamburg
	@echo ingest-hamburg: OK

.PHONY: features-berlin
features-berlin: ## Build features for Berlin (as-of issue time)
	@"$(PY)" apps\features\build.py --city berlin
	@echo features-berlin: OK

.PHONY: features-munich
features-munich: ## Build features for Munich
	@"$(PY)" apps\features\build.py --city munich
	@echo features-munich: OK

.PHONY: features-hamburg
features-hamburg: ## Build features for Hamburg
	@"$(PY)" apps\features\build.py --city hamburg
	@echo features-hamburg: OK

.PHONY: train-pm25
train-pm25: ## Train the global + optional per-city bias-corrected PM2.5 model
	@"$(PY)" apps\train\train_xgb.py --pollutant pm25
	@echo train-pm25: OK

.PHONY: infer-hourly
infer-hourly: ## Run hourly inference for all 3 cities (current cycle)
	@"$(PY)" apps\infer\infer_hourly.py --cities berlin hamburg munich --pollutant pm25
	@echo infer-hourly: OK

.PHONY: verify-hourly
verify-hourly: ## Verify forecasts vs observations & benchmarks
	@"$(PY)" apps\verify\verify_hourly.py --cities berlin hamburg munich --pollutant pm25
	@echo verify-hourly: OK

.PHONY: publish-static
publish-static: ## Export CSV/JSON for the static dashboard
	@"$(PY)" apps\publish\export_static.py --cities berlin hamburg munich --pollutant pm25 --out web\public
	@echo publish-static: OK

# --- Utilities ---
.PHONY: clean
clean: ## Remove caches/build artifacts (keeps data/models)
	@if exist .pytest_cache rmdir /S /Q .pytest_cache
	@if exist build rmdir /S /Q build
	@if exist dist rmdir /S /Q dist
	@for /d %%D in (__pycache__) do rmdir /S /Q "%%D"
	@echo clean: done
# ---------------------------------------------------------------------
