# PM2.5 Forecasting for Berlin — Stage 0 Scaffold

This repository is the Stage 0 scaffold for a PM2.5 forecasting system. It includes:
- Repo structure and documentation templates (`/docs`)
- User-space friendly bootstrap (`make bootstrap`)
- Minimal `docker-compose.yml` (to be extended later in CI)
- Environment config via `.env`

> Created on 2025-08-10.

## Quickstart

```bash
# 1) Create and activate local environment (no admin rights needed)
make bootstrap

# 2) Inspect environment
source .venv/bin/activate
python -V
pip list
```

## Environment variables

Set directories outside the repo for data and models if possible. Defaults can be overridden in `.env`.

- `DATA_ROOT` — Where raw/processed data is stored (default: `$HOME/pm25_data`)
- `MODELS_ROOT` — Where persisted models live (default: `$HOME/pm25_models`)
- `CACHE_ROOT` — Caches (default: `$HOME/pm25_cache`)
- `ARTIFACTS_ROOT` — Artifacts like figs/reports (default: `$HOME/pm25_artifacts`)
- `LOGS_ROOT` — Logs (default: `$HOME/pm25_logs`)

These variables are also mounted as volumes later in containers (see `docker-compose.yml`).

## Repo layout

See `docs/REPO_TREE_STAGE0.md` for the commented tree.

