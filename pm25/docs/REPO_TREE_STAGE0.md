# Repo Tree – PM2.5 Berlin (Stage 0 Scaffold)

pm25/
├─ apps/
│  ├─ etl/                # source adapters (uba_obs.py, cams_pm25.py, icon_gfs.py)
│  ├─ features/           # as-of issue time builder (build.py)
│  ├─ train/              # model training (train_xgb.py), eval
│  ├─ infer/              # hourly inference (infer_hourly.py)
│  ├─ verify/             # verification vs. obs & benchmarks (verify_hourly.py)
│  ├─ publish/            # export JSON/CSV for static dashboard (export_static.py)
│  └─ serve/              # (later) FastAPI app
├─ config/
│  ├─ cities/
│  │  └─ berlin.yml       # stations, tiles, timezone, holidays
│  ├─ schemas/            # json schemas for raw/curated/features
│  └─ env/
│     └─ .env.example     # 12-factor config (DATA_ROOT, MODELS_ROOT, etc.)
├─ data/                  # local Parquet during Stage 1 (gitignored)
├─ models/                # model registry: artifacts + metadata (gitignored)
├─ docs/
│  ├─ CONTEXT.md
│  ├─ PRD.md
│  ├─ NFRs.md
│  ├─ EvalProtocol.md     # to be added
│  ├─ ADR-001_storage.md
│  ├─ ADR-002_model_family.md
│  └─ ADR-003_scheduler.md
├─ infra/
│  └─ terraform/          # stub for S3 bucket + IAM (cloud flip)
├─ tests/
│  ├─ unit/               # unit tests (parsers, features, metrics)
│  ├─ contract/           # raw → curated schema tests
│  └─ eval/               # frozen folds, tolerances
├─ web/
│  └─ public/             # static dashboard export target (Vercel/Pages)
├─ .gitignore
├─ docker-compose.yml     # skeleton; containers may build in CI if Docker unavailable
├─ Makefile               # bootstrap, ingest_berlin, features_berlin, etc.
├─ pyproject.toml         # deps + tool configs
└─ README.md
