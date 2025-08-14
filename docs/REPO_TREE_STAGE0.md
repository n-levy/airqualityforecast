# Repo Tree – PM2.5 Berlin (Stage 0 Scaffold)

stage1/
├─ apps/
│  ├─ etl/                # source adapters (uba\_obs.py, cams\_stage1.py, icon\_gfs.py)
│  ├─ features/           # as-of issue time builder (build.py)
│  ├─ train/              # model training (train\_xgb.py), eval
│  ├─ infer/              # hourly inference (infer\_hourly.py)
│  ├─ verify/             # verification vs. obs \& benchmarks (verify\_hourly.py)
│  ├─ publish/            # export JSON/CSV for static dashboard (export\_static.py)
│  └─ serve/              # (later) FastAPI app
├─ config/
│  ├─ cities/
│  │  └─ berlin.yml       # stations, tiles, timezone, holidays
│  ├─ schemas/            # json schemas for raw/curated/features
│  └─ env/
│     └─ .env.example     # 12-factor config (DATA\_ROOT, MODELS\_ROOT, etc.)
├─ data/                  # local Parquet during Stage 1 (gitignored)
├─ models/                # model registry: artifacts + metadata (gitignored)
├─ docs/
│  ├─ CONTEXT.md
│  ├─ PRD.md
│  ├─ NFRs.md
│  ├─ EvalProtocol.md     # to be added
│  ├─ ADR-001\_storage.md
│  ├─ ADR-002\_model\_family.md
│  └─ ADR-003\_scheduler.md
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
├─ Makefile               # bootstrap, ingest\_berlin, features\_berlin, etc.
├─ pyproject.toml         # deps + tool configs
└─ README.md

