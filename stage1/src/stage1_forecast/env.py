"""
Robust .env loader for Stage 1.
Search order:
1) <repo_root>/.env
2) <repo_root>/config/env/.env
3) First .env found by python-dotenv walking up from CWD
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv

REQUIRED = ("DATA_ROOT", "MODELS_ROOT")
OPTIONAL = ("CACHE_ROOT",)

def _repo_root() -> Path:
    # This file lives at .../stage1/src/stage1_forecast/env.py
    # parents[2] -> .../stage1 (repo root for this subproject)
    return Path(__file__).resolve().parents[2]

def _candidate_env_paths() -> list[Path]:
    rr = _repo_root()
    return [rr / ".env", rr / "config" / "env" / ".env"]

def _load_best_env() -> Optional[Path]:
    for p in _candidate_env_paths():
        if p.is_file():
            load_dotenv(dotenv_path=p, override=False)
            print(f"Loaded .env from: {p}")
            return p
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
        print(f"Loaded .env via find_dotenv: {found}")
        return Path(found)
    print("No .env found")
    return None

def load_and_validate() -> dict:
    _load_best_env()
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing or empty: {', '.join(missing)}")
    cfg = {k: os.getenv(k) for k in (*REQUIRED, *OPTIONAL)}
    print("DATA_ROOT:", cfg.get("DATA_ROOT"))
    print("MODELS_ROOT:", cfg.get("MODELS_ROOT"))
    return cfg
