from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

def load_env() -> dict:
    """
    Load environment variables from a .env file.
    Search order:
      1) <repo_root>/.env
      2) <repo_root>/config/env/.env
      3) First .env found walking up from CWD
    Returns a dict with key paths and prints a short summary.
    """
    # Try repo root .env
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [repo_root / ".env", repo_root / "config" / "env" / ".env"]
    env_path = None
    for c in candidates:
        if c.exists():
            env_path = str(c)
            break
    if env_path is None:
        # Fallback to python-dotenv discovery
        found = find_dotenv(usecwd=True)
        env_path = found if found else ""

    if env_path:
        load_dotenv(env_path)

    cfg = {
        "DATA_ROOT": os.getenv("DATA_ROOT"),
        "MODELS_ROOT": os.getenv("MODELS_ROOT"),
        "CACHE_ROOT": os.getenv("CACHE_ROOT"),
    }
    print(f"[env] loaded={bool(env_path)} path='{env_path}' "
          f"DATA_ROOT={cfg['DATA_ROOT']} MODELS_ROOT={cfg['MODELS_ROOT']}")
    return cfg
