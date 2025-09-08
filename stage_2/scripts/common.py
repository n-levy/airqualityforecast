from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# ----- Load .env (for OPENAQ_API_KEY etc.) -----
ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=False)

# ----- Project directories -----
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = ROOT / "logs"  # IMPORTANT: keep this name; validate_data.py imports LOG_DIR

def ensure_dirs() -> None:
    for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def _ensure_utf8_stdio() -> None:
    # Make stdout/stderr robust to Unicode (e.g., when PowerShell pipes the output)
    for name in ("stdout", "stderr"):
        try:
            s = getattr(sys, name)
            if hasattr(s, "reconfigure"):
                s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

def setup_logging(name: str = "stage2", level: int = logging.INFO) -> logging.Logger:
    _ensure_utf8_stdio()
    ensure_dirs()

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # avoid duplicate handlers if called twice

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fh_path = LOG_DIR / f"{name}_{ts}.log"
        fh = logging.FileHandler(fh_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If file handler fails for any reason, carry on with console logging only
        pass

    logger.propagate = False
    return logger

def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
