from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd  # <-- add pandas import for the helper
import yaml

# Directories resolved relative to stage_3/scripts/common3.py -> stage_3/
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "providers_raw"
PROC_DIR = DATA_DIR / "providers_processed"


def setup_logging(name: str) -> logging.Logger:
    """
    Create a logger that logs to console and to logs/<name>.log.
    Safe to call multiple times (won't duplicate handlers).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_date_iso(series: pd.Series) -> pd.Series:
    """
    Fast path for ISO dates (YYYY-MM-DD); if most values fail,
    fall back to flexible parsing once. Keeps errors='coerce' semantics.
    """
    out = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    # If more than half are NaT (e.g., provider sent mixed formats), try flexible parsing
    if out.isna().mean() > 0.5:
        out = pd.to_datetime(series, errors="coerce", utc=False)
    return out
