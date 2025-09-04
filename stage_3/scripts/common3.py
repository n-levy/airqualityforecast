from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "providers_raw"
PROC_DIR = DATA_DIR / "providers_processed"

def setup_logging(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
