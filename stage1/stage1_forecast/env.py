from __future__ import annotations
import os
from pathlib import Path

def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def data_root() -> Path:
    return _ensure(Path(os.environ.get('DATA_ROOT', r'C:\aqf311\data')))

def models_root() -> Path:
    return _ensure(Path(os.environ.get('MODELS_ROOT', r'C:\aqf311\models')))

def cache_root() -> Path:
    return _ensure(Path(os.environ.get('CACHE_ROOT', r'C:\aqf311\.cache')))
