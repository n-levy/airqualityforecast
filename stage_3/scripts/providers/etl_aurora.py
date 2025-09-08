# stage_3/scripts/providers/etl_aurora.py
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Ensure we can import from .../scripts even if PYTHONPATH isn't set
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.dates import parse_date_iso  # noqa: E402

PROVIDER = "aurora"
log = logging.getLogger("etl_aurora")


def _resolve_sample_csv(cfg: dict, provider: str) -> Path:
    prov = (cfg.get("providers") or {}).get(provider) or {}
    for key in ("sample_csv", "sample", "sample_path", "sample_file"):
        if key in prov and prov[key]:
            return Path(prov[key])
    samples_dir = (cfg.get("paths") or {}).get("samples_dir") or "data/samples"
    filename = prov.get("filename") or f"{provider}_sample.csv"
    return Path(samples_dir) / filename


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    need = ["city", "date", "pm25", "pm10", "no2", "o3"]
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[need].copy()

    df["city"] = df["city"].astype("string")
    df["date"] = parse_date_iso(df["date"])
    for col in ["pm25", "pm10", "no2", "o3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["city", "date"]).reset_index(drop=True)
    return df


def etl_provider(config_path: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    sample_csv = _resolve_sample_csv(cfg, PROVIDER)
    out_raw = Path(cfg["paths"]["providers_raw"]) / f"{PROVIDER}_raw.csv"
    out_parquet = (
        Path(cfg["paths"]["providers_processed"]) / f"{PROVIDER}_forecast.parquet"
    )
    cities = list(cfg.get("cities", []))

    log.info("[aurora] Reading sample CSV: %s", sample_csv)
    df = pd.read_csv(sample_csv)

    if cities:
        log.info("[aurora] Filtering to cities: %s", cities)
        df = df[df["city"].isin(cities)].copy()

    out_raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_raw, index=False)
    log.info("[aurora] Wrote raw CSV: %s (rows=%d)", out_raw, len(df))

    df_norm = normalize(df)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_norm.to_parquet(out_parquet, index=False)
    log.info("[aurora] Wrote processed parquet: %s", out_parquet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    etl_provider(Path(args.config))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    try:
        main()
    except Exception:
        log.exception("ETL AURORA failed")
        raise
