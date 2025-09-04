
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from common3 import setup_logging, load_yaml_config, ensure_dirs, RAW_DIR, PROC_DIR

PROVIDER = "noaa_gefs_aerosol"

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = ["city", "date", "pm25", "pm10", "no2", "o3"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[need_cols].copy()
    df["city"] = df["city"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["pm25","pm10","no2","o3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["city","date"])
    return df

def etl_provider(cfg_path: Path) -> Path:
    logger = setup_logging("etl_" + PROVIDER)
    ensure_dirs()
    cfg = load_yaml_config(cfg_path)
    options = cfg["providers"][PROVIDER]["options"]
    mode = options.get("mode", "sample")
    cities = [c.lower() for c in options.get("cities", [])]
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")

    if mode == "sample":
        sample_path = Path(cfg["paths"]["samples_dir"]) / options["sample_file"]
        if not sample_path.exists():
            raise FileNotFoundError("Sample file not found: " + str(sample_path))
        df = pd.read_csv(sample_path)
        logger.info("Loaded sample data: " + str(sample_path) + " (rows=" + str(len(df)) + ")")
    else:
        sample_path = Path(cfg["paths"]["samples_dir"]) / options["sample_file"]
        df = pd.read_csv(sample_path)
        logger.warning("Live mode not configured; using sample data instead.")

    if cities:
        df = df[df["city"].str.lower().isin(cities)].copy()

    df = normalize(df)

    raw_out = RAW_DIR / ("raw_" + PROVIDER + "_" + ts + ".csv")
    proc_out = PROC_DIR / (PROVIDER + "_forecast.parquet")
    df.to_csv(raw_out, index=False, encoding="utf-8")
    df.to_parquet(proc_out, index=False)

    logger.info("Wrote raw CSV: " + str(raw_out) + " (rows=" + str(len(df)) + ")")
    logger.info("Wrote processed parquet: " + str(proc_out))
    return proc_out

def main():
    ap = argparse.ArgumentParser(description="Stage 3 ETL for " + PROVIDER + " (sample-mode default)")
    ap.add_argument("--config", required=True, help="Path to providers.yaml")
    args = ap.parse_args()
    etl_provider(Path(args.config))

if __name__ == "__main__":
    main()
