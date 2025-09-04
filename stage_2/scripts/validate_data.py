from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from common import setup_logging, load_yaml_config, ensure_dirs, RAW_DIR, INTERIM_DIR, LOG_DIR

BASE_REQUIRED = ["city", "date"]
POLLUTANTS = ["pm25", "pm10", "no2", "o3"]
OPTIONAL_NUMERIC = ["temp_c", "humidity"]

def latest_raw_csv() -> Path:
    files = sorted(list(RAW_DIR.glob("raw_air_quality_*.csv")) + list(RAW_DIR.glob("raw_air_quality_openaq_*.csv")))
    if not files:
        raise FileNotFoundError("No raw files found in data/raw/. Did you run fetch_data.py?")
    return files[-1]

def validate_columns(df: pd.DataFrame) -> List[str]:
    missing = [c for c in BASE_REQUIRED if c not in df.columns]
    if not any(col in df.columns for col in POLLUTANTS):
        missing.append("at_least_one_pollutant(pm25|pm10|no2|o3)")
    return missing

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> List[str]:
    errors = []
    for col in cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.isna().any() and df[col].notna().any():
                    errors.append(f"column {col} has non-numeric values that could not be coerced")
                df[col] = coerced
    return errors

def main():
    parser = argparse.ArgumentParser(description="Stage 2 - Validate raw data (soft negatives)")
    parser.add_argument("--config", type=str, required=True, help="Path to data_sources.yaml")
    args = parser.parse_args()

    logger = setup_logging("validate_data")
    ensure_dirs()

    _ = load_yaml_config(Path(args.config))  # reserved for future branching

    src = latest_raw_csv()
    logger.info(f"Validating raw file: {src}")
    df = pd.read_csv(src)

    # 1) column presence (hard)
    missing = validate_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2) parse date (hard)
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as e:
        raise ValueError(f"date parsing failed: {e}")

    # 3) numeric coercion (hard if non-coercible values exist)
    numeric_cols = [c for c in POLLUTANTS + OPTIONAL_NUMERIC if c in df.columns]
    hard_errors = coerce_numeric(df, numeric_cols)
    if hard_errors:
        raise ValueError("Validation errors: " + "; ".join(hard_errors))

    # 4) soft checks (do not fail the run)
    warnings: List[str] = []
    soft_records: List[Dict[str, Any]] = []

    # negative pollutants → soft warning
    for col in [c for c in POLLUTANTS if c in df.columns]:
        mask = df[col].notna() & (df[col] < 0)
        if mask.any():
            cnt = int(mask.sum())
            warnings.append(f"{col} has {cnt} negative values (will be clamped to 0 in processing).")
            # capture a few sample rows for the report
            sample = df.loc[mask, ["city", "date", col]].copy()
            sample["issue"] = f"negative_{col}"
            soft_records.append(sample.head(20))  # keep it small

    # humidity outside [0,1] → soft warning
    if "humidity" in df.columns:
        mask_low = df["humidity"].notna() & (df["humidity"] < 0)
        mask_hi = df["humidity"].notna() & (df["humidity"] > 1)
        if mask_low.any() or mask_hi.any():
            cnt = int(mask_low.sum() + mask_hi.sum())
            warnings.append(f"humidity out of [0,1] in {cnt} rows (will be clipped in processing).")
            sample = df.loc[(mask_low | mask_hi), ["city", "date", "humidity"]].copy()
            sample["issue"] = "humidity_out_of_bounds"
            soft_records.append(sample.head(20))

    # write soft warnings report if any
    if warnings:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        report_path = LOG_DIR / "validation_soft_warnings.csv"
        if soft_records:
            rep = pd.concat(soft_records, ignore_index=True)
            rep.to_csv(report_path, index=False)
        logger.warning(" | ".join(warnings))
        if soft_records:
            logger.warning(f"Sample rows with issues saved to: {report_path}")

    # 5) write to interim and finish
    out_path = INTERIM_DIR / "validated_air_quality.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Validation passed (with {len(warnings)} warnings). Wrote: {out_path}")

if __name__ == "__main__":
    main()
