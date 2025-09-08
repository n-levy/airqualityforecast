from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common import setup_logging, load_yaml_config, ensure_dirs, INTERIM_DIR, PROCESSED_DIR, LOG_DIR

POLLUTANTS = ["pm25", "pm10", "no2", "o3"]
OPTIONAL_NUMERIC = ["temp_c", "humidity"]


def clean_city(s: str) -> str:
    return (s or "").strip().title()


def summarize(df_in: pd.DataFrame, neg_counts_before: Dict[str, int]) -> str:
    """
    Build a human-readable summary string and a DataFrame for CSV export.
    """
    lines: List[str] = []
    lines.append("=== Stage 2 Processing Summary ===")

    # Basic shape + date range
    total_rows = len(df_in)
    date_min = df_in["date"].min()
    date_max = df_in["date"].max()
    lines.append(f"Rows: {total_rows}")
    lines.append(f"Date range: {date_min.date() if pd.notna(date_min) else 'NA'} → {date_max.date() if pd.notna(date_max) else 'NA'}")

    # Per-city counts
    per_city = df_in.groupby("city", dropna=False)["date"].count().sort_values(ascending=False)
    lines.append("\nRows per city:")
    for city, cnt in per_city.items():
        lines.append(f"  - {city}: {cnt}")

    # Coverage & stats per pollutant
    lines.append("\nPollutant coverage and stats:")
    stat_rows = []
    for col in POLLUTANTS:
        if col in df_in.columns:
            nonnull = df_in[col].notna().sum()
            coverage = (nonnull / total_rows * 100.0) if total_rows else 0.0
            neg_fixed = neg_counts_before.get(col, 0)
            col_min = df_in[col].min()
            col_med = df_in[col].median()
            col_max = df_in[col].max()
            stat_rows.append(
                {
                    "metric": col,
                    "non_null": int(nonnull),
                    "coverage_pct": round(coverage, 2),
                    "negatives_clamped": int(neg_fixed),
                    "min": float(col_min) if pd.notna(col_min) else None,
                    "median": float(col_med) if pd.notna(col_med) else None,
                    "max": float(col_max) if pd.notna(col_max) else None,
                }
            )
            lines.append(
                f"  - {col}: coverage={coverage:.2f}% | negatives_clamped={neg_fixed} | min/med/max={col_min}/{col_med}/{col_max}"
            )

    # Humidity note
    if "humidity" in df_in.columns:
        hmin = df_in["humidity"].min()
        hmax = df_in["humidity"].max()
        lines.append(f"\nHumidity range after clipping: {hmin} to {hmax} (expected [0,1])")

    # Turn per-pollutant rows into DataFrame for CSV
    summary_df = pd.DataFrame(stat_rows, columns=["metric", "non_null", "coverage_pct", "negatives_clamped", "min", "median", "max"])

    # Write text + CSV summaries
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    txt_path = LOG_DIR / f"processing_summary_{ts}.txt"
    csv_path = LOG_DIR / f"processing_summary_{ts}.csv"

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    if not summary_df.empty:
        summary_df.to_csv(csv_path, index=False, encoding="utf-8")

    # Return text summary for console logging
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 - Process data (clamp negatives, clip humidity, summarize)")
    parser.add_argument("--config", type=str, required=True, help="Path to data_sources.yaml")
    args = parser.parse_args()

    logger = setup_logging("process_data")
    ensure_dirs()

    _ = load_yaml_config(Path(args.config))  # reserved for future branching

    src = INTERIM_DIR / "validated_air_quality.csv"
    if not src.exists():
        raise FileNotFoundError(f"Validated file not found: {src}. Run validate_data.py first.")
    df = pd.read_csv(src, parse_dates=["date"])

    # Normalize city names
    df["city"] = df["city"].astype(str).map(clean_city)

    # Count negatives BEFORE clamping (for summary)
    neg_counts_before: Dict[str, int] = {}
    for col in POLLUTANTS:
        if col in df.columns:
            neg_counts_before[col] = int((df[col].notna() & (df[col] < 0)).sum())

    # Clamp negatives for pollutants to 0, and humidity to [0,1]
    for col in POLLUTANTS:
        if col in df.columns:
            df.loc[df[col].notna() & (df[col] < 0), col] = 0.0
    if "humidity" in df.columns:
        df.loc[df["humidity"].notna() & (df["humidity"] < 0), "humidity"] = 0.0
        df.loc[df["humidity"].notna() & (df["humidity"] > 1), "humidity"] = 1.0

    # Sort and impute small gaps per city
    df = df.sort_values(["city", "date"])
    numeric_cols = [c for c in POLLUTANTS + OPTIONAL_NUMERIC if c in df.columns]
    if numeric_cols:
        df[numeric_cols] = (
            df.groupby("city")[numeric_cols]
              .apply(lambda g: g.ffill().bfill())
              .reset_index(level=0, drop=True)
        )

        # Conservative outlier clipping per column
        for col in numeric_cols:
            q_low = df[col].quantile(0.001)
            q_hi = df[col].quantile(0.999)
            df[col] = df[col].clip(lower=q_low, upper=q_hi)

    # Drop rows where all pollutants are NaN (if any pollutants exist)
    pol_cols_present = [c for c in POLLUTANTS if c in df.columns]
    if pol_cols_present:
        df = df.dropna(subset=pol_cols_present, how="all")

    # Save processed dataset
    out_path = PROCESSED_DIR / "clean_air_quality.parquet"
    df.to_parquet(out_path, index=False)

    # Build + persist summary, and log it to console
    summary_text = summarize(df, neg_counts_before)
    logger.info(summary_text)
    logger.info(f"Processed dataset written: {out_path}")


if __name__ == "__main__":
    main()
