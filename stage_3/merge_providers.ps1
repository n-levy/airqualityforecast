from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = ["city", "date", "pm25", "pm10", "no2", "o3"]

def load_one(parquet_path: Path, provider: str) -> pd.DataFrame:
    if not parquet_path.exists():
        print(f"WARNING: missing {provider} file: {parquet_path}")
        return pd.DataFrame(columns=["provider"] + REQUIRED)
    df = pd.read_parquet(parquet_path)
    # Ensure columns and types align
    for c in REQUIRED:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[REQUIRED].copy()
    df["city"] = df["city"].astype("string")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["pm25","pm10","no2","o3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["provider"] = provider
    # Drop rows without city/date
    df = df.dropna(subset=["city","date"])
    return df[["provider"] + REQUIRED]

def main() -> int:
    root = Path(__file__).resolve().parents[1]  # .../stage_3
    out_dir = root / "data" / "providers_processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "cams": out_dir / "cams_forecast.parquet",
        "aurora": out_dir / "aurora_forecast.parquet",
        "noaa_gefs_aerosol": out_dir / "noaa_gefs_aerosol_forecast.parquet",
    }

    frames = []
    for prov, p in files.items():
        frames.append(load_one(p, prov))

    if not frames:
        print("ERROR: no provider frames available")
        return 2

    df_all = pd.concat(frames, ignore_index=True)
    if df_all.empty:
        print("ERROR: merged dataframe is empty")
        return 3

    # Sort for nice reading
    df_all = df_all.sort_values(["city", "date", "provider"])
    out_parquet = out_dir / "all_providers.parquet"
    df_all.to_parquet(out_parquet, index=False)

    # Optional CSV for humans
    out_csv = out_dir / "all_providers_sample.csv"
    df_all.head(50).to_csv(out_csv, index=False, encoding="utf-8")

    print(f"Wrote: {out_parquet} (rows={len(df_all)})")
    print(f"Wrote sample CSV: {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
