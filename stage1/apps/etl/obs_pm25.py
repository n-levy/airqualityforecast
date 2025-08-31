# stage1/apps/etl/obs_pm25.py
"""
Fetch hourly PM2.5 observations for a city (Berlin/Hamburg/Munich),
normalize to UTC `valid_time`, and write Parquet partitioned by date.

Usage (Windows PowerShell):
  C:\aqf311\.venv\Scripts\python.exe stage1\apps\etl\obs_pm25.py --city berlin --since 2025-07-01 --until 2025-07-07
"""

from __future__ import annotations
import argparse, os
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from dateutil import parser as dtparse
import yaml
from jsonschema import validate
import pyarrow as pa
import pyarrow.parquet as pq

# Local package loader (loads .env and prints DATA_ROOT/MODELS_ROOT)
from stage1_forecast.env import load_and_validate

ROOT = Path(__file__).resolve().parents[2]  # .../stage1

def load_city_config(name: str) -> dict:
    cfg_path = ROOT / "config" / "cities" / f"{name}.yml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_schema(kind: str) -> dict:
    # kind: "raw" or "curated"
    p = ROOT / "config" / "schemas" / ("raw" if kind=="raw" else "curated") / "observations_pm25.json"
    import json
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def daterange_utc(since: datetime, until: datetime):
    d = since
    while d <= until:
        yield d
        d += timedelta(days=1)

def fetch_uba_pm25(station_ids: list[str], day_utc: datetime) -> pd.DataFrame:
    """
    TODO: Implement the actual UBA fetch here.
    Return a DataFrame with columns: station_id (str), ts (ISO8601), value (float), unit (str)
    ts may be timezone-aware or naive; we will normalize below.
    For now this returns an empty DataFrame to keep the skeleton runnable.
    """
    cols = ["station_id", "ts", "value", "unit"]
    return pd.DataFrame(columns=cols)

def normalize_curated(city: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    # Validate raw rows
    validate(instance=raw_df.to_dict(orient="records"), schema=load_schema("raw"))
    # Parse timestamps and force UTC
    ts = pd.to_datetime(raw_df["ts"], utc=True, errors="coerce")
    out = pd.DataFrame({
        "city": city,
        "valid_time": ts,
        "value": pd.to_numeric(raw_df["value"], errors="coerce"),
        "unit": raw_df["unit"].astype(str)
    }).dropna(subset=["valid_time", "value"])
    # Keep only integer hours
    out = out[out["valid_time"].dt.minute.eq(0) & out["valid_time"].dt.second.eq(0)]
    # Sort and de-duplicate
    out = out.sort_values("valid_time").drop_duplicates(subset=["valid_time"], keep="last")
    # Validate curated rows
    validate(instance=out.to_dict(orient="records"), schema=load_schema("curated"))
    return out

def write_partitioned_parquet(df: pd.DataFrame, data_root: Path, city: str):
    if df.empty:
        return 0
    df["date"] = df["valid_time"].dt.strftime("%Y-%m-%d")
    base = Path(data_root) / "curated" / "obs" / city / "pm25"
    base.mkdir(parents=True, exist_ok=True)
    written = 0
    for date_str, part in df.groupby("date"):
        part = part.drop(columns=["date"])
        folder = base / f"date={date_str}"
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / "part-0000.parquet"
        if file_path.exists():
            # idempotent: skip if already present (simple policy)
            continue
        table = pa.Table.from_pandas(part)
        pq.write_table(table, file_path)
        written += len(part)
    return written

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["berlin", "hamburg", "munich"])
    ap.add_argument("--since", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--until", required=False, help="YYYY-MM-DD (UTC); defaults to --since")
    args = ap.parse_args()

    cfg = load_and_validate()  # prints DATA_ROOT/MODELS_ROOT
    data_root = Path(cfg["DATA_ROOT"])

    city_cfg = load_city_config(args.city)
    station_ids = [s["id"] for s in city_cfg.get("stations", [])]

    since = dtparse.isoparse(args.since).replace(tzinfo=timezone.utc)
    until = dtparse.isoparse(args.until).replace(tzinfo=timezone.utc) if args.until else since

    total_rows = 0
    for day in daterange_utc(since, until):
        raw = fetch_uba_pm25(station_ids, day_utc=day)
        curated = normalize_curated(args.city, raw)
        n = write_partitioned_parquet(curated, Path(data_root), args.city)
        print(f"{args.city} {day.date()} → wrote {n} rows")
        total_rows += n

    print(f"Done. Total rows written: {total_rows}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
