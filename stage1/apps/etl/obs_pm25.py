# stage1/apps/etl/obs_pm25.py
"""
Create curated hourly PM2.5 observations for a city.

- Reads `config/cities/<city>.yml` for station metadata.
- Generates a city-level hourly series (mean of stations) in UTC.
- Writes Parquet partitioned by `date=YYYY-MM-DD`:
    %DATA_ROOT%/curated/obs/<city>/pm25/date=YYYY-MM-DD/data.parquet

By default we **fake** station observations so you can run fully offline.
To fake, set:
  PowerShell:    $env:OBS_FAKE = "1"
To attempt real pulls (stub you can extend):
  PowerShell:    $env:OBS_FAKE = "0"

Usage (Windows PowerShell):
  C:\aqf311\.venv\Scripts\python.exe apps\etl\obs_pm25.py --city berlin --since 2025-07-01 --until 2025-07-07
"""
from __future__ import annotations

import argparse, os, sys, math, json, random
from pathlib import Path
from typing import Iterable, List
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import yaml
from dateutil import parser as dtparse
from dotenv import load_dotenv

# -----------------------
# Helpers / environment
# -----------------------
def load_env() -> dict:
    # Load .env if present (same folder as script or project root)
    for candidate in [Path(".env"), Path(__file__).resolve().parents[2] / ".env"]:
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
    home = Path(os.getenv("USERPROFILE") or os.getenv("HOME") or ".")
    return {
        "DATA_ROOT": Path(os.getenv("DATA_ROOT") or (home / "stage1_data")),
        "LOGS_ROOT": Path(os.getenv("LOGS_ROOT") or (home / "stage1_logs")),
    }

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def rfc3339_utc(ts: pd.Series | Iterable[datetime]) -> pd.Series:
    """Ensure pandas UTC tz-aware datetimes."""
    s = pd.to_datetime(ts, utc=True)
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s

def daterange_utc(start: datetime, end: datetime) -> Iterable[datetime]:
    """Yield UTC midnights from start..end inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

# -----------------------
# City & raw generation
# -----------------------
def load_city_config(city: str) -> dict:
    path = Path("config") / "cities" / f"{city}.yml"
    if not path.exists():
        raise FileNotFoundError(f"City config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def generate_fake_station_hourly(city_cfg: dict, day_utc: datetime) -> pd.DataFrame:
    """
    Produce fake station-level PM2.5 per hour for a given UTC day.
    Shape: columns ['station_id','ts','value','unit']
    """
    rng = pd.date_range(day_utc, day_utc + timedelta(days=1), freq="1h", inclusive="left", tz="UTC")
    rows = []
    base = 12.0
    # Slight day-of-week effect
    dow = day_utc.weekday()
    base += {5: 4.0, 6: 4.0}.get(dow, 0.0)  # weekend

    for st in city_cfg["stations"]:
        sid = st["id"]
        phase = random.random() * 2 * math.pi
        noise = np.random.default_rng(abs(hash(sid)) % 2**32)
        for i, ts in enumerate(rng):
            diurnal = 6.0 * (1 + math.sin((i/24.0)*2*math.pi + phase))  # morning/evening bumps
            val = max(0.1, base + diurnal + noise.normal(0, 1.2))
            rows.append({"station_id": sid, "ts": ts, "value": round(val, 2), "unit": "µg/m³"})
    df = pd.DataFrame(rows)
    # as strings for "raw" style compatibility if ever exported
    return df

def fetch_uba_pm25(city_cfg: dict, day_utc: datetime) -> pd.DataFrame:
    """
    Placeholder for a real fetcher. For now we always call the fake generator.
    Extend with OpenAQ/UBA/EPA as needed.
    """
    return generate_fake_station_hourly(city_cfg, day_utc)

# -----------------------
# Curate (city-level)
# -----------------------
def curate_city_hourly(city: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert station-level hourly observations to curated city hourly means.
    Output columns: ['city','valid_time','value','unit'] with valid_time tz-aware UTC.
    """
    if raw_df.empty:
        return pd.DataFrame(columns=["city","valid_time","value","unit"])

    # Ensure UTC timezone on timestamps
    ts = rfc3339_utc(raw_df["ts"])
    df = raw_df.copy()
    df["ts"] = ts

    # Aggregate mean over stations per hour
    g = df.groupby("ts", as_index=False).agg(value=("value","mean"))
    g["city"] = city
    g["unit"] = "µg/m³"
    g = g.rename(columns={"ts":"valid_time"})

    # Sort and enforce dtype
    g = g[["city","valid_time","value","unit"]].sort_values("valid_time").reset_index(drop=True)
    # Make sure pandas dtype is datetime64[ns, UTC]
    g["valid_time"] = pd.to_datetime(g["valid_time"], utc=True)
    return g

# -----------------------
# Write Parquet partitioned
# -----------------------
def write_partitioned_parquet(curated: pd.DataFrame, data_root: Path, city: str) -> int:
    if curated.empty:
        return 0
    # Partition by calendar date (UTC)
    curated = curated.copy()
    curated["date"] = curated["valid_time"].dt.strftime("%Y-%m-%d")

    total = 0
    for date_str, part in curated.groupby("date"):
        out_dir = data_root / "curated" / "obs" / city / "pm25" / f"date={date_str}"
        ensure_dir(out_dir)
        out_path = out_dir / "data.parquet"
        part = part.drop(columns=["date"])
        part.to_parquet(out_path, index=False)
        total += len(part)
    return total

# -----------------------
# CLI
# -----------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Curate hourly PM2.5 observations per city.")
    ap.add_argument("--city", required=True, choices=["berlin","hamburg","munich"], help="City key (lowercase)")
    ap.add_argument("--since", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--until", required=True, help="End date inclusive (YYYY-MM-DD)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    return ap.parse_args(argv)

def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    env = load_env()
    data_root = env["DATA_ROOT"]
    ensure_dir(data_root)

    city_cfg = load_city_config(args.city)
    since = dtparse.parse(args.since).date()
    until = dtparse.parse(args.until).date()
    if until < since:
        raise SystemExit("--until must be >= --since")

    total_rows = 0
    for d in pd.date_range(since, until, freq="1D"):
        day_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        raw = fetch_uba_pm25(city_cfg, day_utc)
        curated = curate_city_hourly(args.city, raw)
        n = write_partitioned_parquet(curated, data_root, args.city)
        if args.verbose:
            print(f"{args.city} {day_utc.date()} → wrote {n} rows")
        total_rows += n

    print(f"Done. Total curated rows written: {total_rows}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
