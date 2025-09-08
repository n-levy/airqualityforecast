# scripts/align_and_join.py
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(paths_cfg):
    proc = paths_cfg["processed_dir"]
    ds = paths_cfg["datasets_dir"]
    if not proc.startswith("stage_3"):
        proc = os.path.join("stage_3", proc)
    if not ds.startswith("stage_3"):
        ds = os.path.join("stage_3", ds)
    os.makedirs(ds, exist_ok=True)
    return proc, ds


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    proc, ds_dir = resolve_paths(cfg["paths"])

    obs = pd.read_parquet(os.path.join(proc, "openaq_hourly.parquet"))
    fc = pd.read_parquet(os.path.join(proc, "cams_forecast_hourly.parquet"))

    # Normalize dtypes/keys
    obs["utc_datetime"] = pd.to_datetime(obs["utc_datetime"], utc=True).dt.floor("H")
    fc["utc_datetime"] = pd.to_datetime(fc["utc_datetime"], utc=True).dt.floor("H")
    fc["forecast_reference_time"] = pd.to_datetime(
        fc["forecast_reference_time"], utc=True
    )

    for df in (obs, fc):
        df["city"] = df["city"].astype("string")
        df["pollutant"] = (
            df["pollutant"]
            .astype("string")
            .str.lower()
            .replace({"pm2.5": "pm25", "pm2p5": "pm25"})
        )

    m = obs.merge(fc, on=["city", "utc_datetime", "pollutant"], how="inner")

    # Add static lat/lon columns from config
    coords = {c["name"]: (c["lat"], c["lon"]) for c in cfg["cities"]}
    m["lat"] = m["city"].map(lambda x: coords[x][0])
    m["lon"] = m["city"].map(lambda x: coords[x][1])

    cols = [
        "city",
        "lat",
        "lon",
        "utc_datetime",
        "pollutant",
        "obs_value",
        "fcst_value",
        "forecast_reference_time",
        "lead_hours",
        "source_obs",
        "source_fcst",
    ]
    m = (
        m[cols]
        .sort_values(
            [
                "city",
                "utc_datetime",
                "pollutant",
                "forecast_reference_time",
                "lead_hours",
            ]
        )
        .reset_index(drop=True)
    )

    outp = os.path.join(ds_dir, "city_hour_forecast_vs_obs.parquet")
    m.to_parquet(outp, index=False)
    print(f"Wrote {outp}  rows={len(m)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
