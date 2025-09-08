# scripts/providers/etl_cams_realtime.py
from __future__ import annotations

import argparse
import io
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import xarray as xr
import yaml


def _bbox_from_point_km(
    lat: float, lon: float, half_km: float
) -> Tuple[float, float, float, float]:
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(0.01, abs(pd.np.cos(pd.np.deg2rad(lat)))))  # type: ignore
    north, south = lat + dlat, lat - dlat
    west, east = lon - dlon, lon + dlon
    return (north, west, south, east)


try:
    from utils.geo import bbox_from_point_km as bbox_from_point_km  # type: ignore
except Exception:
    bbox_from_point_km = _bbox_from_point_km

LOG = logging.getLogger("etl_cams_realtime")

VAR_MAP = {
    "pm25": "particulate_matter_2.5um",
    "pm10": "particulate_matter_10um",
}


def load_config(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs(processed_dir: Path):
    processed_dir.mkdir(parents=True, exist_ok=True)


def cds_retrieve_to_bytes(dataset: str, params: Dict) -> bytes:
    try:
        import cdsapi
    except ImportError as ex:
        raise RuntimeError("cdsapi is not installed in the current environment") from ex
    c = cdsapi.Client()
    params = {**params, "format": "netcdf"}
    r = c.retrieve(dataset, params)
    bio = io.BytesIO()
    r.download(bio)
    return bio.getvalue()


def read_nc_as_df(nc_bytes: bytes) -> pd.DataFrame:
    with xr.open_dataset(io.BytesIO(nc_bytes)) as ds:
        ren = {}
        for logical, cds_name in VAR_MAP.items():
            if cds_name in ds.variables:
                ren[cds_name] = logical
        if ren:
            ds = ds.rename(ren)
        vars_present = [v for v in ["pm25", "pm10"] if v in ds.variables]
        if not vars_present:
            return pd.DataFrame(columns=["utc_datetime", "pm25", "pm10"])
        df = ds[vars_present].to_dataframe().reset_index()
        if "time" in df.columns:
            df = df.rename(columns={"time": "utc_datetime"})
        return df


def fetch_city_today(city: Dict, cfg: Dict, section: str) -> pd.DataFrame:
    cconf = cfg[section]
    dataset = cconf["dataset"]
    city_name = city["name"]
    lat, lon = float(city["lat"]), float(city["lon"])
    half_km = float(cconf["area_halfwidth_km"])
    north, west, south, east = bbox_from_point_km(lat, lon, half_km)

    cds_vars = [VAR_MAP[v] for v in cconf["variables"] if v in VAR_MAP]

    # pick days to fetch: today - backfill_days ... today
    today = datetime.utcnow().date()
    start = today - timedelta(days=int(cconf.get("backfill_days", 0)))
    days = [
        (start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((today - start).days + 1)
    ]

    recs = []
    for ymd in days:
        for hhmm in cconf["times"]:
            frt = (
                pd.to_datetime(f"{ymd} {hhmm}", utc=True)
                .tz_convert("UTC")
                .tz_localize(None)
            )
            params = {
                "date": ymd,
                "time": hhmm,
                "type": "forecast",
                "model": cconf.get("model", "ensemble"),
                "level": "0",
                "variable": cds_vars,
                "leadtime_hour": [str(h) for h in cconf["lead_hours"]],
                "area": [north, west, south, east],
            }
            try:
                nc_bytes = cds_retrieve_to_bytes(dataset, params)
            except Exception as ex:
                LOG.error(
                    "CDS realtime retrieve failed for %s %s %s: %s",
                    city_name,
                    ymd,
                    hhmm,
                    ex,
                )
                continue
            df = read_nc_as_df(nc_bytes)
            if df.empty:
                LOG.warning(
                    "Empty CAMS realtime dataframe for %s %s %s", city_name, ymd, hhmm
                )
                continue
            keep_cols = [c for c in ["pm25", "pm10"] if c in df.columns]
            g = (
                df.groupby(["utc_datetime"], as_index=False)[keep_cols]
                .mean()
                .sort_values("utc_datetime")
            )
            g["lead_hours"] = (
                (
                    (
                        pd.to_datetime(g["utc_datetime"], utc=True)
                        - pd.to_datetime(frt, utc=True)
                    ).dt.total_seconds()
                    / 3600.0
                )
                .round()
                .astype("Int64")
            )
            long = g.melt(
                id_vars=["utc_datetime", "lead_hours"],
                value_vars=keep_cols,
                var_name="pollutant",
                value_name="fcst_value",
            )
            long["city"] = city_name
            long["forecast_reference_time"] = frt
            long["source_fcst"] = "CAMS"
            long["utc_datetime"] = (
                pd.to_datetime(long["utc_datetime"], utc=True)
                .tz_convert("UTC")
                .tz_localize(None)
            )
            long["forecast_reference_time"] = pd.to_datetime(
                long["forecast_reference_time"], utc=True
            ).tz_localize(None)
            recs.append(long)
    if not recs:
        return pd.DataFrame(
            columns=[
                "city",
                "utc_datetime",
                "pollutant",
                "fcst_value",
                "forecast_reference_time",
                "lead_hours",
                "source_fcst",
            ]
        )
    out = pd.concat(recs, ignore_index=True)
    out = out[
        [
            "city",
            "utc_datetime",
            "pollutant",
            "fcst_value",
            "forecast_reference_time",
            "lead_hours",
            "source_fcst",
        ]
    ]
    out["lead_hours"] = out["lead_hours"].astype("Int64")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config/providers.yaml")
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )

    cfg = load_config(Path(args.config))
    processed_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dirs(processed_dir)

    cities = cfg.get("cities", [])
    frames = []
    for c in cities:
        LOG.info("CAMS realtime | city: %s", c["name"])
        fr = fetch_city_today(c, cfg, "cams_realtime")
        if not fr.empty:
            frames.append(fr)

    if not frames:
        LOG.error("No CAMS realtime data produced.")
        sys.exit(3)

    out = pd.concat(frames, ignore_index=True).sort_values(
        ["city", "utc_datetime", "pollutant", "lead_hours"]
    )
    outp = processed_dir / cfg["cams_realtime"]["output_file"]
    out.to_parquet(outp, index=False)
    LOG.info("Wrote %s (%d rows)", outp, len(out))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("ETL CAMS realtime failed")
        raise
