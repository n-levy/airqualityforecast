#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAQ historical ETL (v3 API)

Flow (per city):
  1) /v3/locations?coordinates=lat,lon&radius=...  -> nearby location IDs
  2) /v3/locations/{location_id}/sensors           -> list sensors, filter to wanted parameters (e.g., pm25, pm10)
  3) /v3/sensors/{sensor_id}/hours                 -> pull hourly aggregates in [date_from, date_to]
  4) Combine sensors by utc hour -> mean per parameter
  5) Pivot long->wide to columns: pm25, pm10; add city
  6) Write to parquet

Config (config/providers.yaml) expected keys:
  paths:
    processed_dir: data/providers_processed
  cities:
    - name: Berlin
      lat: 52.52
      lon: 13.405
    - ...
  openaq_history:
    enabled: true
    output_file: openaq_hourly_history.parquet
    parameters: [pm25, pm10]          # optional (defaults provided)
    radius_m: 15000                   # optional
    max_locations: 50                 # optional
    date_from: "2025-07-01T00:00:00Z" # optional; can be CLI override
    date_to:   "2025-07-07T23:59:59Z" # optional; can be CLI override

CLI:
  python scripts/providers/etl_openaq_history.py --config config/providers.yaml
  # Optional overrides:
  --date-start 2025-07-01T00:00:00Z --date-end 2025-07-07T23:59:59Z --city Berlin
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
import yaml
from dateutil import parser as dtparser

# ----------------------------
# Logging
# ----------------------------
LOG = logging.getLogger("etl_openaq_history")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ----------------------------
# Constants & helpers
# ----------------------------
OPENAQ_BASE = "https://api.openaq.org/v3"
OPENAQ_API_KEY_ENV = "OPENAQ_API_KEY"

DEFAULT_PARAMETERS = ["pm25", "pm10"]
DEFAULT_RADIUS_M = 15000
DEFAULT_MAX_LOCATIONS = 50
DEFAULT_LIMIT = 100
REQUEST_TIMEOUT = 60
RETRY_COUNT = 3
RETRY_BACKOFF = 2.0  # seconds, exponential


def _get_api_key() -> str:
    key = os.environ.get(OPENAQ_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"Missing API key: set environment variable {OPENAQ_API_KEY_ENV} before running."
        )
    return key


def _headers() -> Dict[str, str]:
    return {
        "X-API-Key": _get_api_key(),
        "Accept": "application/json",
        "User-Agent": "aq-forecast-etl/1.0",
    }


def _retryable_status(status: int) -> bool:
    # Retry on transient server errors & gateway issues
    return status in (429, 500, 502, 503, 504)


def _request_json(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Single-page request helper. Raises for non-OK unless status retryable.
    """
    url = f"{OPENAQ_BASE}{path}"
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.request(
                method,
                url,
                headers=_headers(),
                params=params or {},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            if attempt <= RETRY_COUNT:
                sleep_s = RETRY_BACKOFF ** (attempt - 1)
                LOG.warning(
                    "Request error %s %s (attempt %d/%d): %s; retrying in %.1fs",
                    method,
                    path,
                    attempt,
                    RETRY_COUNT,
                    e,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            raise

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as e:
                LOG.error("Invalid JSON from %s %s: %s", method, url, e)
                raise

        if _retryable_status(resp.status_code) and attempt <= RETRY_COUNT:
            sleep_s = RETRY_BACKOFF ** (attempt - 1)
            LOG.warning(
                "Retryable HTTP %d for %s %s; retrying in %.1fs",
                resp.status_code,
                method,
                url,
                sleep_s,
            )
            time.sleep(sleep_s)
            continue

        # Non-retryable or out of attempts
        LOG.error(
            "HTTP %d at %s?%s",
            resp.status_code,
            url,
            (resp.request.body or resp.request.url),
        )
        resp.raise_for_status()


def _paged_results(
    path: str,
    params: Dict[str, Any],
    limit_param: str = "limit",
    page_param: str = "page",
    max_pages: int = 50,
) -> Iterable[Dict[str, Any]]:
    """
    Generic paginator for OpenAQ v3 endpoints supporting page/limit.

    Yields items from .results across pages, until no more or max_pages reached.
    """
    page = 1
    while page <= max_pages:
        p = dict(params)
        p[limit_param] = p.get(limit_param, DEFAULT_LIMIT)
        p[page_param] = page
        data = _request_json("GET", path, p)
        results = data.get("results", [])
        if not results:
            return
        for item in results:
            yield item
        # If fewer than limit, likely last page
        if len(results) < p[limit_param]:
            return
        page += 1


# ----------------------------
# OpenAQ v3 client functions
# ----------------------------
def find_nearby_locations(
    lat: float,
    lon: float,
    radius_m: int,
    max_locations: int,
) -> List[Dict[str, Any]]:
    """
    Returns up to max_locations location dicts near (lat, lon).
    """
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_m,
        "limit": min(DEFAULT_LIMIT, max_locations),
    }
    locs: List[Dict[str, Any]] = []
    for item in _paged_results(
        "/locations", params, max_pages=math.ceil(max_locations / DEFAULT_LIMIT) + 1
    ):
        locs.append(item)
        if len(locs) >= max_locations:
            break
    return locs


def list_sensors_for_location(location_id: int) -> List[Dict[str, Any]]:
    """
    Returns sensor dicts for a given location.
    """
    # The sensors endpoint might paginate; account for it just in case.
    params = {"limit": DEFAULT_LIMIT}
    sensors: List[Dict[str, Any]] = []
    for item in _paged_results(
        f"/locations/{location_id}/sensors", params, max_pages=50
    ):
        sensors.append(item)
    return sensors


def fetch_sensor_hours(
    sensor_id: int,
    dt_from: str,
    dt_to: str,
) -> List[Dict[str, Any]]:
    """
    Pull hourly aggregates for a sensor within [dt_from, dt_to].
    dt_from/dt_to can be date or ISO datetime; API accepts both.
    Paginates until exhausted.
    """
    params = {
        "datetime_from": dt_from,
        "datetime_to": dt_to,
        "limit": DEFAULT_LIMIT,
    }
    rows: List[Dict[str, Any]] = []
    for item in _paged_results(f"/sensors/{sensor_id}/hours", params, max_pages=200):
        rows.append(item)
    return rows


# ----------------------------
# Data shaping helpers
# ----------------------------
def normalize_hours_rows(
    rows: List[Dict[str, Any]],
    location_id: Optional[int] = None,
    sensor_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convert raw hour rows to a tidy DF with columns:
      utc_datetime, parameter, value, location_id, sensor_id
    """
    if not rows:
        return pd.DataFrame(
            columns=["utc_datetime", "parameter", "value", "location_id", "sensor_id"]
        )

    recs = []
    for r in rows:
        period = r.get("period") or {}
        # Prefer explicit UTC if present; fall back to 'datetimeFrom' string
        utc = None
        if "datetimeFrom" in period:
            # Shapes observed:
            #  - {"utc": "2025-07-01T00:00:00Z", "local": "..."} OR string
            dtf = period["datetimeFrom"]
            if isinstance(dtf, dict):
                utc = dtf.get("utc")
            elif isinstance(dtf, str):
                utc = dtf
        if not utc and "datetime" in r:
            # older shapes or alternative fields
            utc = r["datetime"]

        param = None
        p = r.get("parameter") or {}
        # 'parameter' can be dict with 'name' and 'units'
        if isinstance(p, dict):
            param = p.get("name") or p.get("id")
        elif isinstance(p, str):
            param = p

        value = r.get("value")
        if utc and param is not None and value is not None:
            recs.append(
                {
                    "utc_datetime": utc,
                    "parameter": str(param).lower(),
                    "value": value,
                    "location_id": location_id,
                    "sensor_id": sensor_id,
                }
            )

    df = pd.DataFrame.from_records(recs)
    if not df.empty:
        # Parse timestamp and normalize to UTC
        df["utc_datetime"] = pd.to_datetime(df["utc_datetime"], utc=True)
        # De-duplicate (in case overlapping pages or sensors return same period)
        df = df.drop_duplicates(subset=["utc_datetime", "parameter", "sensor_id"])
    return df


def long_to_city_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input long DF columns: utc_datetime, parameter, value, [city]
    Output wide DF per city: utc_datetime, pm25, pm10, city
    Sensors are averaged per hour by parameter.
    """
    if df.empty:
        return df
    # Compute mean across sensors per city/utc/parameter
    grp = (
        df.groupby(["city", "utc_datetime", "parameter"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "mean_value"})
    )
    wide = grp.pivot_table(
        index=["city", "utc_datetime"],
        columns="parameter",
        values="mean_value",
        aggfunc="mean",
    )
    wide = wide.reset_index()
    # Flatten columns (pivot adds a name)
    wide.columns.name = None

    # Ensure expected columns exist (even if NaN)
    for col in DEFAULT_PARAMETERS:
        if col not in wide.columns:
            wide[col] = pd.NA

    # Order columns
    cols = ["city", "utc_datetime"] + sorted(
        [c for c in wide.columns if c not in ("city", "utc_datetime")]
    )
    wide = wide[cols].sort_values(["city", "utc_datetime"]).reset_index(drop=True)
    return wide


# ----------------------------
# Main ETL
# ----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OpenAQ historical ETL (v3)")
    ap.add_argument("--config", required=True, help="Path to providers.yaml")
    ap.add_argument(
        "--date-start",
        help="Override start datetime (ISO 8601, e.g., 2025-07-01T00:00:00Z)",
    )
    ap.add_argument("--date-end", help="Override end datetime (ISO 8601)")
    ap.add_argument("--city", help="Run only a single city by name (exact match)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    paths = cfg.get("paths", {})
    processed_dir = paths.get("processed_dir", "data/providers_processed")
    os.makedirs(processed_dir, exist_ok=True)

    cities = cfg.get("cities", [])
    if args.city:
        cities = [c for c in cities if c.get("name") == args.city]
        if not cities:
            LOG.error("City %s not found in config.", args.city)
            return 2

    provider_cfg = cfg.get("openaq_history") or {}
    if not provider_cfg or not provider_cfg.get("enabled", True):
        LOG.info("OpenAQ history not enabled in config; nothing to do.")
        return 0

    date_from = args.date_start or provider_cfg.get("date_from")
    date_to = args.date_end or provider_cfg.get("date_to")
    if not date_from or not date_to:
        LOG.error(
            "Both date_from and date_to are required (in config or CLI overrides)."
        )
        return 2

    # Validate date strings
    try:
        _ = dtparser.isoparse(date_from)
        _ = dtparser.isoparse(date_to)
    except Exception as e:
        LOG.error("Invalid date_from/date_to: %s", e)
        return 2

    parameters = [
        str(p).lower() for p in provider_cfg.get("parameters", DEFAULT_PARAMETERS)
    ]
    radius_m = int(provider_cfg.get("radius_m", DEFAULT_RADIUS_M))
    max_locations = int(provider_cfg.get("max_locations", DEFAULT_MAX_LOCATIONS))
    output_file = provider_cfg.get("output_file", "openaq_hourly_history.parquet")
    out_path = os.path.join(processed_dir, output_file)

    LOG.info(
        "OpenAQ history | cities: %d | %s → %s | params=%s | radius=%dm | max_locations=%d",
        len(cities),
        date_from,
        date_to,
        parameters,
        radius_m,
        max_locations,
    )

    all_long_rows: List[pd.DataFrame] = []

    for c in cities:
        name = c.get("name")
        lat = float(c.get("lat"))
        lon = float(c.get("lon"))
        LOG.info("OpenAQ history | city: %s | %s → %s", name, date_from, date_to)

        # 1) Nearby locations
        locs = find_nearby_locations(lat, lon, radius_m, max_locations)
        if not locs:
            LOG.warning(
                "No OpenAQ locations found near %s (lat=%.4f lon=%.4f)", name, lat, lon
            )
            continue

        # 2) Sensors per location
        city_frames: List[pd.DataFrame] = []
        for loc in locs:
            loc_id = loc.get("id")
            if loc_id is None:
                continue
            sensors = list_sensors_for_location(loc_id)
            # 2a) Filter by parameter names
            wanted_sensors = []
            for s in sensors:
                p = (s.get("parameter") or {}).get("name")
                if p and str(p).lower() in parameters:
                    wanted_sensors.append(s)

            if not wanted_sensors:
                continue

            # 3) Hours per sensor
            for s in wanted_sensors:
                sid = s.get("id")
                if sid is None:
                    continue
                rows = fetch_sensor_hours(sid, date_from, date_to)
                if not rows:
                    continue
                df = normalize_hours_rows(rows, location_id=loc_id, sensor_id=sid)
                if not df.empty:
                    df["city"] = name
                    city_frames.append(df)

        if not city_frames:
            LOG.warning("OpenAQ returned no rows for city %s", name)
            continue

        city_long = pd.concat(city_frames, ignore_index=True)
        all_long_rows.append(city_long)

    if not all_long_rows:
        LOG.error("No OpenAQ history data produced.")
        return 3

    long_df = pd.concat(all_long_rows, ignore_index=True)

    # Keep only requested parameters
    long_df = long_df[long_df["parameter"].isin(parameters)]

    wide_df = long_to_city_wide(long_df)

    # Final sanity checks
    if wide_df.empty:
        LOG.error("OpenAQ produced rows but aggregation yielded empty DataFrame.")
        return 3

    # Ensure parquet engine
    parquet_kwargs: Dict[str, Any] = {}
    try:
        import pyarrow  # noqa: F401

        parquet_kwargs["engine"] = "pyarrow"
    except Exception:
        parquet_kwargs["engine"] = "auto"  # fallback to fastparquet if installed

    wide_df.to_parquet(out_path, index=False, **parquet_kwargs)
    LOG.info("Wrote %d rows to %s", len(wide_df), out_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")
        sys.exit(130)
