#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAQ historical ETL (v3 API) — Windows-friendly, verbose, and controllable runtime.

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
  openaq_history:
    enabled: true
    output_file: openaq_hourly_history.parquet
    parameters: [pm25, pm10]          # optional
    radius_m: 15000                   # optional
    max_locations: 50                 # optional (default if CLI not provided)
    date_from: "2025-07-01T00:00:00Z" # optional; can be CLI override
    date_to:   "2025-07-07T23:59:59Z" # optional; can be CLI override
    # Optional API key sources (if you don't want to use env var):
    # api_key: "your-key-here"
    # api_key_file: "C:/path/to/openaq.key"

CLI examples (PowerShell):
  # tiny smoke test for speed:
  python .\scripts\providers\etl_openaq_history.py --config .\config\providers.yaml `
    --city Berlin --date-start 2025-07-01T00:00:00Z --date-end 2025-07-02T23:59:59Z `
    --max-locations 3 --max-sensors-per-location 2 --locations-pages 1

Env toggles:
  $env:AQ_LOG_DEBUG = "1"     # DEBUG logs from this script and urllib3
  $env:AQ_HTTP_TIMEOUT = "60" # per-request timeout (sec)
  $env:AQ_PAGE_LIMIT  = "100" # OpenAQ page size for pagination
  $env:AQ_TLS_VERIFY  = "1"   # "0" to disable TLS verify for diagnosis only
  $env:OPENAQ_API_KEY = "<YOUR-KEY>"  # if not using api_key/api_key_file in YAML
"""

from __future__ import annotations

import argparse
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------
# Logging
# ----------------------------
LOG = logging.getLogger("etl_openaq_history")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Enable DEBUG logs if env var AQ_LOG_DEBUG=1 is present
if os.environ.get("AQ_LOG_DEBUG") == "1":
    LOG.setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)


# ----------------------------
# Constants & env overrides
# ----------------------------
OPENAQ_BASE = "https://api.openaq.org/v3"
OPENAQ_API_KEY_ENV = "OPENAQ_API_KEY"

DEFAULT_PARAMETERS = ["pm25", "pm10"]
DEFAULT_RADIUS_M = 15000

# Defaults (can be overridden via YAML/CLI/env as described)
DEFAULT_LIMIT = int(os.environ.get("AQ_PAGE_LIMIT", "100"))  # per-page items
REQUEST_TIMEOUT = int(os.environ.get("AQ_HTTP_TIMEOUT", "60"))  # seconds

# Retry strategy — robust to DNS/connect/read/status flakes
RETRY_TOTAL = 8
RETRY_CONNECT = 8
RETRY_READ = 4
RETRY_STATUS = 4
RETRY_BACKOFF = 1.5  # seconds backoff factor (exponential)

# TLS verification (1 default). Set AQ_TLS_VERIFY=0 to temporarily disable for corp SSL debugging.
_TLS_VERIFY_ENV = os.environ.get("AQ_TLS_VERIFY", "1").strip()
TLS_VERIFY: bool | str
if _TLS_VERIFY_ENV in ("0", "false", "False", "FALSE"):
    TLS_VERIFY = False
else:
    TLS_VERIFY = True

# Global API key & session (set in main)
API_KEY: Optional[str] = None
SESSION: Optional[requests.Session] = None


# ----------------------------
# Helpers for API key handling
# ----------------------------
def _read_file_strip(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def set_api_key_from_env_or_cfg(provider_cfg: Dict[str, Any]) -> str:
    """
    Resolve API key from:
      1) Environment variable OPENAQ_API_KEY
      2) providers.yaml -> openaq_history.api_key
      3) providers.yaml -> openaq_history.api_key_file (path)
    """
    key = os.environ.get(OPENAQ_API_KEY_ENV, "").strip()
    if not key:
        key = (provider_cfg or {}).get("api_key", "") or ""
    if not key:
        key_file = (provider_cfg or {}).get("api_key_file")
        if key_file:
            try:
                key = _read_file_strip(key_file)
            except Exception as e:
                LOG.error("Failed to read api_key_file '%s': %s", key_file, e)
                key = ""
    if not key:
        raise RuntimeError(
            "Missing OpenAQ API key. Set env OPENAQ_API_KEY, or put 'api_key'/'api_key_file' under openaq_history in providers.yaml."
        )
    return key


def _headers() -> Dict[str, str]:
    if not API_KEY:
        raise RuntimeError(
            "API_KEY not initialized. Call set_api_key_from_env_or_cfg() in main() first."
        )
    return {
        "X-API-Key": API_KEY,
        "Accept": "application/json",
        "User-Agent": "aq-forecast-etl/1.3",
    }


def _build_session() -> requests.Session:
    """
    Build a Session with robust retries for:
      - DNS/connect errors
      - Read timeouts
      - HTTP 429/5xx status codes
    """
    s = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        connect=RETRY_CONNECT,
        read=RETRY_READ,
        status=RETRY_STATUS,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    # If corporate proxies exist in env (HTTPS_PROXY/HTTP_PROXY), requests will honor them automatically.
    return s


def _retryable_status(code: int) -> bool:
    # Kept for clarity in logs; actual status retries are handled by Retry above.
    return code in (429, 500, 502, 503, 504)


def _request_json(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Single-page request helper using the global SESSION.
    """
    if SESSION is None:
        raise RuntimeError(
            "HTTP SESSION not initialized. Call _build_session() in main()."
        )
    url = f"{OPENAQ_BASE}{path}"

    LOG.debug("HTTP %s %s params=%s", method, path, params)
    resp = SESSION.request(
        method,
        url,
        headers=_headers(),
        params=params or {},
        timeout=REQUEST_TIMEOUT,
        verify=TLS_VERIFY,
    )
    LOG.debug("HTTP %s %s -> %s", method, path, resp.status_code)

    if resp.status_code == 200:
        try:
            return resp.json()
        except ValueError as e:
            LOG.error("Invalid JSON from %s %s: %s", method, url, e)
            raise

    # At this point Retry has already done its part; surface the error cleanly:
    LOG.error("HTTP %d at %s (params=%s)", resp.status_code, url, params)
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
    Yields items from .results across pages.
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
    locations_pages: int,
) -> List[Dict[str, Any]]:
    """
    Returns up to max_locations location dicts near (lat, lon).
    locations_pages controls how many paged requests we make.
    """
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_m,
        "limit": min(DEFAULT_LIMIT, max_locations),
    }
    locs: List[Dict[str, Any]] = []
    for item in _paged_results("/locations", params, max_pages=max(1, locations_pages)):
        locs.append(item)
        if len(locs) >= max_locations:
            break
    return locs


def list_sensors_for_location(location_id: int) -> List[Dict[str, Any]]:
    """
    Returns sensor dicts for a given location.
    """
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
        utc = None
        if "datetimeFrom" in period:
            dtf = period["datetimeFrom"]
            if isinstance(dtf, dict):
                utc = dtf.get("utc")
            elif isinstance(dtf, str):
                utc = dtf
        if not utc and "datetime" in r:
            utc = r["datetime"]

        p = r.get("parameter") or {}
        if isinstance(p, dict):
            param = p.get("name") or p.get("id")
        else:
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
        df["utc_datetime"] = pd.to_datetime(df["utc_datetime"], utc=True)
        df = df.drop_duplicates(subset=["utc_datetime", "parameter", "sensor_id"])
    return df


def long_to_city_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input long DF columns: utc_datetime, parameter, value, city
    Output wide DF per city: utc_datetime, pm25, pm10, city
    Mean across sensors per hour per parameter.
    """
    if df.empty:
        return df

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
    wide.columns.name = None

    # Ensure expected columns exist
    for col in DEFAULT_PARAMETERS:
        if col not in wide.columns:
            wide[col] = pd.NA

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
    ap.add_argument("--config", required=True, help="Path to config/providers.yaml")
    ap.add_argument(
        "--date-start",
        help="Override start datetime (ISO 8601, e.g., 2025-07-01T00:00:00Z)",
    )
    ap.add_argument("--date-end", help="Override end datetime (ISO 8601)")
    ap.add_argument("--city", help="Run only a single city by name (exact match)")
    ap.add_argument(
        "--max-locations",
        type=int,
        help="Cap number of locations per city (overrides YAML)",
    )
    ap.add_argument(
        "--max-sensors-per-location",
        type=int,
        default=None,
        help="Cap number of sensors per location after parameter filter",
    )
    ap.add_argument(
        "--locations-pages",
        type=int,
        default=1,
        help="How many pages of locations to scan (default: 1 for speed)",
    )
    return ap.parse_args()


def main() -> int:
    global API_KEY, SESSION

    t0 = time.time()
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

    # Resolve API key now and keep it global for headers()
    API_KEY = set_api_key_from_env_or_cfg(provider_cfg)

    # Build HTTP session with robust retries
    SESSION = _build_session()

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
    # YAML default if CLI not provided:
    yaml_max_locations = int(provider_cfg.get("max_locations", 50))
    max_locations = (
        int(args.max_locations) if args.max_locations else yaml_max_locations
    )
    max_sensors_per_location = args.max_sensors_per_location  # None = unlimited
    locations_pages = max(1, int(args.locations_pages or 1))

    output_file = provider_cfg.get("output_file", "openaq_hourly_history.parquet")
    out_path = os.path.join(processed_dir, output_file)

    LOG.info(
        "OpenAQ history | cities: %d | %s → %s | params=%s | radius=%dm | max_locations=%d | sensor_cap/location=%s | loc_pages=%d",
        len(cities),
        date_from,
        date_to,
        parameters,
        radius_m,
        max_locations,
        (
            str(max_sensors_per_location)
            if max_sensors_per_location is not None
            else "∞"
        ),
        locations_pages,
    )

    all_long_rows: List[pd.DataFrame] = []

    for c in cities:
        c_start = time.time()
        name = c.get("name")
        lat = float(c.get("lat"))
        lon = float(c.get("lon"))
        LOG.info("OpenAQ history | city: %s | %s → %s", name, date_from, date_to)

        # 1) Nearby locations
        locs = find_nearby_locations(lat, lon, radius_m, max_locations, locations_pages)
        LOG.info(
            "City %s: %d nearby locations fetched (limit=%d, pages=%d)",
            name,
            len(locs),
            max_locations,
            locations_pages,
        )
        if not locs:
            LOG.warning(
                "No OpenAQ locations found near %s (lat=%.4f lon=%.4f)", name, lat, lon
            )
            continue

        # 2) Sensors per location
        city_frames: List[pd.DataFrame] = []
        total_sensor_calls = 0

        for idx, loc in enumerate(locs, start=1):
            loc_id = loc.get("id")
            loc_name = loc.get("name") or f"loc_{loc_id}"
            if loc_id is None:
                continue

            sensors = list_sensors_for_location(loc_id)
            LOG.info(
                "City %s: location %s (%d/%d) has %d sensors (pre-filter)",
                name,
                loc_name,
                idx,
                len(locs),
                len(sensors),
            )

            # Filter sensors by parameter names
            wanted_sensors = []
            for s in sensors:
                p = (s.get("parameter") or {}).get("name")
                if p and str(p).lower() in parameters:
                    wanted_sensors.append(s)

            if not wanted_sensors:
                LOG.debug(
                    "City %s: location %s has no sensors for %s",
                    name,
                    loc_name,
                    parameters,
                )
                continue

            if max_sensors_per_location is not None:
                wanted_sensors = wanted_sensors[: max(0, max_sensors_per_location)]

            LOG.info(
                "City %s: location %s -> fetching %d sensors after filter/cap",
                name,
                loc_name,
                len(wanted_sensors),
            )

            # 3) Hours per sensor
            for si, s in enumerate(wanted_sensors, start=1):
                sid = s.get("id")
                if sid is None:
                    continue
                total_sensor_calls += 1
                rows = fetch_sensor_hours(sid, date_from, date_to)
                df = normalize_hours_rows(rows, location_id=loc_id, sensor_id=sid)
                LOG.debug(
                    "City %s: loc %s sensor %s (%d/%d) -> %d rows",
                    name,
                    loc_name,
                    sid,
                    si,
                    len(wanted_sensors),
                    len(df),
                )
                if not df.empty:
                    df["city"] = name
                    city_frames.append(df)

        if not city_frames:
            LOG.warning("OpenAQ returned no rows for city %s", name)
            continue

        city_long = pd.concat(city_frames, ignore_index=True)
        LOG.info(
            "City %s: total sensors fetched=%d, rows=%d (elapsed %.1fs)",
            name,
            total_sensor_calls,
            len(city_long),
            time.time() - c_start,
        )
        all_long_rows.append(city_long)

    if not all_long_rows:
        LOG.error("No OpenAQ history data produced.")
        return 3

    long_df = pd.concat(all_long_rows, ignore_index=True)
    LOG.info("All cities combined: long rows=%d", len(long_df))

    # Keep only requested parameters
    long_df = long_df[long_df["parameter"].isin(parameters)]

    wide_df = long_to_city_wide(long_df)
    LOG.info(
        "All cities combined: wide rows=%d, cols=%s",
        len(wide_df),
        list(wide_df.columns),
    )

    if wide_df.empty:
        LOG.error("OpenAQ produced rows but aggregation yielded empty DataFrame.")
        return 3

    # Choose parquet engine
    parquet_kwargs: Dict[str, Any] = {}
    try:
        import pyarrow  # noqa: F401

        parquet_kwargs["engine"] = "pyarrow"
    except Exception:
        parquet_kwargs["engine"] = "auto"

    wide_df.to_parquet(out_path, index=False, **parquet_kwargs)
    LOG.info(
        "Wrote %d rows to %s (total elapsed %.1fs)",
        len(wide_df),
        out_path,
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")
        sys.exit(130)
