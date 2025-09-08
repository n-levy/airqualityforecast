from __future__ import annotations

"""
Stage 2 - Fetch raw data (OpenAQ v3)

What’s new/fixed:
- **Correct query params** for /v3/sensors/{id}/days: uses `date_from` / `date_to`.
- Default history 365 days (and hard-capped to ≤ 365).
- Strict client-side filtering to the requested window (belt & suspenders).
- City normalization (Munich ↔ München, umlauts/accents).
- Cap sensors per (city, parameter) group (default 1; configurable).
- Pagination guard, per-request timeout, and rate-limit backoff.
- Clear log of the effective window and options.
"""

import argparse
import os
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests import Response
from tqdm import tqdm

from common import setup_logging, load_yaml_config, ensure_dirs, RAW_DIR


# ------------------------ HTTP / utility helpers ------------------------

def _headers(api_key: Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if api_key:
        h["X-API-Key"] = api_key
    return h

def _rate_limit_sleep(resp: Response, default_seconds: int = 5) -> None:
    """Sleep according to Retry-After header if present, else a small default."""
    ra = resp.headers.get("Retry-After")
    try:
        time.sleep(int(ra) if ra else default_seconds)
    except Exception:
        time.sleep(default_seconds)

def _join(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path

def _normalize_city_key(s: str) -> str:
    """
    Normalize a city name for matching:
    - lowercase
    - replace German umlauts with ASCII digraphs (ä->ae, ö->oe, ü->ue, ß->ss)
    - strip remaining accents/diacritics
    - map common English↔local aliases (Munich -> Muenchen)
    """
    s = (s or "").strip().lower()
    s = (
        s.replace("ä", "ae")
         .replace("ö", "oe")
         .replace("ü", "ue")
         .replace("ß", "ss")
    )
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    if s == "munich":
        s = "muenchen"
    return s


# ------------------------ Discovery (locations -> sensors) ------------------------

def discover_city_sensors(
    session: requests.Session,
    base_url: str,
    iso: Optional[str],
    cities: List[str],
    parameters: List[str],
    request_timeout_s: int,
    logger,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with keys: id (sensor_id), city (locality), parameter_name, lastSeen (if present).
    Uses /v3/locations and filters sensors to requested cities & parameters.
    """
    targets_city = {_normalize_city_key(c) for c in cities}
    targets_param = {p.strip().lower() for p in parameters}

    page, limit = 1, 200
    out: List[Dict[str, Any]] = []

    logger.info(f"Discovering sensors via /v3/locations (iso={iso}, cities={cities}, parameters={parameters})")
    while True:
        params = {"limit": limit, "page": page}
        if iso:
            params["iso"] = iso
        url = _join(base_url, "/locations")

        resp = session.get(url, params=params, timeout=request_timeout_s)
        if not resp.ok:
            raise RuntimeError(f"/v3/locations failed (HTTP {resp.status_code}): {resp.text[:200]}")

        payload = resp.json()
        results = payload.get("results", [])
        if not results:
            break

        for loc in results:
            locality = (loc.get("locality") or "").strip()
            if _normalize_city_key(locality) not in targets_city:
                continue
            for s in loc.get("sensors", []) or []:
                sid = s.get("id")
                param_obj = s.get("parameter") or {}
                pname = (param_obj.get("name") or "").lower()
                if sid and pname in targets_param:
                    out.append(
                        {
                            "id": int(sid),
                            "city": locality,
                            "parameter_name": pname,
                            "lastSeen": s.get("lastSeen", {}).get("utc") if isinstance(s.get("lastSeen"), dict) else s.get("lastSeen"),
                        }
                    )

        if len(results) < limit:
            break
        page += 1

    if not out:
        raise RuntimeError("No matching sensors found in /v3/locations for the given cities/parameters.")
    logger.info(f"Discovered {len(out)} matching sensors.")
    return out


def cap_sensors_per_city_param(
    sensors: List[Dict[str, Any]],
    max_sensors: int,
    logger,
) -> List[Dict[str, Any]]:
    """
    Keep at most `max_sensors` sensors for each (city, parameter_name), preferring most recent lastSeen if available.
    """
    if max_sensors <= 0:
        return sensors

    grouped: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for s in sensors:
        key = (s["city"], s["parameter_name"])
        grouped[key].append(s)

    capped: List[Dict[str, Any]] = []
    for key, arr in grouped.items():
        # sort by lastSeen desc if present, else stable
        def _sort_key(d):
            ls = d.get("lastSeen")
            try:
                return datetime.fromisoformat(ls.replace("Z", "+00:00")) if isinstance(ls, str) else datetime.min.replace(tzinfo=timezone.utc)
            except Exception:
                return datetime.min.replace(tzinfo=timezone.utc)

        arr_sorted = sorted(arr, key=_sort_key, reverse=True)
        capped.extend(arr_sorted[:max_sensors])

    kept = len(capped)
    dropped = len(sensors) - kept
    if dropped > 0:
        logger.info(f"Capped sensors per (city,parameter): kept {kept}, dropped {dropped} (max={max_sensors}).")
    return capped


# ------------------------ Daily measurements per sensor ------------------------

def fetch_sensor_days(
    session: requests.Session,
    base_url: str,
    sensor_id: int,
    date_from_iso: str,
    date_to_iso: str,
    request_timeout_s: int,
    sensor_max_pages: int,
    logger,
) -> List[Dict[str, Any]]:
    """
    Calls /v3/sensors/{id}/days with date_from/date_to and returns a list of daily rows.
    Hard caps pages with sensor_max_pages, obeys Retry-After 429.
    """
    page, limit = 1, 1000
    rows: List[Dict[str, Any]] = []

    while True:
        if page > sensor_max_pages:
            logger.warning(f"Sensor {sensor_id}: reached page cap ({sensor_max_pages}), stopping pagination.")
            break

        url = _join(base_url, f"/sensors/{sensor_id}/days")
        params = {
            "date_from": date_from_iso,  # <-- correct param name
            "date_to": date_to_iso,      # <-- correct param name
            "limit": limit,
            "page": page,
        }
        resp = session.get(url, params=params, timeout=request_timeout_s)

        if resp.status_code == 429:
            logger.warning(f"Sensor {sensor_id}: rate limited. Backing off…")
            _rate_limit_sleep(resp)
            continue

        if not resp.ok:
            raise RuntimeError(f"/v3/sensors/{sensor_id}/days failed (HTTP {resp.status_code}): {resp.text[:200]}")

        payload = resp.json()
        result_page = payload.get("results", [])
        rows.extend(result_page)

        # pagination: stop when result count < limit
        if len(result_page) < limit:
            break
        page += 1

    return rows


# ------------------------ OpenAQ v3 main ------------------------

def fetch_from_openaq_v3(options: Dict[str, Any], logger) -> Path:
    base_url: str = options.get("base_url", "https://api.openaq.org/v3")
    api_key: str = options.get("api_key") or os.getenv("OPENAQ_API_KEY", "")
    iso: Optional[str] = options.get("iso")
    parameters: List[str] = options.get("parameters", ["pm25", "pm10", "no2", "o3"])
    cities: List[str] = options.get("cities", [])

    # Default to 365 days if not specified; always cap at 365 for performance
    days_back: int = min(int(options.get("days_back", 365)), 365)
    aggregate: str = options.get("aggregate", "daily")

    # Defaults for speed and reliability; overridable via YAML
    max_sensors_per_city_param: int = int(options.get("max_sensors_per_city_param", 1))
    request_timeout_s: int = int(options.get("request_timeout_s", 30))
    sensor_max_pages: int = int(options.get("sensor_max_pages", 30))

    if aggregate != "daily":
        raise NotImplementedError("Only 'daily' aggregation is implemented for OpenAQ v3 in Stage 2.")

    # Window
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=days_back)
    date_from_iso = date_from.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_to_iso = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Keys for fast comparisons + belt & suspenders after DataFrame creation
    date_from_key = date_from.strftime("%Y-%m-%d")
    date_to_key = date_to.strftime("%Y-%m-%d")

    logger.info(
        f"Requesting window {date_from_iso} → {date_to_iso} UTC "
        f"(days_back={days_back}, iso={iso}, cities={cities}, parameters={parameters}, "
        f"max_sensors_per_city_param={max_sensors_per_city_param}, sensor_max_pages={sensor_max_pages})"
    )

    session = requests.Session()
    session.headers.update(_headers(api_key))

    # 1) Discover sensors
    sensors = discover_city_sensors(session, base_url, iso, cities, parameters, request_timeout_s, logger)
    sensors = cap_sensors_per_city_param(sensors, max_sensors_per_city_param, logger)

    # 2) Iterate sensors with a progress bar
    disable_pbar = not sys.stdout.isatty()
    records: List[Dict[str, Any]] = []

    for s in tqdm(sensors, desc="Downloading daily data per sensor", disable=disable_pbar):
        sid = s["id"]
        city = s["city"]
        pname = s["parameter_name"]

        try:
            rows = fetch_sensor_days(session, base_url, sid, date_from_iso, date_to_iso, request_timeout_s, sensor_max_pages, logger)
        except RuntimeError as e:
            logger.error(str(e))
            continue

        for r in rows:
            # From docs: period.datetimeFrom.utc is start of the day window
            period = r.get("period") or {}
            dt_utc = (period.get("datetimeFrom") or {}).get("utc") if isinstance(period.get("datetimeFrom"), dict) else period.get("datetimeFrom")
            value = r.get("value")
            if not dt_utc or value is None:
                continue
            day_key = dt_utc[:10]  # YYYY-MM-DD
            # Strict client-side clamp to [date_from, date_to]
            if day_key < date_from_key or day_key > date_to_key:
                continue
            records.append({"city": city, "date": day_key, pname: value})

    if not records:
        raise RuntimeError("No data returned from OpenAQ v3 for requested inputs (cities/parameters/date range).")

    # 3) Build DataFrame and re-enforce the clamp (belt & suspenders)
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df = df[(df["date"] >= date_from.date()) & (df["date"] <= date_to.date())]

    # 4) Pivot to wide table per day
    df = df.groupby(["city", "date"]).agg("mean").reset_index()

    # Ensure expected pollutant columns exist
    for col in ["pm25", "pm10", "no2", "o3"]:
        if col not in df.columns:
            df[col] = pd.NA

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_csv = RAW_DIR / f"raw_air_quality_openaq_{ts}.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"Wrote OpenAQ v3 raw CSV: {out_csv} (rows={len(df)})")
    return out_csv


# ------------------------ CSV local (unchanged) ------------------------

def fetch_from_csv_local(options: Dict[str, Any], logger) -> Path:
    src_path = Path(options["path"]).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Local CSV not found: {src_path}")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dst = RAW_DIR / f"raw_air_quality_{ts}.csv"
    dst.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info(f"Copied local CSV to {dst}")
    return dst


# ------------------------ CLI ------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 - Fetch raw data (OpenAQ v3)")
    parser.add_argument("--config", type=str, required=True, help="Path to data_sources.yaml")
    args = parser.parse_args()

    logger = setup_logging("fetch_data")
    ensure_dirs()

    cfg = load_yaml_config(Path(args.config))
    active = cfg.get("active_source")
    src_cfg = cfg["sources"][active]
    provider = src_cfg["provider"]
    options = src_cfg.get("options", {})

    logger.info(f"Active source: {active} (provider: {provider})")

    if provider == "csv_local":
        fetch_from_csv_local(options, logger)
    elif provider == "openaq":
        fetch_from_openaq_v3(options, logger)
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    main()

