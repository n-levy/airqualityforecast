from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # .../stage_3
SAMPLES = ROOT / "data" / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)

# Base template (2 days × 3 cities) using the values you've seen so far
BASE_ROWS = [
    # city,     date,         pm25, pm10, no2,  o3
    ("Berlin", "2025-09-01", 12.1, 20.3, 25.0, 35.2),
    ("Berlin", "2025-09-02", 10.0, 18.5, 22.2, 40.0),
    ("Hamburg", "2025-09-01", 11.0, 17.8, 23.5, 33.0),
    ("Hamburg", "2025-09-02", 12.5, 19.1, 21.0, 41.2),
    ("München", "2025-09-01", 9.5, 16.0, 20.0, 36.0),
    ("München", "2025-09-02", 10.7, 16.9, 19.2, 38.5),
]
BASE = pd.DataFrame(BASE_ROWS, columns=["city", "date", "pm25", "pm10", "no2", "o3"])

rng = np.random.default_rng(42)


def tweak_for_provider(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    out = df.copy()
    # Provider-specific systematic differences (simple, deterministic)
    if provider == "cams":
        out["pm25"] *= 1.00
        out["pm10"] *= 1.00
        out["no2"] *= 1.00
        out["o3"] *= 1.00
        noise = rng.normal(0.0, 0.05, size=len(out))  # μg/m³-ish minor noise
    elif provider == "noaa_gefs_aerosol":
        out["pm25"] *= 1.03
        out["pm10"] *= 1.02
        out["no2"] *= 0.98
        out["o3"] *= 0.99
        noise = rng.normal(-0.10, 0.05, size=len(out))
    else:
        noise = rng.normal(0.0, 0.05, size=len(out))

    # Apply noise to all pollutant cols, keep non-negative, round nicely
    for c in ["pm25", "pm10", "no2", "o3"]:
        out[c] = (out[c] + noise).clip(lower=0).round(2)

    return out


def write_provider(provider: str, filename: str):
    dfp = tweak_for_provider(BASE, provider)
    path = SAMPLES / filename
    dfp.to_csv(path, index=False, encoding="utf-8")
    print(f"Wrote {provider} sample: {path}")


def main():
    write_provider("cams", "cams_sample.csv")
    write_provider("noaa_gefs_aerosol", "noaa_gefs_aerosol_sample.csv")


if __name__ == "__main__":
    main()
