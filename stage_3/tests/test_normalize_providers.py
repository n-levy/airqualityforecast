import importlib
from pathlib import Path

import pandas as pd

# Load each ETL module directly to access normalize()
mods = {
    "cams": "scripts.providers.etl_cams",
    "aurora": "scripts.providers.etl_aurora",
    "noaa": "scripts.providers.etl_noaa_gefs_aerosol",
}


def _normalize(mod_name):
    return importlib.import_module(mod_name).normalize


def test_normalize_happy_path():
    df = pd.DataFrame(
        {
            "city": ["Berlin", "Hamburg"],
            "date": ["2025-09-01", "2025-09-02"],
            "pm25": [10, 12.5],
            "pm10": [20, 19],
            "no2": [22, 21],
            "o3": [40, 41],
            "extra": [1, 2],
        }
    )
    for name, mod in mods.items():
        norm = _normalize(mod)
        out = norm(df.copy())
        assert list(out.columns) == ["city", "date", "pm25", "pm10", "no2", "o3"]
        assert out.shape[0] == 2
        assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_normalize_fills_missing_columns_and_drops_bad_rows():
    # Make the second row FULLY valid so exactly one row survives
    df = pd.DataFrame(
        {
            "city": ["Berlin", "Hamburg"],  # second row now valid city
            "date": ["bad-date", "2025-09-02"],  # first row invalid date
            "pm25": [None, 12.0],  # coercible
        }
    )
    for name, mod in mods.items():
        norm = _normalize(mod)
        out = norm(df.copy())
        # only the second (valid) row should survive
        assert out.shape[0] == 1
        assert out.iloc[0]["city"] == "Hamburg"
        assert pd.notna(out.iloc[0]["date"])
