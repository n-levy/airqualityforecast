from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REQUIRED = {"city", "date", "pm25", "pm10", "no2", "o3"}


def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {p}")
        return 2

    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"ERROR: could not read parquet: {e}")
        return 2

    missing = REQUIRED - set(df.columns)
    if missing:
        print(f"ERROR: missing columns: {missing}")
        return 3

    if df.empty:
        print("ERROR: dataframe is empty")
        return 4

    # Type sanity (best-effort)
    try:
        # Coerce and check
        df["city"] = df["city"].astype("string")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        num_cols = ["pm25", "pm10", "no2", "o3"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Basic NA checks
        if df["city"].isna().any():
            print("ERROR: 'city' has NA values after coercion")
            return 5
        if df["date"].isna().any():
            print("ERROR: 'date' has NA values after coercion")
            return 5

        # Reasonable numeric content (not all NA)
        for c in num_cols:
            if df[c].isna().all():
                print(f"ERROR: all values in '{c}' are NA")
                return 6

        # Non-negative check (typical for concentrations)
        for c in num_cols:
            if (df[c] < 0).any():
                print(f"ERROR: negative values found in '{c}'")
                return 7

    except Exception as e:
        print(f"ERROR: validation exception: {e}")
        return 9

    print(f"OK: {p} rows={len(df)} cols={list(df.columns)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_provider.py <path_to_parquet>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
