# scripts/utils/dates.py
from __future__ import annotations

import pandas as pd


def parse_date_iso(s: pd.Series) -> pd.Series:
    """
    Parse YYYY-MM-DD strings to datetime64[ns]; invalids -> NaT.
    Using an explicit format avoids the pandas 'Could not infer format' warning.
    """
    return pd.to_datetime(s.astype("string"), format="%Y-%m-%d", errors="coerce")
