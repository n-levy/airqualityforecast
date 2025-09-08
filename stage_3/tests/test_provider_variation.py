from pathlib import Path

import pandas as pd


def test_providers_not_identical_for_same_city_date():
    p = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "providers_processed"
        / "all_providers.parquet"
    )
    df = pd.read_parquet(p)

    # For each (city, date), at least one pollutant must vary across providers
    g = df.groupby(["city", "date"])
    varied = (
        g[["pm25", "pm10", "no2", "o3"]].nunique().max(axis=1)
    )  # max distinct across pollutants
    assert (
        varied >= 2
    ).all(), "Found (city,date) groups with identical values across all providers."
