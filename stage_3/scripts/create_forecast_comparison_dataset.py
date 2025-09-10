#!/usr/bin/env python3
"""
Create a forecast comparison dataset combining:
1. Actual observations (from OpenAQ or synthetic)
2. CAMS forecasts made 24h in advance
3. NOAA GEFS-Aerosol forecasts made 24h in advance

This creates a benchmark dataset for evaluating forecast accuracy.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def create_synthetic_observations(cities: List[str], dates: List[str]) -> pd.DataFrame:
    """
    Create synthetic 'actual' observations for the benchmark.
    In a real scenario, this would come from OpenAQ or similar.
    """
    np.random.seed(42)  # For reproducibility

    data = []
    for city in cities:
        for date_str in dates:
            # Create baseline values that vary by city
            city_factors = {
                "Berlin": {"pm25": 11.0, "pm10": 19.0, "no2": 24.0, "o3": 36.0},
                "Hamburg": {"pm25": 10.5, "pm10": 18.0, "no2": 22.0, "o3": 34.0},
                "Munich": {"pm25": 9.8, "pm10": 17.0, "no2": 20.0, "o3": 37.0},
                "München": {"pm25": 9.8, "pm10": 17.0, "no2": 20.0, "o3": 37.0},
            }

            base = city_factors.get(city, city_factors["Berlin"])

            # Add daily variation
            day_factor = 1.0 + np.random.normal(0, 0.1)

            # Add some realistic noise
            noise = np.random.normal(0, 0.8)

            data.append(
                {
                    "city": city,
                    "date": date_str,
                    "actual_pm25": max(0, (base["pm25"] * day_factor + noise)),
                    "actual_pm10": max(0, (base["pm10"] * day_factor + noise * 1.2)),
                    "actual_no2": max(0, (base["no2"] * day_factor + noise * 0.9)),
                    "actual_o3": max(0, (base["o3"] * day_factor + noise * 1.1)),
                }
            )

    df = pd.DataFrame(data)

    # Round to reasonable precision
    for col in ["actual_pm25", "actual_pm10", "actual_no2", "actual_o3"]:
        df[col] = df[col].round(2)

    log.info(
        f"Created synthetic observations: {len(df)} rows, {len(cities)} cities, {len(dates)} dates"
    )
    return df


def load_forecast_data(forecast_path: Path, provider: str) -> pd.DataFrame:
    """Load and prepare forecast data from parquet files."""
    try:
        df = pd.read_parquet(forecast_path)
        df["provider"] = provider
        log.info(f"Loaded {provider} forecast data: {len(df)} rows")
        return df
    except Exception as e:
        log.error(f"Failed to load {provider} forecast from {forecast_path}: {e}")
        return pd.DataFrame()


def create_forecast_comparison_dataset(
    config_path: Path, output_path: Path = None
) -> pd.DataFrame:
    """
    Create the main forecast comparison dataset.

    For each date, we simulate:
    - Actual observations (what actually happened)
    - CAMS forecast made 24h earlier
    - NOAA forecast made 24h earlier
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get cities from config
    cities = [city["name"] for city in config.get("cities", [])]

    # Define the date range for our benchmark
    # Using the same dates as the sample forecast data
    dates = ["2025-09-01", "2025-09-02"]

    log.info(
        f"Creating forecast comparison dataset for cities: {cities}, dates: {dates}"
    )

    # 1. Create synthetic actual observations
    actuals_df = create_synthetic_observations(cities, dates)

    # 2. Load forecast data
    processed_dir = Path(config["paths"]["processed_dir"])

    cams_df = load_forecast_data(processed_dir / "cams_forecast.parquet", "cams")
    noaa_df = load_forecast_data(
        processed_dir / "noaa_gefs_aerosol_forecast.parquet", "noaa_gefs_aerosol"
    )

    if cams_df.empty or noaa_df.empty:
        log.error("Missing forecast data. Run the ETL scripts first.")
        return pd.DataFrame()

    # 3. Prepare forecast data (simulate 24h-ahead forecasts)
    forecast_dfs = []

    for df, provider in [(cams_df, "cams"), (noaa_df, "noaa_gefs_aerosol")]:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["forecast_date"] = df["date"]  # The date this forecast is for
        df["forecast_made_date"] = (
            pd.to_datetime(df["date"]) - timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")

        # Rename columns to indicate these are forecasts
        df = df.rename(
            columns={
                "pm25": f"forecast_{provider}_pm25",
                "pm10": f"forecast_{provider}_pm10",
                "no2": f"forecast_{provider}_no2",
                "o3": f"forecast_{provider}_o3",
            }
        )

        # Select only the forecast columns for this provider
        forecast_cols = [
            col for col in df.columns if col.startswith(f"forecast_{provider}_")
        ]
        forecast_dfs.append(
            df[["city", "forecast_date", "forecast_made_date"] + forecast_cols]
        )

    # 4. Merge everything together
    result_df = actuals_df.copy()

    # Add forecast made date (24h before forecast date)
    result_df["forecast_made_date"] = (
        pd.to_datetime(result_df["date"]) - timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    for i, forecast_df in enumerate(forecast_dfs):
        # For first merge, include forecast_made_date, for subsequent merges, exclude it to avoid conflicts
        merge_cols = ["city", "date"]
        if i == 0:
            # On first merge, also check forecast_made_date matches
            result_df = result_df.merge(
                forecast_df.drop(
                    columns=["forecast_made_date"]
                ),  # Drop to avoid conflict
                left_on=["city", "date"],
                right_on=["city", "forecast_date"],
                how="left",
            )
        else:
            result_df = result_df.merge(
                forecast_df.drop(
                    columns=["forecast_made_date"]
                ),  # Drop to avoid conflict
                left_on=["city", "date"],
                right_on=["city", "forecast_date"],
                how="left",
            )

    # Clean up columns
    cols_to_drop = [col for col in result_df.columns if "forecast_date" in col]
    result_df = result_df.drop(columns=cols_to_drop, errors="ignore")

    # Add forecast lead time info
    result_df["forecast_lead_hours"] = 24

    # Reorder columns for clarity
    id_cols = ["city", "date", "forecast_made_date", "forecast_lead_hours"]
    actual_cols = [col for col in result_df.columns if col.startswith("actual_")]
    forecast_cols = [col for col in result_df.columns if col.startswith("forecast_")]

    # Only include columns that actually exist, avoiding duplicates
    available_cols = []
    seen_cols = set()

    for col_list in [id_cols, actual_cols, forecast_cols]:
        for col in col_list:
            if col in result_df.columns and col not in seen_cols:
                available_cols.append(col)
                seen_cols.add(col)

    result_df = result_df[available_cols]

    log.info(
        f"Created comparison dataset: {len(result_df)} rows, {len(result_df.columns)} columns"
    )

    # 5. Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(output_path, index=False)
        result_df.to_csv(output_path.with_suffix(".csv"), index=False)
        log.info(
            f"Saved dataset to {output_path} and {output_path.with_suffix('.csv')}"
        )

    return result_df


def calculate_forecast_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate forecast errors and accuracy metrics."""

    error_data = []

    for _, row in df.iterrows():
        base_data = {
            "city": row["city"],
            "date": row["date"],
            "forecast_lead_hours": row["forecast_lead_hours"],
        }

        # Calculate errors for each pollutant and provider
        pollutants = ["pm25", "pm10", "no2", "o3"]
        providers = ["cams", "noaa_gefs_aerosol"]

        for pollutant in pollutants:
            actual_col = f"actual_{pollutant}"

            if actual_col in row and pd.notna(row[actual_col]):
                actual_val = row[actual_col]

                for provider in providers:
                    forecast_col = f"forecast_{provider}_{pollutant}"

                    if forecast_col in row and pd.notna(row[forecast_col]):
                        forecast_val = row[forecast_col]

                        error_data.append(
                            {
                                **base_data,
                                "pollutant": pollutant,
                                "provider": provider,
                                "actual": actual_val,
                                "forecast": forecast_val,
                                "error": forecast_val - actual_val,
                                "abs_error": abs(forecast_val - actual_val),
                                "rel_error": abs(forecast_val - actual_val)
                                / max(actual_val, 0.1)
                                * 100,
                            }
                        )

    return pd.DataFrame(error_data)


def main():
    parser = argparse.ArgumentParser(description="Create forecast comparison dataset")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--output", help="Output path for the dataset")
    parser.add_argument(
        "--include-errors", action="store_true", help="Also create error analysis"
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else None

    if not output_path:
        # Default output path
        output_path = (
            config_path.parent.parent
            / "data"
            / "analysis"
            / "forecast_comparison_dataset.parquet"
        )

    # Create the main dataset
    dataset = create_forecast_comparison_dataset(config_path, output_path)

    if dataset.empty:
        log.error("Failed to create dataset")
        return 1

    # Print summary
    print("\n" + "=" * 80)
    print("FORECAST COMPARISON DATASET SUMMARY")
    print("=" * 80)
    print(f"Shape: {dataset.shape}")
    print(f"Cities: {dataset['city'].unique().tolist()}")
    print(f"Dates: {dataset['date'].unique().tolist()}")
    print(f"Forecast lead time: {dataset['forecast_lead_hours'].iloc[0]} hours")
    print("\nSample data:")
    print(dataset.head())

    # Create error analysis if requested
    if args.include_errors:
        errors_df = calculate_forecast_errors(dataset)
        if not errors_df.empty:
            error_output = output_path.with_name("forecast_errors.parquet")
            errors_df.to_parquet(error_output, index=False)
            errors_df.to_csv(error_output.with_suffix(".csv"), index=False)

            print(f"\nError analysis saved to {error_output}")
            print(f"Error summary by provider:")
            print(
                errors_df.groupby("provider")["abs_error"]
                .agg(["mean", "std", "count"])
                .round(2)
            )

    return 0


if __name__ == "__main__":
    exit(main())
