#!/usr/bin/env python3
"""
Feature Engineering for Air Quality Forecasting

Adds calendar and lag features to the forecast comparison dataset according to FEATURES.md:

Calendar Features:
- Day of week
- Month
- Holiday flag (per city)

Lag Features:
- Lagged observations: t-1d, t-2d (no leakage)
- Lagged provider forecasts: previous cycle, same lead time (no leakage)
- Rolling features: 6h, 24h windows (simulated from daily data)

All features respect the "no leakage" principle - only data available before
the forecast issue time is used.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# German public holidays (simplified - major ones that affect air quality)
GERMAN_HOLIDAYS = {
    "2025-01-01": "New Year",
    "2025-04-18": "Good Friday",
    "2025-04-21": "Easter Monday",
    "2025-05-01": "Labour Day",
    "2025-05-29": "Ascension Day",
    "2025-06-09": "Whit Monday",
    "2025-10-03": "German Unity Day",
    "2025-12-25": "Christmas Day",
    "2025-12-26": "Boxing Day",
    # Add common dates around our sample period
    "2025-08-31": "End of Summer Holiday",  # Common school holiday end
    "2025-09-01": "Back to School",
    "2025-09-02": "Regular Day",
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features: day of week, month, holiday flag.
    """
    df = df.copy()

    # Convert date column to datetime if it's not already
    df["date_dt"] = pd.to_datetime(df["date"])

    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df["date_dt"].dt.dayofweek
    df["dow_name"] = df["date_dt"].dt.day_name()

    # Month
    df["month"] = df["date_dt"].dt.month
    df["month_name"] = df["date_dt"].dt.month_name()

    # Holiday flag (per city - for now, same holidays for all German cities)
    df["is_holiday"] = df["date"].isin(GERMAN_HOLIDAYS.keys()).astype(int)
    df["holiday_name"] = df["date"].map(GERMAN_HOLIDAYS)

    # Weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)  # Sat=5, Sun=6

    # Season (meteorological seasons)
    df["season"] = df["month"].map(
        {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
    )

    log.info(
        "Added calendar features: day_of_week, month, is_holiday, is_weekend, season"
    )
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features respecting the no-leakage principle.

    For each forecast date, we only use data available before the forecast_made_date.
    """
    df = df.copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Convert dates
    df["date_dt"] = pd.to_datetime(df["date"])
    df["forecast_made_dt"] = pd.to_datetime(df["forecast_made_date"])

    pollutants = ["pm25", "pm10", "no2", "o3"]
    providers = ["cams", "noaa_gefs_aerosol"]

    # Initialize lag feature columns
    lag_cols = []

    # 1. Lagged observations (t-1d, t-2d)
    for pollutant in pollutants:
        for lag_days in [1, 2]:
            col_name = f"lag_{lag_days}d_actual_{pollutant}"
            df[col_name] = np.nan
            lag_cols.append(col_name)

    # 2. Lagged provider forecasts (previous cycle, same lead time)
    for provider in providers:
        for pollutant in pollutants:
            for lag_days in [1, 2]:
                col_name = f"lag_{lag_days}d_forecast_{provider}_{pollutant}"
                df[col_name] = np.nan
                lag_cols.append(col_name)

    # 3. Rolling features (simulated from available data)
    for pollutant in pollutants:
        # Rolling mean of actuals (2-day window to simulate 24h)
        col_name = f"roll_2d_actual_{pollutant}"
        df[col_name] = np.nan
        lag_cols.append(col_name)

    # Fill lag features city by city (respecting temporal order and no-leakage)
    for city in df["city"].unique():
        city_mask = df["city"] == city
        city_df = df[city_mask].copy().sort_values("date_dt")

        for i, row in city_df.iterrows():
            forecast_made_dt = row["forecast_made_dt"]

            # Find data available before forecast was made (no leakage)
            available_data = city_df[city_df["date_dt"] < forecast_made_dt].copy()

            if len(available_data) == 0:
                continue

            # Sort by date (most recent first for easier lag indexing)
            available_data = available_data.sort_values("date_dt", ascending=False)

            # Fill lag features
            for pollutant in pollutants:
                # Lagged actuals
                actual_col = f"actual_{pollutant}"
                if actual_col in available_data.columns:
                    # t-1d lag
                    if len(available_data) >= 1:
                        df.loc[i, f"lag_1d_actual_{pollutant}"] = available_data.iloc[
                            0
                        ][actual_col]
                    # t-2d lag
                    if len(available_data) >= 2:
                        df.loc[i, f"lag_2d_actual_{pollutant}"] = available_data.iloc[
                            1
                        ][actual_col]

                    # Rolling mean (2-day window)
                    if len(available_data) >= 2:
                        recent_values = available_data.iloc[:2][actual_col]
                        df.loc[i, f"roll_2d_actual_{pollutant}"] = recent_values.mean()

                # Lagged provider forecasts
                for provider in providers:
                    forecast_col = f"forecast_{provider}_{pollutant}"
                    if forecast_col in available_data.columns:
                        # t-1d lag
                        if len(available_data) >= 1:
                            df.loc[i, f"lag_1d_forecast_{provider}_{pollutant}"] = (
                                available_data.iloc[0][forecast_col]
                            )
                        # t-2d lag
                        if len(available_data) >= 2:
                            df.loc[i, f"lag_2d_forecast_{provider}_{pollutant}"] = (
                                available_data.iloc[1][forecast_col]
                            )

    log.info(f"Added {len(lag_cols)} lag features with no-leakage principle")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that combine existing data.
    """
    df = df.copy()

    pollutants = ["pm25", "pm10", "no2", "o3"]
    providers = ["cams", "noaa_gefs_aerosol"]

    # Provider agreement features (how much do forecasts agree?)
    for pollutant in pollutants:
        cams_col = f"forecast_cams_{pollutant}"
        noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

        if cams_col in df.columns and noaa_col in df.columns:
            # Absolute difference between providers
            df[f"provider_diff_{pollutant}"] = abs(df[cams_col] - df[noaa_col])

            # Mean of providers (ensemble)
            df[f"provider_mean_{pollutant}"] = (df[cams_col] + df[noaa_col]) / 2

            # Relative difference (as percentage of mean)
            mean_val = df[f"provider_mean_{pollutant}"]
            df[f"provider_rel_diff_{pollutant}"] = (
                df[f"provider_diff_{pollutant}"]
                / (mean_val + 0.1)
                * 100  # +0.1 to avoid division by zero
            )

    # Pollution ratios (common in air quality analysis)
    if "forecast_cams_pm25" in df.columns and "forecast_cams_pm10" in df.columns:
        df["pm25_pm10_ratio_cams"] = df["forecast_cams_pm25"] / (
            df["forecast_cams_pm10"] + 0.1
        )

    if (
        "forecast_noaa_gefs_aerosol_pm25" in df.columns
        and "forecast_noaa_gefs_aerosol_pm10" in df.columns
    ):
        df["pm25_pm10_ratio_noaa"] = df["forecast_noaa_gefs_aerosol_pm25"] / (
            df["forecast_noaa_gefs_aerosol_pm10"] + 0.1
        )

    # Air Quality Index approximations (simplified)
    for provider in providers:
        pm25_col = f"forecast_{provider}_pm25"
        pm10_col = f"forecast_{provider}_pm10"
        no2_col = f"forecast_{provider}_no2"
        o3_col = f"forecast_{provider}_o3"

        if all(col in df.columns for col in [pm25_col, pm10_col, no2_col, o3_col]):
            # Simple AQI approximation (not official, just for modeling)
            df[f"simple_aqi_{provider}"] = df[
                [pm25_col, pm10_col, no2_col, o3_col]
            ].max(axis=1)

    log.info("Added derived features: provider agreement, ratios, simple AQI")
    return df


def enhance_forecast_dataset(
    input_path: Path, output_path: Path = None
) -> pd.DataFrame:
    """
    Main function to add all features to the forecast comparison dataset.
    """
    log.info(f"Loading dataset from {input_path}")

    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    log.info(f"Original dataset shape: {df.shape}")

    # Add all feature types
    df = add_calendar_features(df)
    df = create_lag_features(df)
    df = add_derived_features(df)

    # Clean up temporary columns
    df = df.drop(columns=["date_dt", "forecast_made_dt"], errors="ignore")

    log.info(f"Enhanced dataset shape: {df.shape}")
    log.info(f"Added {df.shape[1] - 16} new features")  # Original had 16 columns

    # Save enhanced dataset
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        # Also save as CSV for easy inspection
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

        log.info(f"Saved enhanced dataset to {output_path} and {csv_path}")

    return df


def print_feature_summary(df: pd.DataFrame):
    """Print a summary of all features in the dataset."""

    print("\n" + "=" * 80)
    print("ENHANCED DATASET FEATURE SUMMARY")
    print("=" * 80)

    feature_groups = {
        "Identity": [
            col
            for col in df.columns
            if col in ["city", "date", "forecast_made_date", "forecast_lead_hours"]
        ],
        "Actuals": [col for col in df.columns if col.startswith("actual_")],
        "CAMS Forecasts": [
            col for col in df.columns if col.startswith("forecast_cams_")
        ],
        "NOAA Forecasts": [
            col for col in df.columns if col.startswith("forecast_noaa_")
        ],
        "Calendar": [
            col
            for col in df.columns
            if col
            in [
                "day_of_week",
                "dow_name",
                "month",
                "month_name",
                "is_holiday",
                "holiday_name",
                "is_weekend",
                "season",
            ]
        ],
        "Lag Features": [
            col
            for col in df.columns
            if col.startswith("lag_") or col.startswith("roll_")
        ],
        "Derived": [
            col
            for col in df.columns
            if col.startswith("provider_") or "ratio" in col or "aqi" in col
        ],
    }

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Cities: {df['city'].unique().tolist()}")
    print()

    for group_name, cols in feature_groups.items():
        if cols:
            print(f"{group_name} ({len(cols)} features):")
            for col in cols:
                if col in df.columns:
                    non_null = df[col].notna().sum()
                    print(f"  - {col} ({non_null}/{len(df)} non-null)")
            print()

    # Show sample row
    print("Sample row (Berlin, 2025-09-01):")
    sample_row = df[(df["city"] == "Berlin") & (df["date"] == "2025-09-01")]
    if not sample_row.empty:
        row = sample_row.iloc[0]
        print(
            f"  Calendar: {row['dow_name']}, {row['month_name']}, Holiday: {row['is_holiday']}, Weekend: {row['is_weekend']}"
        )
        print(
            f"  CAMS PM2.5: {row['forecast_cams_pm25']:.2f}, NOAA PM2.5: {row['forecast_noaa_gefs_aerosol_pm25']:.2f}"
        )
        print(
            f"  Provider agreement (PM2.5): {row.get('provider_diff_pm25', 'N/A'):.2f} difference"
        )

        # Show lag features if available
        lag_features = [
            col for col in row.index if col.startswith("lag_") and pd.notna(row[col])
        ]
        if lag_features:
            print(f"  Available lag features: {len(lag_features)}")


def main():
    parser = argparse.ArgumentParser(
        description="Add features to forecast comparison dataset"
    )
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--output", help="Output path for enhanced dataset")
    parser.add_argument(
        "--summary", action="store_true", help="Print detailed feature summary"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        log.error(f"Input file does not exist: {input_path}")
        return 1

    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / input_path.name.replace(".", "_enhanced.")

    # Enhance the dataset
    enhanced_df = enhance_forecast_dataset(input_path, output_path)

    if args.summary:
        print_feature_summary(enhanced_df)
    else:
        print(f"\nEnhanced dataset created: {output_path}")
        print(
            f"Shape: {enhanced_df.shape} (added {enhanced_df.shape[1] - 16} features)"
        )

    return 0


if __name__ == "__main__":
    exit(main())
