#!/usr/bin/env python3
"""
Create Extended Dataset with Multiple Ensemble Methods

This script:
1. Extends the dataset to cover the full month of August 2025
2. Implements multiple ensemble methods as described in the documentation:
   - Simple Average (baseline)
   - Weighted Average (performance-based weights)
   - Ridge Regression combiner
   - XGBoost combiner (max_depth=3)
   - Per-city bias correction
3. Performs comprehensive performance comparison

Based on ADR-004: Global combiner model per pollutant + optional per-city bias correction
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def generate_august_dates() -> List[str]:
    """Generate all dates for August 2025."""
    dates = []
    start_date = datetime(2025, 8, 1)

    for day in range(31):  # August has 31 days
        current_date = start_date + timedelta(days=day)
        dates.append(current_date.strftime("%Y-%m-%d"))

    return dates


def create_extended_synthetic_data(cities: List[str], dates: List[str]) -> pd.DataFrame:
    """
    Create extended synthetic dataset for August 2025 with realistic patterns.
    """
    np.random.seed(42)  # For reproducibility

    data = []

    for city in cities:
        # City-specific baseline patterns
        city_factors = {
            "Berlin": {"pm25": 11.0, "pm10": 19.0, "no2": 24.0, "o3": 36.0},
            "Hamburg": {"pm25": 10.5, "pm10": 18.0, "no2": 22.0, "o3": 34.0},
            "Munich": {"pm25": 9.8, "pm10": 17.0, "no2": 20.0, "o3": 37.0},
        }

        base = city_factors.get(city, city_factors["Berlin"])

        for i, date_str in enumerate(dates):
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")

            # Create realistic temporal patterns
            day_of_year = date_obj.timetuple().tm_yday
            day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday

            # Seasonal trend (August = late summer, decreasing O3, stable PM)
            seasonal_factor = (
                1.0 - (i / len(dates)) * 0.1
            )  # Slight decrease through August

            # Weekly pattern (weekends have lower NO2, higher O3)
            weekend_factor = 0.85 if day_of_week >= 5 else 1.0

            # Day-to-day variability
            daily_noise = np.random.normal(0, 0.15)

            # Weather-like patterns (some persistence)
            weather_factor = 1.0 + 0.2 * np.sin(i * 0.3) + 0.1 * np.cos(i * 0.7)

            # Calculate synthetic actuals with realistic patterns
            total_factor = seasonal_factor * weather_factor * (1 + daily_noise)

            actual_pm25 = max(0.5, base["pm25"] * total_factor)
            actual_pm10 = max(1.0, base["pm10"] * total_factor)
            actual_no2 = max(1.0, base["no2"] * total_factor * weekend_factor)
            actual_o3 = max(
                5.0, base["o3"] * total_factor * (2.0 - weekend_factor)
            )  # Higher on weekends

            # Create forecast data with provider-specific biases and errors
            # CAMS characteristics: slight underestimation, good for PM
            cams_bias = {"pm25": -0.1, "pm10": -0.05, "no2": 0.05, "o3": -0.15}
            cams_noise = np.random.normal(0, 0.8)

            forecast_cams_pm25 = max(0.5, actual_pm25 + cams_bias["pm25"] + cams_noise)
            forecast_cams_pm10 = max(
                1.0, actual_pm10 + cams_bias["pm10"] + cams_noise * 1.2
            )
            forecast_cams_no2 = max(
                1.0, actual_no2 + cams_bias["no2"] + cams_noise * 0.9
            )
            forecast_cams_o3 = max(5.0, actual_o3 + cams_bias["o3"] + cams_noise * 1.1)

            # NOAA characteristics: slight overestimation, good for O3
            noaa_bias = {"pm25": 0.15, "pm10": 0.1, "no2": -0.05, "o3": 0.05}
            noaa_noise = np.random.normal(0, 0.9)

            forecast_noaa_pm25 = max(0.5, actual_pm25 + noaa_bias["pm25"] + noaa_noise)
            forecast_noaa_pm10 = max(
                1.0, actual_pm10 + noaa_bias["pm10"] + noaa_noise * 1.1
            )
            forecast_noaa_no2 = max(
                1.0, actual_no2 + noaa_bias["no2"] + noaa_noise * 1.0
            )
            forecast_noaa_o3 = max(5.0, actual_o3 + noaa_bias["o3"] + noaa_noise * 1.05)

            # Forecast made date (24h in advance)
            forecast_made_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")

            data.append(
                {
                    "city": city,
                    "date": date_str,
                    "forecast_made_date": forecast_made_date,
                    "forecast_lead_hours": 24,
                    # Actuals
                    "actual_pm25": round(actual_pm25, 2),
                    "actual_pm10": round(actual_pm10, 2),
                    "actual_no2": round(actual_no2, 2),
                    "actual_o3": round(actual_o3, 2),
                    # CAMS forecasts
                    "forecast_cams_pm25": round(forecast_cams_pm25, 2),
                    "forecast_cams_pm10": round(forecast_cams_pm10, 2),
                    "forecast_cams_no2": round(forecast_cams_no2, 2),
                    "forecast_cams_o3": round(forecast_cams_o3, 2),
                    # NOAA forecasts
                    "forecast_noaa_gefs_aerosol_pm25": round(forecast_noaa_pm25, 2),
                    "forecast_noaa_gefs_aerosol_pm10": round(forecast_noaa_pm10, 2),
                    "forecast_noaa_gefs_aerosol_no2": round(forecast_noaa_no2, 2),
                    "forecast_noaa_gefs_aerosol_o3": round(forecast_noaa_o3, 2),
                }
            )

    df = pd.DataFrame(data)
    log.info(
        f"Created extended synthetic dataset: {len(df)} rows, {len(cities)} cities, {len(dates)} dates"
    )
    return df


def add_ensemble_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multiple ensemble forecast methods as described in the documentation.
    """
    df = df.copy()
    pollutants = ["pm25", "pm10", "no2", "o3"]

    log.info("Adding ensemble forecasts...")

    for pollutant in pollutants:
        cams_col = f"forecast_cams_{pollutant}"
        noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"
        actual_col = f"actual_{pollutant}"

        if cams_col in df.columns and noaa_col in df.columns:
            # 1. Simple Average (baseline)
            df[f"forecast_simple_avg_{pollutant}"] = (df[cams_col] + df[noaa_col]) / 2

            # 2. Weighted Average (based on historical performance)
            # Use a simple performance-based weighting (inverse of recent errors)
            df_sorted = df.sort_values(["city", "date"]).copy()

            # Calculate rolling performance weights
            df_sorted[f"forecast_weighted_avg_{pollutant}"] = np.nan

            for city in df["city"].unique():
                city_mask = df_sorted["city"] == city
                city_data = df_sorted[city_mask].copy()

                if len(city_data) < 5:  # Need minimum data for weighting
                    # Fall back to simple average
                    df_sorted.loc[city_mask, f"forecast_weighted_avg_{pollutant}"] = (
                        city_data[cams_col] + city_data[noaa_col]
                    ) / 2
                    continue

                for i in range(len(city_data)):
                    if i < 4:  # First few days, use simple average
                        weight = 0.5
                    else:
                        # Calculate weights based on recent performance (last 4 days)
                        recent_data = city_data.iloc[max(0, i - 4) : i]

                        cams_recent_mae = np.abs(
                            recent_data[cams_col] - recent_data[actual_col]
                        ).mean()
                        noaa_recent_mae = np.abs(
                            recent_data[noaa_col] - recent_data[actual_col]
                        ).mean()

                        # Inverse error weighting (better model gets higher weight)
                        if cams_recent_mae + noaa_recent_mae == 0:
                            weight = 0.5
                        else:
                            weight = noaa_recent_mae / (
                                cams_recent_mae + noaa_recent_mae
                            )

                    # Weighted forecast
                    cams_val = city_data.iloc[i][cams_col]
                    noaa_val = city_data.iloc[i][noaa_col]
                    weighted_forecast = weight * cams_val + (1 - weight) * noaa_val

                    df_sorted.iloc[
                        city_data.index[i] - df_sorted.index[0],
                        df_sorted.columns.get_loc(f"forecast_weighted_avg_{pollutant}"),
                    ] = weighted_forecast

            # Merge back to original dataframe
            df[f"forecast_weighted_avg_{pollutant}"] = df_sorted[
                f"forecast_weighted_avg_{pollutant}"
            ]

    log.info("Added simple average and weighted average ensembles")
    return df


def add_ml_ensemble_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add machine learning-based ensemble forecasts: Ridge and XGBoost combiners.
    """
    df = df.copy()
    pollutants = ["pm25", "pm10", "no2", "o3"]

    log.info("Adding ML-based ensemble forecasts...")

    # Sort by city and date for time series consistency
    df_sorted = df.sort_values(["city", "date"]).copy()

    for pollutant in pollutants:
        cams_col = f"forecast_cams_{pollutant}"
        noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"
        actual_col = f"actual_{pollutant}"

        # Initialize ensemble columns
        df_sorted[f"forecast_ridge_{pollutant}"] = np.nan
        df_sorted[f"forecast_xgboost_{pollutant}"] = np.nan

        # Train models per city (city-specific patterns)
        for city in df["city"].unique():
            city_mask = df_sorted["city"] == city
            city_data = df_sorted[city_mask].copy().reset_index(drop=True)

            if len(city_data) < 10:  # Need minimum data for ML
                # Fall back to simple average
                df_sorted.loc[city_mask, f"forecast_ridge_{pollutant}"] = (
                    city_data[cams_col] + city_data[noaa_col]
                ) / 2
                df_sorted.loc[city_mask, f"forecast_xgboost_{pollutant}"] = (
                    city_data[cams_col] + city_data[noaa_col]
                ) / 2
                continue

            # Create features for ML models
            X = city_data[[cams_col, noaa_col]].values
            y = city_data[actual_col].values

            # Use time series cross-validation approach
            train_size = max(
                7, int(len(city_data) * 0.6)
            )  # At least 7 days for training

            predictions_ridge = np.full(len(city_data), np.nan)
            predictions_xgb = np.full(len(city_data), np.nan)

            # Rolling window training and prediction
            for i in range(train_size, len(city_data)):
                # Training data: everything before current point
                X_train = X[:i]
                y_train = y[:i]
                X_test = X[i].reshape(1, -1)

                # Ridge Regression
                try:
                    ridge = Ridge(alpha=1.0)  # Simple Ridge with fixed alpha
                    ridge.fit(X_train, y_train)
                    predictions_ridge[i] = ridge.predict(X_test)[0]
                except:
                    predictions_ridge[i] = (X_test[0, 0] + X_test[0, 1]) / 2

                # XGBoost (Gradient Boosting with max_depth=3 as per ADR)
                try:
                    xgb = GradientBoostingRegressor(
                        max_depth=3, n_estimators=50, learning_rate=0.1, random_state=42
                    )
                    xgb.fit(X_train, y_train)
                    predictions_xgb[i] = xgb.predict(X_test)[0]
                except:
                    predictions_xgb[i] = (X_test[0, 0] + X_test[0, 1]) / 2

            # For early predictions (before enough training data), use simple average
            for i in range(train_size):
                predictions_ridge[i] = (X[i, 0] + X[i, 1]) / 2
                predictions_xgb[i] = (X[i, 0] + X[i, 1]) / 2

            # Assign predictions back to dataframe
            city_indices = df_sorted[city_mask].index
            df_sorted.loc[city_indices, f"forecast_ridge_{pollutant}"] = (
                predictions_ridge
            )
            df_sorted.loc[city_indices, f"forecast_xgboost_{pollutant}"] = (
                predictions_xgb
            )

    # Merge back to original order
    df = df_sorted.sort_index()

    log.info("Added Ridge and XGBoost ensemble forecasts")
    return df


def add_bias_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-city bias correction as described in ADR-004: y ≈ α_c + β_c·ŷ_global
    """
    df = df.copy()
    pollutants = ["pm25", "pm10", "no2", "o3"]

    log.info("Adding per-city bias correction...")

    df_sorted = df.sort_values(["city", "date"]).copy()

    for pollutant in pollutants:
        # Apply bias correction to the best performing ensemble (simple average as baseline)
        global_forecast_col = f"forecast_simple_avg_{pollutant}"
        actual_col = f"actual_{pollutant}"
        bias_corrected_col = f"forecast_bias_corrected_{pollutant}"

        df_sorted[bias_corrected_col] = np.nan

        for city in df["city"].unique():
            city_mask = df_sorted["city"] == city
            city_data = df_sorted[city_mask].copy().reset_index(drop=True)

            if len(city_data) < 10:
                # Not enough data for bias correction, use global forecast
                df_sorted.loc[city_mask, bias_corrected_col] = city_data[
                    global_forecast_col
                ]
                continue

            predictions = np.full(len(city_data), np.nan)

            # Rolling bias correction
            min_train_size = 7
            for i in range(min_train_size, len(city_data)):
                # Training data: recent history
                train_data = city_data.iloc[
                    max(0, i - 21) : i
                ]  # 21-day window as per PRD

                if len(train_data) < 3:
                    predictions[i] = city_data.iloc[i][global_forecast_col]
                    continue

                X_train = train_data[global_forecast_col].values.reshape(-1, 1)
                y_train = train_data[actual_col].values

                try:
                    # Simple linear bias correction: y = α + β * ŷ_global
                    from sklearn.linear_model import LinearRegression

                    bias_model = LinearRegression()
                    bias_model.fit(X_train, y_train)

                    global_pred = city_data.iloc[i][global_forecast_col]
                    predictions[i] = bias_model.predict([[global_pred]])[0]

                    # Apply physical constraints
                    predictions[i] = max(0.1, predictions[i])  # Non-negative

                except:
                    predictions[i] = city_data.iloc[i][global_forecast_col]

            # For early predictions, use global forecast
            for i in range(min_train_size):
                predictions[i] = city_data.iloc[i][global_forecast_col]

            # Assign back to dataframe
            city_indices = df_sorted[city_mask].index
            df_sorted.loc[city_indices, bias_corrected_col] = predictions

    df = df_sorted.sort_index()

    log.info("Added per-city bias correction")
    return df


def calculate_comprehensive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive performance metrics for all ensemble methods.
    """
    results = []
    pollutants = ["pm25", "pm10", "no2", "o3"]

    # All forecast methods to evaluate
    methods = [
        "forecast_cams",
        "forecast_noaa_gefs_aerosol",
        "forecast_simple_avg",
        "forecast_weighted_avg",
        "forecast_ridge",
        "forecast_xgboost",
        "forecast_bias_corrected",
    ]

    method_names = {
        "forecast_cams": "CAMS",
        "forecast_noaa_gefs_aerosol": "NOAA",
        "forecast_simple_avg": "Simple Average",
        "forecast_weighted_avg": "Weighted Average",
        "forecast_ridge": "Ridge Ensemble",
        "forecast_xgboost": "XGBoost Ensemble",
        "forecast_bias_corrected": "Bias Corrected",
    }

    for method in methods:
        for pollutant in pollutants:
            forecast_col = f"{method}_{pollutant}"
            actual_col = f"actual_{pollutant}"

            if forecast_col not in df.columns:
                continue

            # Get valid data
            valid_mask = df[actual_col].notna() & df[forecast_col].notna()

            if not valid_mask.any():
                continue

            actual = df.loc[valid_mask, actual_col]
            forecast = df.loc[valid_mask, forecast_col]

            # Calculate metrics
            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))
            mbe = (forecast - actual).mean()
            mape = np.abs((forecast - actual) / (actual + 0.1)).mean() * 100
            correlation = (
                np.corrcoef(actual, forecast)[0, 1] if len(actual) > 1 else np.nan
            )
            r2 = r2_score(actual, forecast)

            # Additional metrics
            hit_rate = (np.abs(forecast - actual) <= (actual * 0.2 + 0.5)).mean()

            results.append(
                {
                    "method": method_names.get(method, method),
                    "pollutant": pollutant,
                    "n_samples": len(actual),
                    "mae": mae,
                    "rmse": rmse,
                    "mbe": mbe,
                    "mape": mape,
                    "correlation": correlation,
                    "r2": r2,
                    "hit_rate": hit_rate,
                }
            )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Create extended dataset with multiple ensemble methods"
    )
    parser.add_argument(
        "--output-dir", default="data/analysis", help="Output directory"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Berlin", "Hamburg", "Munich"],
        help="Cities to include",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate August dates
    august_dates = generate_august_dates()
    log.info(f"Generated {len(august_dates)} dates for August 2025")

    # 2. Create extended synthetic dataset
    df = create_extended_synthetic_data(args.cities, august_dates)

    # 3. Add all ensemble methods
    df = add_ensemble_forecasts(df)
    df = add_ml_ensemble_forecasts(df)
    df = add_bias_correction(df)

    # 4. Save extended dataset
    output_file = output_dir / "august_forecast_comparison_with_ensembles.csv"
    df.to_csv(output_file, index=False)
    df.to_parquet(output_file.with_suffix(".parquet"), index=False)

    log.info(f"Saved extended dataset: {output_file} (shape: {df.shape})")

    # 5. Calculate comprehensive performance metrics
    metrics_df = calculate_comprehensive_metrics(df)

    metrics_file = output_dir / "august_ensemble_performance_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)

    log.info(f"Saved performance metrics: {metrics_file}")

    print(f"\n{'='*80}")
    print("EXTENDED DATASET CREATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Dataset shape: {df.shape}")
    print(f"Time period: August 2025 ({len(august_dates)} days)")
    print(f"Cities: {args.cities}")
    print(
        f"Forecast methods: {len(set(col.rsplit('_', 1)[0] for col in df.columns if 'forecast_' in col))}"
    )
    print(f"Output files:")
    print(f"  - Dataset: {output_file}")
    print(f"  - Metrics: {metrics_file}")

    return 0


if __name__ == "__main__":
    exit(main())
