#!/usr/bin/env python3
"""
Walk-Forward Validation for Air Quality Forecasting

This script implements a time-series walk-forward validation approach:
1. Train on 2022-2023, predict Jan 1st 2024
2. Train on 2022-2023 + Jan 1st 2024, predict Jan 2nd 2024
3. Continue day-by-day through 2024

Compares 5 ensemble methods against 2 benchmark models.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def load_comprehensive_dataset(data_path: Path) -> pd.DataFrame:
    """Load the comprehensive dataset."""
    log.info(f"Loading comprehensive dataset from {data_path}")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])
    log.info(f"Loaded {len(df)} records with {len(df.columns)} features")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Identify feature columns, target columns, and benchmark columns."""

    # Target columns (actual values)
    target_cols = [col for col in df.columns if col.startswith("actual_")]

    # Benchmark forecast columns
    benchmark_cols = [col for col in df.columns if col.startswith("forecast_")]

    # Feature columns (exclude metadata, targets, raw forecasts, and categorical cols)
    exclude_cols = (
        {
            "city",
            "datetime",
            "date",
            "forecast_made_date",
            "holiday_type",
            "week_position",
        }
        | set(target_cols)
        | set(benchmark_cols)
    )

    # Get numeric columns only
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            # Check if column is numeric
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                feature_cols.append(col)
            else:
                log.info(
                    f"Excluding non-numeric column: {col} (dtype: {df[col].dtype})"
                )

    log.info(
        f"Features: {len(feature_cols)}, Targets: {len(target_cols)}, Benchmarks: {len(benchmark_cols)}"
    )
    return feature_cols, target_cols, benchmark_cols


def create_ensemble_models() -> Dict[str, object]:
    """Create ensemble models for comparison."""
    models = {
        "ridge_ensemble": Ridge(alpha=1.0, random_state=42),
        "elastic_net_ensemble": ElasticNet(
            alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000
        ),
        "random_forest_ensemble": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
        ),
        "gradient_boosting_ensemble": GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1
        ),
        "simple_average_ensemble": "simple_average",  # Special case for simple averaging
    }
    return models


def simple_average_prediction(
    cams_pred: np.ndarray, noaa_pred: np.ndarray
) -> np.ndarray:
    """Calculate simple average of CAMS and NOAA predictions."""
    return (cams_pred + noaa_pred) / 2


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    benchmark_cols: List[str],
) -> pd.DataFrame:
    """Perform walk-forward validation for 2024."""

    # Split data by year
    train_2022_2023 = df[df["year"].isin([2022, 2023])].copy()
    test_2024 = df[df["year"] == 2024].copy().sort_values("datetime")

    log.info(f"Initial training data: {len(train_2022_2023)} records (2022-2023)")
    log.info(f"Test data: {len(test_2024)} records (2024)")

    # Get unique dates in 2024
    unique_dates_2024 = sorted(test_2024["date"].dt.date.unique())
    log.info(f"Predicting {len(unique_dates_2024)} days in 2024")

    # Initialize results storage
    results = []
    models = create_ensemble_models()
    scalers = {}

    # Initialize scalers for each pollutant
    pollutants = ["pm25", "pm10", "no2", "o3"]
    for pollutant in pollutants:
        scalers[pollutant] = StandardScaler()

    # Walk forward through each day in 2024
    for i, current_date in enumerate(unique_dates_2024):
        if i % 30 == 0:  # Log every 30 days
            log.info(f"Processing day {i+1}/{len(unique_dates_2024)}: {current_date}")

        # Get current day's data
        current_day_data = test_2024[test_2024["date"].dt.date == current_date].copy()

        if len(current_day_data) == 0:
            continue

        # Update training data (add previous days from 2024)
        if i == 0:
            # First day: only use 2022-2023
            training_data = train_2022_2023.copy()
        else:
            # Add all previous days from 2024 to training
            previous_2024_data = test_2024[
                test_2024["date"].dt.date < current_date
            ].copy()
            training_data = pd.concat(
                [train_2022_2023, previous_2024_data], ignore_index=True
            )

        # Process each pollutant separately
        for pollutant in pollutants:
            target_col = f"actual_{pollutant}"
            cams_col = f"forecast_cams_{pollutant}"
            noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

            if target_col not in training_data.columns:
                continue

            # Prepare training data
            X_train = training_data[feature_cols].values
            y_train = training_data[target_col].values

            # Prepare test data
            X_test = current_day_data[feature_cols].values
            y_test = current_day_data[target_col].values

            # Get benchmark predictions
            cams_pred = current_day_data[cams_col].values
            noaa_pred = current_day_data[noaa_col].values

            # Scale features for this pollutant
            if i == 0:  # Fit scaler on first day
                X_train_scaled = scalers[pollutant].fit_transform(X_train)
            else:  # Use existing scaler
                X_train_scaled = scalers[pollutant].transform(X_train)

            X_test_scaled = scalers[pollutant].transform(X_test)

            # Train and predict with each ensemble model
            for model_name, model in models.items():
                try:
                    if model_name == "simple_average_ensemble":
                        # Simple average of CAMS and NOAA
                        ensemble_pred = simple_average_prediction(cams_pred, noaa_pred)
                    else:
                        # Train ML model
                        if i == 0:  # First time training
                            model.fit(X_train_scaled, y_train)
                        else:  # Retrain with updated data
                            model.fit(X_train_scaled, y_train)

                        ensemble_pred = model.predict(X_test_scaled)

                    # Calculate metrics for ensemble
                    mae_ensemble = mean_absolute_error(y_test, ensemble_pred)
                    rmse_ensemble = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                    r2_ensemble = r2_score(y_test, ensemble_pred)

                    # Calculate benchmark metrics
                    mae_cams = mean_absolute_error(y_test, cams_pred)
                    mae_noaa = mean_absolute_error(y_test, noaa_pred)

                    # Store results
                    for j, (actual, ensemble, cams, noaa) in enumerate(
                        zip(y_test, ensemble_pred, cams_pred, noaa_pred)
                    ):
                        results.append(
                            {
                                "date": current_date,
                                "datetime": current_day_data.iloc[j]["datetime"],
                                "city": current_day_data.iloc[j]["city"],
                                "pollutant": pollutant,
                                "model": model_name,
                                "actual": actual,
                                "ensemble_pred": ensemble,
                                "cams_pred": cams,
                                "noaa_pred": noaa,
                                "ensemble_mae": mae_ensemble,
                                "ensemble_rmse": rmse_ensemble,
                                "ensemble_r2": r2_ensemble,
                                "cams_mae": mae_cams,
                                "noaa_mae": mae_noaa,
                                "training_size": len(training_data),
                            }
                        )

                except Exception as e:
                    log.warning(
                        f"Error with {model_name} for {pollutant} on {current_date}: {e}"
                    )
                    continue

    return pd.DataFrame(results)


def calculate_summary_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary metrics for each model and pollutant."""

    summary_results = []

    # Group by model and pollutant
    for (model, pollutant), group in results_df.groupby(["model", "pollutant"]):

        # Calculate ensemble metrics
        ensemble_mae = mean_absolute_error(group["actual"], group["ensemble_pred"])
        ensemble_rmse = np.sqrt(
            mean_squared_error(group["actual"], group["ensemble_pred"])
        )
        ensemble_r2 = r2_score(group["actual"], group["ensemble_pred"])

        # Calculate benchmark metrics
        cams_mae = mean_absolute_error(group["actual"], group["cams_pred"])
        cams_rmse = np.sqrt(mean_squared_error(group["actual"], group["cams_pred"]))
        cams_r2 = r2_score(group["actual"], group["cams_pred"])

        noaa_mae = mean_absolute_error(group["actual"], group["noaa_pred"])
        noaa_rmse = np.sqrt(mean_squared_error(group["actual"], group["noaa_pred"]))
        noaa_r2 = r2_score(group["actual"], group["noaa_pred"])

        # Calculate improvements
        cams_improvement = (cams_mae - ensemble_mae) / cams_mae * 100
        noaa_improvement = (noaa_mae - ensemble_mae) / noaa_mae * 100

        summary_results.append(
            {
                "model": model,
                "pollutant": pollutant,
                "n_predictions": len(group),
                "ensemble_mae": ensemble_mae,
                "ensemble_rmse": ensemble_rmse,
                "ensemble_r2": ensemble_r2,
                "cams_mae": cams_mae,
                "cams_rmse": cams_rmse,
                "cams_r2": cams_r2,
                "noaa_mae": noaa_mae,
                "noaa_rmse": noaa_rmse,
                "noaa_r2": noaa_r2,
                "improvement_vs_cams": cams_improvement,
                "improvement_vs_noaa": noaa_improvement,
            }
        )

    return pd.DataFrame(summary_results)


def main():
    """Main execution function."""

    # Set paths
    data_path = Path("data/analysis/3year_hourly_comprehensive_dataset.csv")
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    # Load data
    df = load_comprehensive_dataset(data_path)

    # Prepare features
    feature_cols, target_cols, benchmark_cols = prepare_features(df)

    # Perform walk-forward validation
    log.info("Starting walk-forward validation...")
    results_df = walk_forward_validation(df, feature_cols, target_cols, benchmark_cols)

    # Save detailed results
    detailed_results_path = output_dir / "walk_forward_detailed_results.csv"
    results_df.to_csv(detailed_results_path, index=False)
    log.info(f"Detailed results saved to {detailed_results_path}")

    # Calculate summary metrics
    summary_df = calculate_summary_metrics(results_df)

    # Save summary results
    summary_results_path = output_dir / "walk_forward_summary_results.csv"
    summary_df.to_csv(summary_results_path, index=False)
    log.info(f"Summary results saved to {summary_results_path}")

    # Print results
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS - 2024 PREDICTIONS")
    print("=" * 80)

    print("\nENSEMBLE MODELS PERFORMANCE:")
    for model in summary_df["model"].unique():
        model_data = summary_df[summary_df["model"] == model]
        avg_mae = model_data["ensemble_mae"].mean()
        avg_r2 = model_data["ensemble_r2"].mean()
        avg_cams_improvement = model_data["improvement_vs_cams"].mean()
        avg_noaa_improvement = model_data["improvement_vs_noaa"].mean()

        print(f"\n{model.upper().replace('_', ' ')}:")
        print(f"  Average MAE: {avg_mae:.3f} μg/m³")
        print(f"  Average R²: {avg_r2:.3f}")
        print(f"  Improvement vs CAMS: {avg_cams_improvement:+.1f}%")
        print(f"  Improvement vs NOAA: {avg_noaa_improvement:+.1f}%")

    print("\nBENCHMARK PERFORMANCE:")
    cams_avg_mae = summary_df["cams_mae"].mean()
    cams_avg_r2 = summary_df["cams_r2"].mean()
    noaa_avg_mae = summary_df["noaa_mae"].mean()
    noaa_avg_r2 = summary_df["noaa_r2"].mean()

    print(f"CAMS: MAE={cams_avg_mae:.3f} μg/m³, R²={cams_avg_r2:.3f}")
    print(f"NOAA: MAE={noaa_avg_mae:.3f} μg/m³, R²={noaa_avg_r2:.3f}")

    print("\nPOLLUTANT-SPECIFIC RESULTS:")
    for pollutant in summary_df["pollutant"].unique():
        print(f"\n{pollutant.upper()}:")
        pollutant_data = summary_df[summary_df["pollutant"] == pollutant]

        # Find best ensemble model for this pollutant
        best_model = pollutant_data.loc[pollutant_data["ensemble_mae"].idxmin()]

        print(f"  Best Ensemble: {best_model['model'].replace('_', ' ').title()}")
        print(f"  Best MAE: {best_model['ensemble_mae']:.3f} μg/m³")
        print(f"  CAMS MAE: {best_model['cams_mae']:.3f} μg/m³")
        print(f"  NOAA MAE: {best_model['noaa_mae']:.3f} μg/m³")
        print(f"  Improvement vs CAMS: {best_model['improvement_vs_cams']:+.1f}%")
        print(f"  Improvement vs NOAA: {best_model['improvement_vs_noaa']:+.1f}%")

    print(f"\nTotal predictions analyzed: {len(results_df):,}")
    print(f"Total days processed: {results_df['date'].nunique()}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
