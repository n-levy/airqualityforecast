#!/usr/bin/env python3
"""
Optimized Walk-Forward Validation for Air Quality Forecasting

This script implements an optimized time-series walk-forward validation:
- Uses weekly chunks instead of daily for faster processing
- Focuses on key ensemble methods
- Processes first quarter of 2024 for demonstration
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def load_and_prepare_data(data_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load data and prepare numeric features."""
    log.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Target columns
    target_cols = [col for col in df.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df.columns if col.startswith("forecast_")]

    # Get numeric feature columns
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

    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                feature_cols.append(col)

    log.info(f"Loaded {len(df)} records with {len(feature_cols)} numeric features")
    return df, feature_cols


def create_optimized_models() -> Dict[str, object]:
    """Create optimized ensemble models."""
    return {
        "simple_average": "simple_average",
        "ridge_ensemble": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=50, random_state=42, n_jobs=-1, max_depth=8
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=50, random_state=42, max_depth=4, learning_rate=0.1
        ),
    }


def walk_forward_weekly(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Perform optimized walk-forward validation using weekly chunks."""

    # Focus on Q1 2024 for demonstration (Jan-Mar)
    train_data = df[df["year"].isin([2022, 2023])].copy()
    test_data = df[(df["year"] == 2024) & (df["month"].isin([1, 2, 3]))].copy()
    test_data = test_data.sort_values("datetime")

    log.info(f"Training data: {len(train_data)} records (2022-2023)")
    log.info(f"Test data: {len(test_data)} records (Q1 2024)")

    # Get weekly periods
    test_data["week"] = test_data["date"].dt.isocalendar().week
    unique_weeks = sorted(test_data["week"].unique())

    log.info(f"Processing {len(unique_weeks)} weeks in Q1 2024")

    results = []
    models = create_optimized_models()
    pollutants = ["pm25", "pm10", "no2", "o3"]

    # Process each week
    for week_num in unique_weeks:
        log.info(f"Processing week {week_num}")

        # Get current week's data
        current_week_data = test_data[test_data["week"] == week_num].copy()

        if len(current_week_data) == 0:
            continue

        # Update training data with previous weeks from 2024
        previous_weeks_2024 = test_data[test_data["week"] < week_num].copy()
        if len(previous_weeks_2024) > 0:
            current_train_data = pd.concat(
                [train_data, previous_weeks_2024], ignore_index=True
            )
        else:
            current_train_data = train_data.copy()

        # Process each pollutant
        for pollutant in pollutants:
            target_col = f"actual_{pollutant}"
            cams_col = f"forecast_cams_{pollutant}"
            noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

            if target_col not in current_train_data.columns:
                continue

            # Prepare data
            X_train = current_train_data[feature_cols].values
            y_train = current_train_data[target_col].values
            X_test = current_week_data[feature_cols].values
            y_test = current_week_data[target_col].values

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Get benchmark predictions
            cams_pred = current_week_data[cams_col].values
            noaa_pred = current_week_data[noaa_col].values

            # Train and predict with each model
            for model_name, model in models.items():
                try:
                    if model_name == "simple_average":
                        ensemble_pred = (cams_pred + noaa_pred) / 2
                    else:
                        model.fit(X_train_scaled, y_train)
                        ensemble_pred = model.predict(X_test_scaled)

                    # Calculate metrics
                    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                    ensemble_r2 = r2_score(y_test, ensemble_pred)
                    cams_mae = mean_absolute_error(y_test, cams_pred)
                    noaa_mae = mean_absolute_error(y_test, noaa_pred)

                    # Calculate improvements
                    cams_improvement = (cams_mae - ensemble_mae) / cams_mae * 100
                    noaa_improvement = (noaa_mae - ensemble_mae) / noaa_mae * 100

                    # Store aggregated weekly results
                    results.append(
                        {
                            "week": week_num,
                            "pollutant": pollutant,
                            "model": model_name,
                            "n_predictions": len(y_test),
                            "ensemble_mae": ensemble_mae,
                            "ensemble_r2": ensemble_r2,
                            "cams_mae": cams_mae,
                            "noaa_mae": noaa_mae,
                            "improvement_vs_cams": cams_improvement,
                            "improvement_vs_noaa": noaa_improvement,
                            "training_size": len(current_train_data),
                        }
                    )

                except Exception as e:
                    log.warning(
                        f"Error with {model_name} for {pollutant} in week {week_num}: {e}"
                    )
                    continue

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame) -> None:
    """Analyze and print results."""

    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS - Q1 2024 PREDICTIONS")
    print("=" * 80)

    # Overall model performance
    print("\nOVERALL ENSEMBLE PERFORMANCE:")
    for model in results_df["model"].unique():
        model_data = results_df[results_df["model"] == model]
        avg_mae = model_data["ensemble_mae"].mean()
        avg_r2 = model_data["ensemble_r2"].mean()
        avg_cams_improvement = model_data["improvement_vs_cams"].mean()
        avg_noaa_improvement = model_data["improvement_vs_noaa"].mean()

        print(f"\n{model.upper().replace('_', ' ')}:")
        print(f"  Average MAE: {avg_mae:.3f} μg/m³")
        print(f"  Average R²: {avg_r2:.3f}")
        print(f"  Improvement vs CAMS: {avg_cams_improvement:+.1f}%")
        print(f"  Improvement vs NOAA: {avg_noaa_improvement:+.1f}%")

    # Benchmark performance
    print("\nBENCHMARK PERFORMANCE:")
    cams_avg_mae = results_df["cams_mae"].mean()
    noaa_avg_mae = results_df["noaa_mae"].mean()
    print(f"CAMS Average MAE: {cams_avg_mae:.3f} μg/m³")
    print(f"NOAA Average MAE: {noaa_avg_mae:.3f} μg/m³")

    # Best model by pollutant
    print("\nBEST MODEL BY POLLUTANT:")
    for pollutant in results_df["pollutant"].unique():
        pollutant_data = results_df[results_df["pollutant"] == pollutant]
        best_row = pollutant_data.loc[pollutant_data["ensemble_mae"].idxmin()]

        print(f"\n{pollutant.upper()}:")
        print(f"  Best Model: {best_row['model'].replace('_', ' ').title()}")
        print(f"  Best MAE: {best_row['ensemble_mae']:.3f} μg/m³")
        print(f"  CAMS MAE: {best_row['cams_mae']:.3f} μg/m³")
        print(f"  NOAA MAE: {best_row['noaa_mae']:.3f} μg/m³")
        print(f"  Improvement vs CAMS: {best_row['improvement_vs_cams']:+.1f}%")
        print(f"  Improvement vs NOAA: {best_row['improvement_vs_noaa']:+.1f}%")

    # Summary statistics
    total_predictions = results_df["n_predictions"].sum()
    total_weeks = results_df["week"].nunique()

    print(f"\nSUMMARY:")
    print(f"Total predictions analyzed: {total_predictions:,}")
    print(f"Total weeks processed: {total_weeks}")
    print(f"Models tested: {results_df['model'].nunique()}")
    print("=" * 80)


def main():
    """Main execution function."""

    # Set paths
    data_path = Path("data/analysis/3year_hourly_comprehensive_dataset.csv")
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    # Load and prepare data
    df, feature_cols = load_and_prepare_data(data_path)

    # Perform walk-forward validation
    log.info("Starting optimized walk-forward validation...")
    results_df = walk_forward_weekly(df, feature_cols)

    # Save results
    results_path = output_dir / "walk_forward_optimized_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")

    # Analyze and print results
    analyze_results(results_df)

    return 0


if __name__ == "__main__":
    exit(main())
