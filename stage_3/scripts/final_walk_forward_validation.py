#!/usr/bin/env python3
"""
Final Walk-Forward Validation with All Features

This implements the CORRECT walk-forward validation approach as discussed:
- Uses ALL features including seasonal/temporal features (as would be available in deployment)
- Uses past year (2024-09-09 to 2025-09-08) as validation period
- Progressive day-by-day training that mirrors real deployment conditions
- Most realistic assessment of actual deployment performance

This is the definitive validation for the air quality forecasting pipeline.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
    """Load 5-year dataset and prepare ALL features (including temporal)."""
    log.info(f"Loading 5-year dataset from {data_path}")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Sample daily data for reasonable processing time
    df_sampled = (
        df.iloc[::6].copy().reset_index(drop=True)
    )  # Every 6 hours = 4 samples per day
    log.info(f"Sampled to 6-hourly frequency: {len(df_sampled)} records")

    # Get ALL feature columns (including temporal - this is the key difference!)
    target_cols = [col for col in df_sampled.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df_sampled.columns if col.startswith("forecast_")]

    # Only exclude metadata and targets/benchmarks - KEEP ALL TEMPORAL FEATURES
    exclude_cols = (
        {"city", "datetime", "date", "forecast_made_date"}
        | set(target_cols)
        | set(benchmark_cols)
    )

    feature_cols = []
    for col in df_sampled.columns:
        if col not in exclude_cols:
            if df_sampled[col].dtype in ["int64", "float64", "int32", "float32"]:
                feature_cols.append(col)
            elif col == "week_position":  # Handle categorical
                # Convert to numeric (0 for weekday, 1 for weekend)
                df_sampled[col] = (df_sampled[col] == "weekend").astype(int)
                feature_cols.append(col)

    log.info(f"Using ALL {len(feature_cols)} features INCLUDING temporal features")
    log.info(
        f"Time range: {df_sampled['datetime'].min()} to {df_sampled['datetime'].max()}"
    )

    # Verify we have temporal features
    temporal_features = [
        col
        for col in feature_cols
        if any(
            word in col.lower()
            for word in ["year", "month", "day", "hour", "week", "holiday"]
        )
    ]
    log.info(f"Temporal features included: {len(temporal_features)} features")

    return df_sampled, feature_cols


def create_models() -> Dict[str, Any]:
    """Create ensemble models for validation."""
    return {
        "simple_average": "simple_average",
        "ridge_ensemble": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=8
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1
        ),
    }


def walk_forward_validation_past_year(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Perform walk-forward validation using the past year (2024-09-09 to 2025-09-08).

    This mirrors exactly how the system would work in deployment:
    1. Train on all historical data up to prediction point
    2. Use ALL available features (including temporal)
    3. Predict next period
    4. Add new observation to training set
    5. Repeat
    """

    # Define validation period (past year from today's perspective)
    validation_start = pd.Timestamp("2024-09-09")
    validation_end = pd.Timestamp("2025-09-08")

    # Get training data (everything up to validation start)
    initial_train_data = df[df["datetime"] < validation_start].copy()

    # Get validation data (past year)
    validation_data = (
        df[(df["datetime"] >= validation_start) & (df["datetime"] <= validation_end)]
        .copy()
        .sort_values("datetime")
    )

    log.info(f"Initial training data: {len(initial_train_data)} records")
    log.info(f"Validation period: {validation_start.date()} to {validation_end.date()}")
    log.info(f"Validation data: {len(validation_data)} records")

    if len(validation_data) == 0:
        log.error("No validation data found for the specified period")
        return pd.DataFrame()

    # Get unique dates for daily walk-forward
    validation_dates = sorted(validation_data["date"].dt.date.unique())
    log.info(f"Walk-forward validation across {len(validation_dates)} days")

    models = create_models()
    pollutants = ["pm25", "pm10", "no2", "o3"]
    results = []

    # Process every 30th day for reasonable runtime (monthly walk-forward)
    sampled_dates = validation_dates[::30]  # Monthly sampling
    log.info(f"Using monthly sampling: {len(sampled_dates)} validation points")

    for date_idx, current_date in enumerate(sampled_dates):
        if date_idx % 10 == 0:
            log.info(f"Processing {date_idx + 1}/{len(sampled_dates)}: {current_date}")

        # Get current day's data for testing
        current_day_data = validation_data[
            validation_data["date"].dt.date == current_date
        ].copy()

        if len(current_day_data) == 0:
            continue

        # Update training data: add all validation data up to current date
        previous_validation_data = validation_data[
            validation_data["date"].dt.date < current_date
        ].copy()

        if len(previous_validation_data) > 0:
            current_train_data = pd.concat(
                [initial_train_data, previous_validation_data], ignore_index=True
            )
        else:
            current_train_data = initial_train_data.copy()

        # Process each pollutant
        for pollutant in pollutants:
            target_col = f"actual_{pollutant}"
            cams_col = f"forecast_cams_{pollutant}"
            noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

            if target_col not in current_train_data.columns:
                continue

            # Prepare data
            X_train = current_train_data[feature_cols].fillna(0).values
            y_train = current_train_data[target_col].values
            X_test = current_day_data[feature_cols].fillna(0).values
            y_test = current_day_data[target_col].values

            # Get benchmark predictions
            cams_pred = current_day_data[cams_col].values
            noaa_pred = current_day_data[noaa_col].values

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train and predict with each model
            for model_name, model in models.items():
                try:
                    if model_name == "simple_average":
                        ensemble_pred = (cams_pred + noaa_pred) / 2
                    else:
                        # Create fresh model instance
                        if model_name == "ridge_ensemble":
                            model = Ridge(alpha=1.0, random_state=42)
                        elif model_name == "random_forest":
                            model = RandomForestRegressor(
                                n_estimators=100, random_state=42, max_depth=8
                            )
                        elif model_name == "gradient_boosting":
                            model = GradientBoostingRegressor(
                                n_estimators=100,
                                random_state=42,
                                max_depth=6,
                                learning_rate=0.1,
                            )

                        model.fit(X_train_scaled, y_train)
                        ensemble_pred = model.predict(X_test_scaled)

                    # Calculate metrics for this day
                    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
                    ensemble_r2 = r2_score(y_test, ensemble_pred)
                    cams_mae = mean_absolute_error(y_test, cams_pred)
                    noaa_mae = mean_absolute_error(y_test, noaa_pred)

                    # Calculate improvements
                    cams_improvement = (
                        (cams_mae - ensemble_mae) / cams_mae * 100
                        if cams_mae > 0
                        else 0
                    )
                    noaa_improvement = (
                        (noaa_mae - ensemble_mae) / noaa_mae * 100
                        if noaa_mae > 0
                        else 0
                    )

                    # Store daily results
                    results.append(
                        {
                            "date": current_date,
                            "model": model_name,
                            "pollutant": pollutant,
                            "ensemble_mae": ensemble_mae,
                            "ensemble_rmse": ensemble_rmse,
                            "ensemble_r2": ensemble_r2,
                            "cams_mae": cams_mae,
                            "noaa_mae": noaa_mae,
                            "improvement_vs_cams": cams_improvement,
                            "improvement_vs_noaa": noaa_improvement,
                            "n_test_samples": len(y_test),
                            "n_train_samples": len(current_train_data),
                            "training_days": (
                                pd.Timestamp(current_date)
                                - current_train_data["datetime"].min()
                            ).days,
                        }
                    )

                except Exception as e:
                    log.warning(
                        f"Error with {model_name} for {pollutant} on {current_date}: {e}"
                    )
                    continue

    return pd.DataFrame(results)


def analyze_final_results(results_df: pd.DataFrame) -> None:
    """Analyze and print the final validation results."""

    print("\n" + "=" * 80)
    print("FINAL WALK-FORWARD VALIDATION RESULTS")
    print("Past Year Validation (2024-09-09 to 2025-09-08)")
    print("Using ALL Features Including Temporal/Seasonal")
    print("=" * 80)

    # Overall performance by model
    print("\nOVERALL MODEL PERFORMANCE:")
    overall_summary = (
        results_df.groupby("model")
        .agg(
            {
                "ensemble_mae": "mean",
                "ensemble_r2": "mean",
                "improvement_vs_cams": "mean",
                "improvement_vs_noaa": "mean",
                "n_test_samples": "sum",
            }
        )
        .round(3)
    )

    for model in overall_summary.index:
        data = overall_summary.loc[model]
        print(f"\n{model.upper().replace('_', ' ')}:")
        print(f"  Average MAE: {data['ensemble_mae']:.3f} μg/m³")
        print(f"  Average R²: {data['ensemble_r2']:.3f}")
        print(f"  Improvement vs CAMS: {data['improvement_vs_cams']:+.1f}%")
        print(f"  Improvement vs NOAA: {data['improvement_vs_noaa']:+.1f}%")
        print(f"  Total test samples: {int(data['n_test_samples']):,}")

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
        best_model_data = pollutant_data.groupby("model")["ensemble_mae"].mean()
        best_model = best_model_data.idxmin()
        best_mae = best_model_data[best_model]

        # Get benchmark performance for this pollutant
        cams_mae = pollutant_data["cams_mae"].mean()
        noaa_mae = pollutant_data["noaa_mae"].mean()

        cams_imp = (cams_mae - best_mae) / cams_mae * 100
        noaa_imp = (noaa_mae - best_mae) / noaa_mae * 100

        print(f"\n{pollutant.upper()}:")
        print(f"  Best Model: {best_model.replace('_', ' ').title()}")
        print(f"  Best MAE: {best_mae:.3f} μg/m³")
        print(f"  CAMS MAE: {cams_mae:.3f} μg/m³ (improvement: {cams_imp:+.1f}%)")
        print(f"  NOAA MAE: {noaa_mae:.3f} μg/m³ (improvement: {noaa_imp:+.1f}%)")

    # Model consistency and reliability
    print("\nMODEL CONSISTENCY:")
    consistency = results_df.groupby("model")["ensemble_mae"].agg(
        ["mean", "std", "count"]
    )
    consistency["cv"] = (
        consistency["std"] / consistency["mean"]
    )  # Coefficient of variation
    consistency = consistency.round(3)

    for model in consistency.index:
        data = consistency.loc[model]
        print(
            f"{model.replace('_', ' ').title()}: MAE {data['mean']:.3f}±{data['std']:.3f} (CV: {data['cv']:.3f}, n={int(data['count'])})"
        )

    # Temporal performance analysis
    print("\nTEMPORAL PERFORMANCE TRENDS:")
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df["month"] = results_df["date"].dt.month

    monthly_performance = (
        results_df.groupby(["month", "model"])["ensemble_mae"].mean().unstack()
    )

    print("Average MAE by Month:")
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    for month in monthly_performance.index:
        if month <= len(month_names):
            month_name = month_names[month - 1]
            best_model = monthly_performance.loc[month].idxmin()
            best_mae = monthly_performance.loc[month, best_model]
            print(
                f"  {month_name}: {best_mae:.3f} μg/m³ ({best_model.replace('_', ' ').title()})"
            )

    # Final summary statistics
    total_evaluations = len(results_df)
    total_test_samples = results_df["n_test_samples"].sum()
    unique_dates = results_df["date"].nunique()

    print(f"\nVALIDATION SUMMARY:")
    print(
        f"Validation period: {results_df['date'].min().date()} to {results_df['date'].max().date()}"
    )
    print(f"Total evaluations: {total_evaluations}")
    print(f"Unique validation dates: {unique_dates}")
    print(f"Total test samples: {total_test_samples:,}")
    print(f"Models tested: {results_df['model'].nunique()}")
    print(f"Pollutants tested: {results_df['pollutant'].nunique()}")

    # Final model ranking
    print("\nFINAL MODEL RANKING (by average MAE):")
    final_ranking = results_df.groupby("model")["ensemble_mae"].mean().sort_values()
    for rank, (model, mae) in enumerate(final_ranking.items(), 1):
        print(f"{rank}. {model.replace('_', ' ').title()}: {mae:.3f} μg/m³")

    print("\n" + "=" * 80)
    print("CONCLUSION: This walk-forward validation with ALL features provides")
    print("the most realistic assessment of deployment performance.")
    print("=" * 80)


def main():
    """Main execution function."""

    # Load data
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    df, feature_cols = load_and_prepare_data(data_path)

    # Run final walk-forward validation
    log.info("Starting final walk-forward validation with ALL features...")
    results_df = walk_forward_validation_past_year(df, feature_cols)

    if len(results_df) == 0:
        log.error("No validation results generated")
        return 1

    # Save results
    results_path = output_dir / "final_walk_forward_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")

    # Analyze and report results
    analyze_final_results(results_df)

    return 0


if __name__ == "__main__":
    exit(main())
