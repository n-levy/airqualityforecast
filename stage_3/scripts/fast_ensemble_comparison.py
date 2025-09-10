#!/usr/bin/env python3
"""
Fast Ensemble Comparison - Walk-Forward Validation

Quick comparison of all ensemble methods using strategic sampling for immediate results.
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
from sklearn.linear_model import Ridge, ElasticNet
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

    # Sample every 16 hours for fast processing
    df_sampled = df.iloc[::16].copy().reset_index(drop=True)
    log.info(f"Sampled to 16-hourly frequency: {len(df_sampled)} records")

    # Get ALL feature columns (including temporal)
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

    return df_sampled, feature_cols


def create_all_models() -> Dict[str, Any]:
    """Create all ensemble models with reduced complexity for speed."""
    return {
        "simple_average": "simple_average",
        "ridge_ensemble": Ridge(alpha=1.0, random_state=42),
        "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=50, random_state=42, n_jobs=-1, max_depth=6
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=50, random_state=42, max_depth=4, learning_rate=0.1
        ),
    }


def fast_walk_forward_validation(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """Fast walk-forward validation with strategic sampling."""

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
    log.info(f"Validation data: {len(validation_data)} records")

    # Strategic sampling: one point per season + mid-points (6 total validation points)
    validation_dates = sorted(validation_data["date"].dt.date.unique())

    # Select 6 strategic dates across the year
    sampled_dates = [
        validation_dates[0],  # Start (Sep 2024)
        validation_dates[len(validation_dates) // 6],  # Late Oct 2024
        validation_dates[len(validation_dates) // 3],  # Jan 2025
        validation_dates[len(validation_dates) // 2],  # Mar 2025
        validation_dates[2 * len(validation_dates) // 3],  # Jun 2025
        validation_dates[-1],  # End (Sep 2025)
    ]

    log.info(f"Using strategic sampling: {len(sampled_dates)} validation points")

    models = create_all_models()
    pollutants = ["pm25", "pm10", "no2", "o3"]
    results = []

    for date_idx, current_date in enumerate(sampled_dates):
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
                        # Create fresh model instance for each evaluation
                        if model_name == "ridge_ensemble":
                            model = Ridge(alpha=1.0, random_state=42)
                        elif model_name == "elastic_net":
                            model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
                        elif model_name == "random_forest":
                            model = RandomForestRegressor(
                                n_estimators=50, random_state=42, n_jobs=-1, max_depth=6
                            )
                        elif model_name == "gradient_boosting":
                            model = GradientBoostingRegressor(
                                n_estimators=50,
                                random_state=42,
                                max_depth=4,
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


def analyze_fast_results(results_df: pd.DataFrame) -> None:
    """Analyze and print fast validation results with complete model comparison."""

    print("\n" + "=" * 80)
    print("FAST ENSEMBLE COMPARISON - WALK-FORWARD VALIDATION")
    print("All Ensemble Methods - Past Year Strategic Sampling")
    print("Using ALL Features Including Temporal/Seasonal")
    print("=" * 80)

    # Overall performance by model
    print("\nCOMPREHENSIVE MODEL RANKING:")
    overall_summary = (
        results_df.groupby("model")
        .agg(
            {
                "ensemble_mae": ["mean", "std"],
                "ensemble_r2": "mean",
                "improvement_vs_cams": "mean",
                "improvement_vs_noaa": "mean",
                "n_test_samples": "sum",
            }
        )
        .round(3)
    )

    # Flatten column names
    overall_summary.columns = [
        "mae_mean",
        "mae_std",
        "r2_mean",
        "cams_improvement",
        "noaa_improvement",
        "test_samples",
    ]

    # Sort by MAE (best first)
    overall_summary = overall_summary.sort_values("mae_mean")

    for rank, (model, data) in enumerate(overall_summary.iterrows(), 1):
        print(f"\n{rank}. {model.upper().replace('_', ' ')}:")
        print(f"   MAE: {data['mae_mean']:.3f} ± {data['mae_std']:.3f} ug/m3")
        print(f"   R²: {data['r2_mean']:.3f}")
        print(f"   Improvement vs CAMS: {data['cams_improvement']:+.1f}%")
        print(f"   Improvement vs NOAA: {data['noaa_improvement']:+.1f}%")
        print(f"   Test samples: {int(data['test_samples']):,}")

    # Benchmark performance
    print("\nBENCHMARK COMPARISON:")
    cams_avg_mae = results_df["cams_mae"].mean()
    noaa_avg_mae = results_df["noaa_mae"].mean()
    best_model_mae = overall_summary["mae_mean"].iloc[0]
    best_model_name = overall_summary.index[0]

    print(f"CAMS Average MAE: {cams_avg_mae:.3f} ug/m3")
    print(f"NOAA Average MAE: {noaa_avg_mae:.3f} ug/m3")
    print(
        f"Best Ensemble ({best_model_name.replace('_', ' ').title()}): {best_model_mae:.3f} ug/m3"
    )
    print(
        f"Best vs CAMS: {((cams_avg_mae - best_model_mae) / cams_avg_mae * 100):+.1f}% improvement"
    )
    print(
        f"Best vs NOAA: {((noaa_avg_mae - best_model_mae) / noaa_avg_mae * 100):+.1f}% improvement"
    )

    # Performance by pollutant
    print("\nPERFORMANCE BY POLLUTANT:")
    for pollutant in sorted(results_df["pollutant"].unique()):
        pollutant_data = results_df[results_df["pollutant"] == pollutant]
        model_performance = (
            pollutant_data.groupby("model")["ensemble_mae"].mean().sort_values()
        )

        print(f"\n{pollutant.upper()}:")
        for rank, (model, mae) in enumerate(model_performance.head(3).items(), 1):
            cams_mae = pollutant_data[pollutant_data["model"] == model][
                "cams_mae"
            ].mean()
            improvement = (cams_mae - mae) / cams_mae * 100
            print(
                f"  {rank}. {model.replace('_', ' ').title()}: {mae:.3f} ug/m3 ({improvement:+.1f}% vs CAMS)"
            )

    # Model consistency
    print("\nMODEL CONSISTENCY (Lower is better):")
    consistency = results_df.groupby("model")["ensemble_mae"].agg(["mean", "std"])
    consistency["cv"] = (
        consistency["std"] / consistency["mean"]
    )  # Coefficient of variation
    consistency = consistency.sort_values("cv")

    for model in consistency.index:
        data = consistency.loc[model]
        print(
            f"{model.replace('_', ' ').title():20}: CV = {data['cv']:.3f} (σ = {data['std']:.3f})"
        )

    # Performance summary table
    print("\nSUMMARY TABLE - MAE by Model and Pollutant (ug/m3):")
    pivot_table = results_df.pivot_table(
        index="pollutant", columns="model", values="ensemble_mae", aggfunc="mean"
    ).round(3)

    # Reorder columns by overall performance
    model_order = overall_summary.index.tolist()
    pivot_table = pivot_table[model_order]

    print(pivot_table.to_string())

    # Statistical details
    total_evaluations = len(results_df)
    total_test_samples = results_df["n_test_samples"].sum()
    unique_dates = results_df["date"].nunique()

    print(f"\nVALIDATION STATISTICS:")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Unique validation dates: {unique_dates}")
    print(f"Total test samples: {total_test_samples:,}")
    print(f"Models compared: {results_df['model'].nunique()}")
    print(f"Pollutants tested: {results_df['pollutant'].nunique()}")

    # Final recommendation
    best_model = overall_summary.index[0]
    best_mae = overall_summary.loc[best_model, "mae_mean"]
    best_std = overall_summary.loc[best_model, "mae_std"]
    best_improvement_cams = overall_summary.loc[best_model, "cams_improvement"]

    print(f"\n🏆 RECOMMENDED MODEL FOR PRODUCTION:")
    print(f"   {best_model.replace('_', ' ').title()}")
    print(f"   MAE: {best_mae:.3f} ± {best_std:.3f} ug/m3")
    print(f"   Average improvement: {best_improvement_cams:+.1f}% vs CAMS")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""

    # Load data
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    df, feature_cols = load_and_prepare_data(data_path)

    # Run fast walk-forward validation
    log.info("Starting fast ensemble comparison with strategic sampling...")
    results_df = fast_walk_forward_validation(df, feature_cols)

    if len(results_df) == 0:
        log.error("No validation results generated")
        return 1

    # Save results
    results_path = output_dir / "fast_ensemble_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")

    # Analyze and report results
    analyze_fast_results(results_df)

    return 0


if __name__ == "__main__":
    exit(main())
