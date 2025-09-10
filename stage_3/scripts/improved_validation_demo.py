#!/usr/bin/env python3
"""
Improved Validation Strategy Demo

Fast demonstration of improved validation approaches using sampled data.
Shows the difference between validation strategies and their effectiveness.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def main():
    """Fast demonstration of improved validation strategies."""

    # Load and sample data
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    log.info("Loading and sampling 5-year dataset...")

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Sample daily data (every 24th record) for speed
    df_sampled = df.iloc[::24].copy().reset_index(drop=True)
    log.info(f"Sampled dataset: {len(df_sampled)} daily records")

    # Get features
    target_cols = [col for col in df_sampled.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df_sampled.columns if col.startswith("forecast_")]

    exclude_cols = (
        {"city", "datetime", "date", "forecast_made_date", "week_position"}
        | set(target_cols)
        | set(benchmark_cols)
    )

    feature_cols = [
        col
        for col in df_sampled.columns
        if col not in exclude_cols and df_sampled[col].dtype in ["int64", "float64"]
    ]

    # Use subset for speed
    feature_cols = feature_cols[:15]
    log.info(f"Using {len(feature_cols)} features")

    results = []
    models = {
        "simple_average": "simple_average",
        "ridge_ensemble": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=50, random_state=42, max_depth=6
        ),
    }

    # 1. BLOCKED TIME SERIES VALIDATION
    log.info("\n=== BLOCKED TIME SERIES VALIDATION ===")

    # Example: Train on 2020-2022, test on 2023 Q1
    train_blocked = df_sampled[
        (df_sampled["year"] >= 2020) & (df_sampled["year"] <= 2022)
    ].copy()
    test_blocked = df_sampled[
        (df_sampled["year"] == 2023) & (df_sampled["month"] <= 3)
    ].copy()

    log.info(
        f"Blocked - Train: {len(train_blocked)} records, Test: {len(test_blocked)} records"
    )

    for pollutant in ["pm25", "pm10"]:
        for model_name, model in models.items():
            result = evaluate_split(
                train_blocked,
                test_blocked,
                feature_cols,
                model_name,
                model,
                pollutant,
                "Blocked Time Series",
            )
            if result:
                results.append(result)

    # 2. SEASONAL SPLIT VALIDATION
    log.info("\n=== SEASONAL SPLIT VALIDATION ===")

    # Example: Train on all summers 2020-2024, test on summer 2025
    train_seasonal = df_sampled[
        (df_sampled["month"].isin([6, 7, 8]))
        & (df_sampled["year"] >= 2020)
        & (df_sampled["year"] <= 2024)
    ].copy()
    test_seasonal = df_sampled[
        (df_sampled["month"].isin([6, 7, 8])) & (df_sampled["year"] == 2025)
    ].copy()

    log.info(
        f"Seasonal - Train: {len(train_seasonal)} records, Test: {len(test_seasonal)} records"
    )

    for pollutant in ["pm25", "pm10"]:
        for model_name, model in models.items():
            result = evaluate_split(
                train_seasonal,
                test_seasonal,
                feature_cols,
                model_name,
                model,
                pollutant,
                "Seasonal Split",
            )
            if result:
                results.append(result)

    # 3. GEOGRAPHIC CROSS-VALIDATION
    log.info("\n=== GEOGRAPHIC CROSS-VALIDATION ===")

    # Example: Train on Berlin + Hamburg, test on Munich
    geo_data = df_sampled[df_sampled["year"] <= 2024].copy()  # Use 2020-2024
    train_geo = geo_data[geo_data["city"].isin(["Berlin", "Hamburg"])].copy()
    test_geo = geo_data[geo_data["city"] == "Munich"].copy()

    log.info(
        f"Geographic - Train: {len(train_geo)} records, Test: {len(test_geo)} records"
    )

    for pollutant in ["pm25", "pm10"]:
        for model_name, model in models.items():
            result = evaluate_split(
                train_geo,
                test_geo,
                feature_cols,
                model_name,
                model,
                pollutant,
                "Geographic Cross",
            )
            if result:
                results.append(result)

    # 4. SIMPLE WALK-FORWARD (for comparison)
    log.info("\n=== SIMPLE WALK-FORWARD (COMPARISON) ===")

    # Train on 2020-2023, test on 2024 Q1
    train_wf = df_sampled[df_sampled["year"] <= 2023].copy()
    test_wf = df_sampled[
        (df_sampled["year"] == 2024) & (df_sampled["month"] <= 3)
    ].copy()

    log.info(
        f"Walk-forward - Train: {len(train_wf)} records, Test: {len(test_wf)} records"
    )

    for pollutant in ["pm25", "pm10"]:
        for model_name, model in models.items():
            result = evaluate_split(
                train_wf,
                test_wf,
                feature_cols,
                model_name,
                model,
                pollutant,
                "Walk-Forward",
            )
            if result:
                results.append(result)

    # Create results dataframe and analyze
    results_df = pd.DataFrame(results)

    # Save results
    output_path = Path("data/analysis/improved_validation_demo_results.csv")
    results_df.to_csv(output_path, index=False)
    log.info(f"Results saved to {output_path}")

    # Print analysis
    print_analysis(results_df)

    return 0


def evaluate_split(
    train_data, test_data, feature_cols, model_name, model, pollutant, strategy
):
    """Evaluate model on a single split."""

    target_col = f"actual_{pollutant}"
    cams_col = f"forecast_cams_{pollutant}"
    noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

    try:
        # Prepare data
        X_train = train_data[feature_cols].fillna(0).values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].fillna(0).values
        y_test = test_data[target_col].values

        # Get benchmark predictions
        cams_pred = test_data[cams_col].values
        noaa_pred = test_data[noaa_col].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Make predictions
        if model_name == "simple_average":
            ensemble_pred = (cams_pred + noaa_pred) / 2
        else:
            # Create fresh model instance
            if model_name == "ridge_ensemble":
                model = Ridge(alpha=1.0, random_state=42)
            elif model_name == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=50, random_state=42, max_depth=6
                )

            model.fit(X_train_scaled, y_train)
            ensemble_pred = model.predict(X_test_scaled)

        # Calculate metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        cams_mae = mean_absolute_error(y_test, cams_pred)
        noaa_mae = mean_absolute_error(y_test, noaa_pred)

        # Calculate improvements (corrected calculation)
        cams_improvement = (
            (cams_mae - ensemble_mae) / cams_mae * 100 if cams_mae > 0 else 0
        )
        noaa_improvement = (
            (noaa_mae - ensemble_mae) / noaa_mae * 100 if noaa_mae > 0 else 0
        )

        return {
            "strategy": strategy,
            "model": model_name,
            "pollutant": pollutant,
            "ensemble_mae": ensemble_mae,
            "ensemble_r2": ensemble_r2,
            "cams_mae": cams_mae,
            "noaa_mae": noaa_mae,
            "improvement_vs_cams": cams_improvement,
            "improvement_vs_noaa": noaa_improvement,
            "n_test_samples": len(y_test),
            "n_train_samples": len(y_train),
        }

    except Exception as e:
        log.warning(f"Error evaluating {model_name} for {pollutant} in {strategy}: {e}")
        return None


def print_analysis(results_df):
    """Print comprehensive analysis of results."""

    print("\n" + "=" * 80)
    print("IMPROVED VALIDATION STRATEGY COMPARISON")
    print("=" * 80)

    # Compare strategies
    print("\nSTRATEGY COMPARISON:")
    strategy_summary = (
        results_df.groupby("strategy")
        .agg(
            {
                "ensemble_mae": "mean",
                "improvement_vs_cams": "mean",
                "improvement_vs_noaa": "mean",
                "ensemble_r2": "mean",
            }
        )
        .round(3)
    )

    for strategy in strategy_summary.index:
        data = strategy_summary.loc[strategy]
        print(f"\n{strategy.upper()}:")
        print(f"  Average MAE: {data['ensemble_mae']:.3f} μg/m³")
        print(f"  Average R²: {data['ensemble_r2']:.3f}")
        print(f"  Improvement vs CAMS: {data['improvement_vs_cams']:+.1f}%")
        print(f"  Improvement vs NOAA: {data['improvement_vs_noaa']:+.1f}%")

    # Best model by strategy
    print("\nBEST MODEL BY STRATEGY:")
    for strategy in results_df["strategy"].unique():
        strategy_data = results_df[results_df["strategy"] == strategy]
        best_model_data = strategy_data.groupby("model")["ensemble_mae"].mean()
        best_model = best_model_data.idxmin()
        best_mae = best_model_data[best_model]

        print(
            f"\n{strategy}: {best_model.replace('_', ' ').title()} (MAE: {best_mae:.3f})"
        )

    # Model consistency across strategies
    print("\nMODEL CONSISTENCY ACROSS STRATEGIES:")
    model_consistency = (
        results_df.groupby("model")
        .agg(
            {
                "ensemble_mae": ["mean", "std"],
                "improvement_vs_cams": ["mean", "std"],
                "improvement_vs_noaa": ["mean", "std"],
            }
        )
        .round(3)
    )

    for model in model_consistency.index:
        mae_mean = model_consistency.loc[model, ("ensemble_mae", "mean")]
        mae_std = model_consistency.loc[model, ("ensemble_mae", "std")]
        cams_mean = model_consistency.loc[model, ("improvement_vs_cams", "mean")]

        print(f"\n{model.replace('_', ' ').title()}:")
        print(f"  MAE: {mae_mean:.3f} ± {mae_std:.3f} μg/m³")
        print(f"  Avg improvement vs CAMS: {cams_mean:+.1f}%")

    # Validation strategy recommendations
    print("\n" + "=" * 80)
    print("VALIDATION STRATEGY RECOMMENDATIONS")
    print("=" * 80)

    # Find most robust strategy (lowest MAE variance)
    strategy_robustness = results_df.groupby("strategy")["ensemble_mae"].agg(
        ["mean", "std"]
    )
    strategy_robustness["cv"] = (
        strategy_robustness["std"] / strategy_robustness["mean"]
    )  # Coefficient of variation
    most_robust = strategy_robustness["cv"].idxmin()

    print(f"\nMOST ROBUST STRATEGY: {most_robust}")
    print(
        f"  Coefficient of Variation: {strategy_robustness.loc[most_robust, 'cv']:.3f}"
    )
    print(f"  Average MAE: {strategy_robustness.loc[most_robust, 'mean']:.3f} μg/m³")

    # Find best performing strategy
    best_strategy = results_df.groupby("strategy")["ensemble_mae"].mean().idxmin()
    best_mae = results_df.groupby("strategy")["ensemble_mae"].mean()[best_strategy]

    print(f"\nBEST PERFORMING STRATEGY: {best_strategy}")
    print(f"  Average MAE: {best_mae:.3f} μg/m³")

    # Comparison with walk-forward
    if "Walk-Forward" in results_df["strategy"].values:
        wf_mae = results_df[results_df["strategy"] == "Walk-Forward"][
            "ensemble_mae"
        ].mean()
        other_strategies = results_df[results_df["strategy"] != "Walk-Forward"]
        other_mae = other_strategies["ensemble_mae"].mean()

        improvement = (wf_mae - other_mae) / wf_mae * 100

        print(f"\nIMPROVED STRATEGIES vs WALK-FORWARD:")
        print(f"  Walk-Forward MAE: {wf_mae:.3f} μg/m³")
        print(f"  Other Strategies MAE: {other_mae:.3f} μg/m³")
        print(f"  Improvement: {improvement:+.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    exit(main())
