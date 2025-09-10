#!/usr/bin/env python3
"""
Add Ensemble Forecast and Performance Comparison

This script:
1. Adds a simple ensemble forecast (average of CAMS and NOAA)
2. Calculates comprehensive performance metrics for all three forecasts
3. Creates a performance comparison analysis
4. Generates visualizable performance summaries

The ensemble approach is a common baseline in forecasting that often
outperforms individual models by reducing variance.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def add_ensemble_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple ensemble forecast as the average of CAMS and NOAA forecasts.
    """
    df = df.copy()

    pollutants = ["pm25", "pm10", "no2", "o3"]

    for pollutant in pollutants:
        cams_col = f"forecast_cams_{pollutant}"
        noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"
        ensemble_col = f"forecast_ensemble_{pollutant}"

        if cams_col in df.columns and noaa_col in df.columns:
            # Simple average - only where both forecasts are available
            df[ensemble_col] = (df[cams_col] + df[noaa_col]) / 2

            # Handle cases where only one forecast is available
            df.loc[df[cams_col].isna() & df[noaa_col].notna(), ensemble_col] = df[
                noaa_col
            ]
            df.loc[df[noaa_col].isna() & df[cams_col].notna(), ensemble_col] = df[
                cams_col
            ]

    log.info(f"Added ensemble forecasts for {len(pollutants)} pollutants")
    return df


def calculate_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive performance metrics for all forecast methods.

    Metrics include:
    - Mean Absolute Error (MAE)
    - Root Mean Square Error (RMSE)
    - Mean Bias Error (MBE)
    - Mean Absolute Percentage Error (MAPE)
    - Correlation coefficient
    - Skill score vs persistence
    """

    results = []
    pollutants = ["pm25", "pm10", "no2", "o3"]
    providers = ["cams", "noaa_gefs_aerosol", "ensemble"]

    for provider in providers:
        for pollutant in pollutants:
            actual_col = f"actual_{pollutant}"
            forecast_col = f"forecast_{provider}_{pollutant}"

            if actual_col in df.columns and forecast_col in df.columns:
                # Get valid data points (both actual and forecast available)
                valid_mask = df[actual_col].notna() & df[forecast_col].notna()

                if not valid_mask.any():
                    continue

                actual = df.loc[valid_mask, actual_col]
                forecast = df.loc[valid_mask, forecast_col]

                # Calculate metrics
                error = forecast - actual
                abs_error = np.abs(error)
                sq_error = error**2

                # Basic metrics
                mae = abs_error.mean()
                rmse = np.sqrt(sq_error.mean())
                mbe = error.mean()  # Mean Bias Error

                # Percentage error (avoid division by very small values)
                actual_nonzero = (
                    actual + 0.1
                )  # Add small constant to avoid division by zero
                ape = np.abs(error / actual_nonzero) * 100
                mape = ape.mean()

                # Correlation
                correlation = (
                    np.corrcoef(actual, forecast)[0, 1] if len(actual) > 1 else np.nan
                )

                # R-squared
                ss_res = sq_error.sum()
                ss_tot = ((actual - actual.mean()) ** 2).sum()
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

                # Skill metrics
                # Hit rate for categorical performance (within ±20% of actual)
                tolerance = 0.2  # 20% tolerance
                hits = (
                    abs_error <= (actual * tolerance + 0.5)
                ).sum()  # +0.5 for absolute tolerance
                hit_rate = hits / len(actual)

                # Index of Agreement (Willmott's d)
                numerator = sq_error.sum()
                denominator = (
                    (np.abs(forecast - actual.mean()) + np.abs(actual - actual.mean()))
                    ** 2
                ).sum()
                index_agreement = (
                    1 - (numerator / denominator) if denominator > 0 else np.nan
                )

                results.append(
                    {
                        "provider": provider,
                        "pollutant": pollutant,
                        "n_samples": len(actual),
                        "mae": mae,
                        "rmse": rmse,
                        "mbe": mbe,
                        "mape": mape,
                        "correlation": correlation,
                        "r2": r2,
                        "hit_rate": hit_rate,
                        "index_agreement": index_agreement,
                        "actual_mean": actual.mean(),
                        "forecast_mean": forecast.mean(),
                        "actual_std": actual.std(),
                        "forecast_std": forecast.std(),
                    }
                )

    return pd.DataFrame(results)


def create_performance_summary(metrics_df: pd.DataFrame) -> Dict:
    """
    Create a comprehensive performance summary comparing all providers.
    """

    summary = {"overall_ranking": {}, "by_pollutant": {}, "key_insights": []}

    # Overall ranking by MAE (primary metric)
    overall_mae = metrics_df.groupby("provider")["mae"].mean().sort_values()
    summary["overall_ranking"]["by_mae"] = overall_mae.to_dict()

    # Overall ranking by RMSE
    overall_rmse = metrics_df.groupby("provider")["rmse"].mean().sort_values()
    summary["overall_ranking"]["by_rmse"] = overall_rmse.to_dict()

    # Overall ranking by correlation
    overall_corr = (
        metrics_df.groupby("provider")["correlation"]
        .mean()
        .sort_values(ascending=False)
    )
    summary["overall_ranking"]["by_correlation"] = overall_corr.to_dict()

    # Performance by pollutant
    for pollutant in metrics_df["pollutant"].unique():
        pollutant_data = metrics_df[metrics_df["pollutant"] == pollutant]

        mae_ranking = pollutant_data.set_index("provider")["mae"].sort_values()
        corr_ranking = pollutant_data.set_index("provider")["correlation"].sort_values(
            ascending=False
        )

        summary["by_pollutant"][pollutant] = {
            "best_mae": mae_ranking.index[0],
            "best_correlation": corr_ranking.index[0],
            "mae_values": mae_ranking.to_dict(),
            "correlation_values": corr_ranking.to_dict(),
        }

    # Key insights
    ensemble_mae = overall_mae.get("ensemble", np.nan)
    cams_mae = overall_mae.get("cams", np.nan)
    noaa_mae = overall_mae.get("noaa_gefs_aerosol", np.nan)

    if not np.isnan(ensemble_mae):
        if ensemble_mae < min(cams_mae, noaa_mae):
            improvement_vs_best = min(cams_mae, noaa_mae) - ensemble_mae
            best_individual = "CAMS" if cams_mae < noaa_mae else "NOAA"
            summary["key_insights"].append(
                f"Ensemble outperforms both individual models (MAE improvement: {improvement_vs_best:.3f} vs best individual {best_individual})"
            )
        else:
            summary["key_insights"].append(
                "Individual models outperform simple ensemble"
            )

    # Correlation insights
    ensemble_corr = overall_corr.get("ensemble", np.nan)
    if not np.isnan(ensemble_corr):
        if ensemble_corr > max(
            overall_corr.get("cams", 0), overall_corr.get("noaa_gefs_aerosol", 0)
        ):
            summary["key_insights"].append(
                f"Ensemble shows highest correlation with observations ({ensemble_corr:.3f})"
            )

    return summary


def print_performance_analysis(metrics_df: pd.DataFrame, summary: Dict):
    """
    Print a comprehensive performance analysis.
    """

    print("\n" + "=" * 80)
    print("FORECAST PERFORMANCE COMPARISON ANALYSIS")
    print("=" * 80)

    # Overall performance table
    print("\nOVERALL PERFORMANCE (across all pollutants):")
    overall_stats = (
        metrics_df.groupby("provider")
        .agg(
            {
                "mae": "mean",
                "rmse": "mean",
                "mbe": "mean",
                "mape": "mean",
                "correlation": "mean",
                "r2": "mean",
                "hit_rate": "mean",
                "n_samples": "sum",
            }
        )
        .round(3)
    )

    # Sort by MAE (primary metric)
    overall_stats = overall_stats.sort_values("mae")
    print(overall_stats.to_string())

    print(f"\nRANKING BY KEY METRICS:")
    print("MAE (lower is better):")
    for i, (provider, mae) in enumerate(
        summary["overall_ranking"]["by_mae"].items(), 1
    ):
        print(f"  {i}. {provider.upper()}: {mae:.3f} μg/m³")

    print("\nCorrelation (higher is better):")
    for i, (provider, corr) in enumerate(
        summary["overall_ranking"]["by_correlation"].items(), 1
    ):
        print(f"  {i}. {provider.upper()}: {corr:.3f}")

    # Performance by pollutant
    print(f"\nPERFORMANCE BY POLLUTANT:")
    for pollutant in sorted(summary["by_pollutant"].keys()):
        data = summary["by_pollutant"][pollutant]
        print(f"\n{pollutant.upper()}:")
        print(
            f"  Best MAE: {data['best_mae'].upper()} ({data['mae_values'][data['best_mae']]:.3f} μg/m³)"
        )
        print(
            f"  Best Correlation: {data['best_correlation'].upper()} ({data['correlation_values'][data['best_correlation']]:.3f})"
        )

        # Show all provider values for this pollutant
        print("  All providers (MAE):")
        for provider, mae in data["mae_values"].items():
            print(f"    {provider.upper()}: {mae:.3f} μg/m³")

    # Key insights
    print(f"\nKEY INSIGHTS:")
    for insight in summary["key_insights"]:
        print(f"• {insight}")

    # Detailed error analysis
    print(f"\nDETAILED ERROR ANALYSIS:")
    print("\nMean Bias Error (MBE) by provider:")
    mbe_stats = metrics_df.groupby("provider")["mbe"].mean().round(3)
    for provider, mbe in mbe_stats.items():
        bias_direction = (
            "overestimates" if mbe > 0 else "underestimates" if mbe < 0 else "unbiased"
        )
        print(f"  {provider.upper()}: {mbe:+.3f} μg/m³ ({bias_direction})")


def main():
    parser = argparse.ArgumentParser(
        description="Add ensemble forecast and compare performance"
    )
    parser.add_argument("--input", required=True, help="Input enhanced dataset path")
    parser.add_argument("--output", help="Output path for dataset with ensemble")
    parser.add_argument("--metrics-output", help="Output path for performance metrics")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        log.error(f"Input file does not exist: {input_path}")
        return 1

    # Load data
    log.info(f"Loading dataset from {input_path}")
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    log.info(f"Original dataset shape: {df.shape}")

    # Add ensemble forecast
    df_with_ensemble = add_ensemble_forecast(df)

    # Calculate performance metrics
    log.info("Calculating performance metrics for all forecasts...")
    metrics_df = calculate_performance_metrics(df_with_ensemble)

    if metrics_df.empty:
        log.error("No valid forecast data found for performance calculation")
        return 1

    # Create performance summary
    summary = create_performance_summary(metrics_df)

    # Save enhanced dataset
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / input_path.name.replace(
            ".", "_with_ensemble."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        df_with_ensemble.to_parquet(output_path, index=False)
    else:
        df_with_ensemble.to_csv(output_path, index=False)

    log.info(f"Saved dataset with ensemble to {output_path}")

    # Save performance metrics
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
    else:
        metrics_path = output_path.parent / "forecast_performance_comparison.csv"

    metrics_df.to_csv(metrics_path, index=False)
    metrics_df.to_parquet(metrics_path.with_suffix(".parquet"), index=False)
    log.info(f"Saved performance metrics to {metrics_path}")

    # Print analysis
    print_performance_analysis(metrics_df, summary)

    return 0


if __name__ == "__main__":
    exit(main())
