#!/usr/bin/env python3
"""
Walk-Forward Validation Demo

Fast demonstration of walk-forward validation approach using sampled data.
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def main():
    """Fast demo of walk-forward validation."""

    # Load data
    data_path = Path("data/analysis/3year_hourly_comprehensive_dataset.csv")
    log.info("Loading and sampling dataset...")

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Sample every 24th record (daily instead of hourly) for speed
    df_sampled = df.iloc[::24].copy().reset_index(drop=True)
    log.info(f"Sampled dataset: {len(df_sampled)} daily records")

    # Get numeric features
    target_cols = [col for col in df_sampled.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df_sampled.columns if col.startswith("forecast_")]

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

    feature_cols = [
        col
        for col in df_sampled.columns
        if col not in exclude_cols and df_sampled[col].dtype in ["int64", "float64"]
    ]

    # Use subset of features for speed
    feature_cols = feature_cols[:20]  # Use first 20 features

    log.info(f"Using {len(feature_cols)} features")

    # Split data
    train_data = df_sampled[df_sampled["year"].isin([2022, 2023])].copy()
    test_data = df_sampled[
        (df_sampled["year"] == 2024) & (df_sampled["month"] == 1)
    ].copy()  # January 2024 only

    log.info(f"Training: {len(train_data)} records, Testing: {len(test_data)} records")

    results = []

    # Simple walk-forward validation
    for i, (_, current_day) in enumerate(test_data.iterrows()):
        if i % 5 == 0:  # Process every 5th day for demo

            # Add previous test days to training
            if i > 0:
                previous_days = test_data.iloc[:i]
                current_train = pd.concat(
                    [train_data, previous_days], ignore_index=True
                )
            else:
                current_train = train_data.copy()

            # Process each pollutant
            for pollutant in ["pm25", "pm10"]:  # Just PM2.5 and PM10 for demo
                target_col = f"actual_{pollutant}"
                cams_col = f"forecast_cams_{pollutant}"
                noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

                # Prepare data
                X_train = current_train[feature_cols].fillna(0).values
                y_train = current_train[target_col].values
                X_test = current_day[feature_cols].fillna(0).values.reshape(1, -1)
                y_test = current_day[target_col]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Benchmark predictions
                cams_pred = current_day[cams_col]
                noaa_pred = current_day[noaa_col]
                simple_avg = (cams_pred + noaa_pred) / 2

                # Ensemble models
                models = {
                    "simple_average": simple_avg,
                    "ridge": Ridge(alpha=1.0, random_state=42),
                    "random_forest": RandomForestRegressor(
                        n_estimators=20, random_state=42, max_depth=5
                    ),
                }

                for model_name, model in models.items():
                    try:
                        if model_name == "simple_average":
                            ensemble_pred = simple_avg
                        else:
                            model.fit(X_train_scaled, y_train)
                            ensemble_pred = model.predict(X_test_scaled)[0]

                        # Calculate metrics
                        ensemble_mae = abs(y_test - ensemble_pred)
                        cams_mae = abs(y_test - cams_pred)
                        noaa_mae = abs(y_test - noaa_pred)

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

                        results.append(
                            {
                                "day": i,
                                "date": current_day["date"],
                                "pollutant": pollutant,
                                "model": model_name,
                                "actual": y_test,
                                "ensemble_pred": ensemble_pred,
                                "cams_pred": cams_pred,
                                "noaa_pred": noaa_pred,
                                "ensemble_mae": ensemble_mae,
                                "cams_mae": cams_mae,
                                "noaa_mae": noaa_mae,
                                "improvement_vs_cams": cams_improvement,
                                "improvement_vs_noaa": noaa_improvement,
                                "training_size": len(current_train),
                            }
                        )

                    except Exception as e:
                        log.warning(f"Error with {model_name} for {pollutant}: {e}")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save results
    output_path = Path("data/analysis/walk_forward_demo_results.csv")
    results_df.to_csv(output_path, index=False)
    log.info(f"Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION DEMO RESULTS - JANUARY 2024")
    print("=" * 70)

    # Summary by model
    print("\nMODEL PERFORMANCE SUMMARY:")
    for model in results_df["model"].unique():
        model_data = results_df[results_df["model"] == model]
        avg_mae = model_data["ensemble_mae"].mean()
        avg_cams_improvement = model_data["improvement_vs_cams"].mean()
        avg_noaa_improvement = model_data["improvement_vs_noaa"].mean()

        print(f"\n{model.upper().replace('_', ' ')}:")
        print(f"  Average MAE: {avg_mae:.3f} μg/m³")
        print(f"  Improvement vs CAMS: {avg_cams_improvement:+.1f}%")
        print(f"  Improvement vs NOAA: {avg_noaa_improvement:+.1f}%")

    # Benchmark performance
    print("\nBENCHMARK PERFORMANCE:")
    cams_avg_mae = results_df["cams_mae"].mean()
    noaa_avg_mae = results_df["noaa_mae"].mean()
    print(f"CAMS Average MAE: {cams_avg_mae:.3f} μg/m³")
    print(f"NOAA Average MAE: {noaa_avg_mae:.3f} μg/m³")

    # Best by pollutant
    print("\nBEST MODEL BY POLLUTANT:")
    for pollutant in results_df["pollutant"].unique():
        pollutant_data = results_df[results_df["pollutant"] == pollutant]
        best_row = pollutant_data.loc[pollutant_data["ensemble_mae"].idxmin()]

        print(f"\n{pollutant.upper()}:")
        print(f"  Best Model: {best_row['model'].replace('_', ' ').title()}")
        print(f"  Best MAE: {best_row['ensemble_mae']:.3f} μg/m³")
        print(
            f"  CAMS MAE: {best_row['cams_mae']:.3f} μg/m³ (improvement: {best_row['improvement_vs_cams']:+.1f}%)"
        )
        print(
            f"  NOAA MAE: {best_row['noaa_mae']:.3f} μg/m³ (improvement: {best_row['improvement_vs_noaa']:+.1f}%)"
        )

    total_predictions = len(results_df)
    print(f"\nTotal predictions: {total_predictions}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
