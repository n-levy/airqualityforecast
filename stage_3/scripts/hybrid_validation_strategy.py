#!/usr/bin/env python3
"""
Hybrid Validation Strategy: Blocked Time Series + Walk-Forward

Combines the strengths of both approaches:
1. Blocked Time Series: Provides stable training sets and realistic temporal gaps
2. Walk-Forward: Tests progressive learning and adaptation to recent patterns

This hybrid approach provides robust validation without seasonal data leakage issues.
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
    """Load 5-year dataset and prepare features."""
    log.info(f"Loading 5-year dataset from {data_path}")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Sample daily data for faster processing
    df_sampled = df.iloc[::24].copy().reset_index(drop=True)
    log.info(f"Sampled to daily frequency: {len(df_sampled)} records")

    # Get numeric feature columns (exclude temporal features to avoid leakage)
    target_cols = [col for col in df_sampled.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df_sampled.columns if col.startswith("forecast_")]

    # Exclude temporal features that could cause data leakage
    exclude_cols = (
        {
            "city",
            "datetime",
            "date",
            "forecast_made_date",
            "week_position",
            "year",
            "month",
            "day",
            "dayofweek",
            "dayofyear",
            "day_of_month",
            "week_of_year",
            "days_to_weekend",
            "days_from_weekend",
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_month_sin",
            "day_of_month_cos",
        }
        | set(target_cols)
        | set(benchmark_cols)
    )

    feature_cols = []
    for col in df_sampled.columns:
        if col not in exclude_cols:
            if df_sampled[col].dtype in ["int64", "float64", "int32", "float32"]:
                feature_cols.append(col)

    log.info(f"Using {len(feature_cols)} non-temporal features")
    log.info(
        f"Time range: {df_sampled['datetime'].min()} to {df_sampled['datetime'].max()}"
    )

    return df_sampled, feature_cols


def create_models() -> Dict[str, Any]:
    """Create ensemble models for validation."""
    return {
        "simple_average": "simple_average",
        "ridge_ensemble": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=50, random_state=42, n_jobs=-1, max_depth=6
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=50, random_state=42, max_depth=4, learning_rate=0.1
        ),
    }


class HybridValidator:
    """Hybrid validation combining blocked time series and walk-forward approaches."""

    def __init__(self):
        self.results = []

    def blocked_time_series_splits(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
        """Generate blocked time series splits with temporal gaps."""
        splits = []

        # Define blocked periods (6-month blocks with 3-month gaps)
        periods = [
            # Train on 2020-2021, test on 2022 H1
            {
                "train_start": "2020-01-01",
                "train_end": "2021-12-31",
                "test_start": "2022-04-01",  # 3-month gap
                "test_end": "2022-09-30",
                "name": "Blocked_2020-21_train_2022-H1_test",
            },
            # Train on 2020-2022, test on 2023 H1
            {
                "train_start": "2020-01-01",
                "train_end": "2022-12-31",
                "test_start": "2023-04-01",  # 3-month gap
                "test_end": "2023-09-30",
                "name": "Blocked_2020-22_train_2023-H1_test",
            },
            # Train on 2020-2023, test on 2024 H1
            {
                "train_start": "2020-01-01",
                "train_end": "2023-12-31",
                "test_start": "2024-04-01",  # 3-month gap
                "test_end": "2024-09-30",
                "name": "Blocked_2020-23_train_2024-H1_test",
            },
        ]

        for period in periods:
            train_data = df[
                (df["datetime"] >= period["train_start"])
                & (df["datetime"] <= period["train_end"])
            ].copy()

            test_data = df[
                (df["datetime"] >= period["test_start"])
                & (df["datetime"] <= period["test_end"])
            ].copy()

            if len(train_data) > 500 and len(test_data) > 50:
                splits.append((train_data, test_data, period["name"]))

        log.info(f"Generated {len(splits)} blocked time series splits")
        return splits

    def walk_forward_splits(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
        """Generate walk-forward splits for final validation period."""
        splits = []

        # Use 2025 data for walk-forward validation (most recent)
        validation_data = df[df["year"] == 2025].copy()
        base_train_data = df[df["year"] <= 2024].copy()

        if len(validation_data) == 0:
            log.warning("No 2025 data available for walk-forward validation")
            return splits

        # Get monthly periods in 2025
        months_2025 = sorted(validation_data["month"].unique())

        for month in months_2025[:6]:  # First 6 months for speed
            # Training: all data up to current month
            current_month_start = f"2025-{month:02d}-01"

            # Test data: current month
            test_data = validation_data[validation_data["month"] == month].copy()

            # Training data: everything before current month
            if month == 1:
                train_data = base_train_data.copy()
            else:
                prev_month_data = validation_data[
                    validation_data["month"] < month
                ].copy()
                train_data = pd.concat(
                    [base_train_data, prev_month_data], ignore_index=True
                )

            if len(train_data) > 500 and len(test_data) > 10:
                split_name = f"WalkForward_2025-{month:02d}"
                splits.append((train_data, test_data, split_name))

        log.info(f"Generated {len(splits)} walk-forward splits")
        return splits

    def evaluate_model_on_split(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_cols: List[str],
        model_name: str,
        model: Any,
        pollutant: str,
        split_name: str,
    ) -> Dict[str, Any]:
        """Evaluate a model on a single train/test split."""

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
                elif model_name == "gradient_boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=50, random_state=42, max_depth=4, learning_rate=0.1
                    )

                model.fit(X_train_scaled, y_train)
                ensemble_pred = model.predict(X_test_scaled)

            # Calculate metrics
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            cams_mae = mean_absolute_error(y_test, cams_pred)
            noaa_mae = mean_absolute_error(y_test, noaa_pred)

            # Calculate improvements
            cams_improvement = (
                (cams_mae - ensemble_mae) / cams_mae * 100 if cams_mae > 0 else 0
            )
            noaa_improvement = (
                (noaa_mae - ensemble_mae) / noaa_mae * 100 if noaa_mae > 0 else 0
            )

            return {
                "split_name": split_name,
                "split_type": "Blocked" if "Blocked" in split_name else "WalkForward",
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
                "n_train_samples": len(y_train),
                "train_start": train_data["datetime"].min(),
                "train_end": train_data["datetime"].max(),
                "test_start": test_data["datetime"].min(),
                "test_end": test_data["datetime"].max(),
            }

        except Exception as e:
            log.warning(
                f"Error evaluating {model_name} for {pollutant} on {split_name}: {e}"
            )
            return None

    def run_hybrid_validation(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """Run complete hybrid validation strategy."""

        log.info("Starting hybrid validation strategy...")

        # Get all splits
        blocked_splits = self.blocked_time_series_splits(df)
        walkforward_splits = self.walk_forward_splits(df)
        all_splits = blocked_splits + walkforward_splits

        models = create_models()
        pollutants = ["pm25", "pm10"]  # Focus on PM for speed

        results = []

        for split_idx, (train_data, test_data, split_name) in enumerate(all_splits):
            log.info(f"Processing {split_name} ({split_idx + 1}/{len(all_splits)})")

            for pollutant in pollutants:
                for model_name, model in models.items():

                    result = self.evaluate_model_on_split(
                        train_data,
                        test_data,
                        feature_cols,
                        model_name,
                        model,
                        pollutant,
                        split_name,
                    )

                    if result:
                        results.append(result)

        return pd.DataFrame(results)


def analyze_hybrid_results(results_df: pd.DataFrame) -> None:
    """Analyze and print hybrid validation results."""

    print("\n" + "=" * 80)
    print("HYBRID VALIDATION RESULTS")
    print("Blocked Time Series + Walk-Forward")
    print("=" * 80)

    # Overall performance
    print("\nOVERALL PERFORMANCE:")
    overall_summary = (
        results_df.groupby("model")
        .agg(
            {
                "ensemble_mae": "mean",
                "ensemble_r2": "mean",
                "improvement_vs_cams": "mean",
                "improvement_vs_noaa": "mean",
            }
        )
        .round(3)
    )

    for model in overall_summary.index:
        data = overall_summary.loc[model]
        print(f"\n{model.upper().replace('_', ' ')}:")
        print(f"  Average MAE: {data['ensemble_mae']:.3f} ug/m³")
        print(f"  Average R²: {data['ensemble_r2']:.3f}")
        print(f"  Improvement vs CAMS: {data['improvement_vs_cams']:+.1f}%")
        print(f"  Improvement vs NOAA: {data['improvement_vs_noaa']:+.1f}%")

    # Compare validation approaches
    print("\nVALIDATION APPROACH COMPARISON:")
    approach_summary = (
        results_df.groupby("split_type")
        .agg({"ensemble_mae": ["mean", "std"], "improvement_vs_cams": "mean"})
        .round(3)
    )

    for approach in approach_summary.index:
        mae_mean = approach_summary.loc[approach, ("ensemble_mae", "mean")]
        mae_std = approach_summary.loc[approach, ("ensemble_mae", "std")]
        cams_imp = approach_summary.loc[approach, ("improvement_vs_cams", "mean")]

        print(f"\n{approach.upper()}:")
        print(f"  MAE: {mae_mean:.3f} ± {mae_std:.3f} ug/m³")
        print(f"  Improvement vs CAMS: {cams_imp:+.1f}%")

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
        print(f"  Best MAE: {best_mae:.3f} ug/m³")
        print(f"  CAMS MAE: {cams_mae:.3f} ug/m³ (improvement: {cams_imp:+.1f}%)")
        print(f"  NOAA MAE: {noaa_mae:.3f} ug/m³ (improvement: {noaa_imp:+.1f}%)")

    # Model consistency across splits
    print("\nMODEL CONSISTENCY:")
    consistency = results_df.groupby("model")["ensemble_mae"].agg(
        ["mean", "std", "count"]
    )
    consistency["cv"] = consistency["std"] / consistency["mean"]
    consistency = consistency.round(3)

    for model in consistency.index:
        data = consistency.loc[model]
        print(
            f"{model.replace('_', ' ').title()}: MAE {data['mean']:.3f}±{data['std']:.3f} (CV: {data['cv']:.3f}, n={int(data['count'])})"
        )

    # Summary statistics
    total_evaluations = len(results_df)
    total_test_samples = results_df["n_test_samples"].sum()

    print(f"\nSUMMARY:")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Total test samples: {total_test_samples:,}")
    print(f"Validation approaches: {results_df['split_type'].nunique()}")
    print(f"Models tested: {results_df['model'].nunique()}")

    # Final ranking
    print("\nFINAL MODEL RANKING (by average MAE):")
    final_ranking = results_df.groupby("model")["ensemble_mae"].mean().sort_values()
    for rank, (model, mae) in enumerate(final_ranking.items(), 1):
        print(f"{rank}. {model.replace('_', ' ').title()}: {mae:.3f} ug/m³")


def main():
    """Main execution function."""

    # Load data
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    df, feature_cols = load_and_prepare_data(data_path)

    # Run hybrid validation
    validator = HybridValidator()
    results_df = validator.run_hybrid_validation(df, feature_cols)

    # Save results
    results_path = output_dir / "hybrid_validation_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")

    # Analyze results
    analyze_hybrid_results(results_df)

    return 0


if __name__ == "__main__":
    exit(main())
