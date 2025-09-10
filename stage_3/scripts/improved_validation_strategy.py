#!/usr/bin/env python3
"""
Improved Validation Strategy for Air Quality Forecasting

Implements a comprehensive validation approach using:
1. Blocked Time Series Cross-Validation
2. Seasonal Split Validation
3. Geographic Cross-Validation
4. Hybrid approach combining all methods

This provides more robust and realistic validation than simple walk-forward validation.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
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

    # Get numeric feature columns
    target_cols = [col for col in df.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df.columns if col.startswith("forecast_")]

    exclude_cols = (
        {"city", "datetime", "date", "forecast_made_date", "week_position"}
        | set(target_cols)
        | set(benchmark_cols)
    )

    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols:
            if df[col].dtype in ["int64", "float64", "int32", "float32"]:
                feature_cols.append(col)

    log.info(f"Loaded {len(df)} records with {len(feature_cols)} features")
    log.info(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df, feature_cols


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


class BlockedTimeSeriesValidator:
    """Blocked time series cross-validation."""

    def __init__(
        self, train_months: int = 12, test_months: int = 3, gap_months: int = 1
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.gap_months = gap_months

    def get_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate blocked time series splits."""
        splits = []

        # Start from 2021 to have enough training data
        start_year = 2021
        end_year = 2025

        for year in range(start_year, end_year):
            for start_month in [1, 4, 7, 10]:  # Quarterly splits

                # Training period: previous 12 months
                train_start = pd.Timestamp(year - 1, start_month, 1)
                train_end = pd.Timestamp(year, start_month, 1) - pd.Timedelta(days=1)

                # Gap period (avoid data leakage)
                test_start = pd.Timestamp(year, start_month, 1) + pd.Timedelta(
                    days=30 * self.gap_months
                )
                test_end = test_start + pd.Timedelta(days=90)  # 3 months test

                # Skip if beyond our data range
                if test_end > df["datetime"].max():
                    continue

                train_data = df[
                    (df["datetime"] >= train_start) & (df["datetime"] <= train_end)
                ].copy()

                test_data = df[
                    (df["datetime"] >= test_start) & (df["datetime"] <= test_end)
                ].copy()

                if len(train_data) > 1000 and len(test_data) > 100:
                    splits.append((train_data, test_data))

        log.info(f"Generated {len(splits)} blocked time series splits")
        return splits


class SeasonalValidator:
    """Seasonal split validation."""

    def get_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate seasonal splits."""
        splits = []

        # Define seasons
        seasons = {
            "Spring": [3, 4, 5],
            "Summer": [6, 7, 8],
            "Fall": [9, 10, 11],
            "Winter": [12, 1, 2],
        }

        for season_name, months in seasons.items():
            # Train on all years except 2025 for this season
            train_data = df[
                (df["month"].isin(months))
                & (df["year"].isin([2020, 2021, 2022, 2023, 2024]))
            ].copy()

            # Test on 2025 for this season (if available)
            test_data = df[(df["month"].isin(months)) & (df["year"] == 2025)].copy()

            if len(train_data) > 1000 and len(test_data) > 100:
                splits.append((train_data, test_data))

        log.info(f"Generated {len(splits)} seasonal splits")
        return splits


class GeographicValidator:
    """Geographic cross-validation."""

    def get_splits(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate geographic splits."""
        splits = []
        cities = df["city"].unique()

        # Use 2020-2024 data for geographic validation
        validation_data = df[df["year"] <= 2024].copy()

        for test_city in cities:
            train_cities = [city for city in cities if city != test_city]

            train_data = validation_data[
                validation_data["city"].isin(train_cities)
            ].copy()

            test_data = validation_data[validation_data["city"] == test_city].copy()

            if len(train_data) > 1000 and len(test_data) > 1000:
                splits.append((train_data, test_data))

        log.info(f"Generated {len(splits)} geographic splits")
        return splits


def evaluate_model_on_split(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    model: Any,
    pollutant: str,
) -> Dict[str, float]:
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
            model.fit(X_train_scaled, y_train)
            ensemble_pred = model.predict(X_test_scaled)

        # Calculate metrics
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
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
        log.warning(f"Error evaluating {model_name} for {pollutant}: {e}")
        return None


def run_validation_strategy(
    df: pd.DataFrame, feature_cols: List[str], strategy_name: str, validator: Any
) -> pd.DataFrame:
    """Run a complete validation strategy."""

    log.info(f"Running {strategy_name} validation...")

    splits = validator.get_splits(df)
    models = create_models()
    pollutants = ["pm25", "pm10", "no2", "o3"]

    results = []

    for split_idx, (train_data, test_data) in enumerate(splits):
        log.info(f"Processing split {split_idx + 1}/{len(splits)}")

        for pollutant in pollutants:
            for model_name, model in models.items():

                # Create fresh model instance for each evaluation
                if model_name != "simple_average":
                    if model_name == "ridge_ensemble":
                        model = Ridge(alpha=1.0, random_state=42)
                    elif model_name == "random_forest":
                        model = RandomForestRegressor(
                            n_estimators=100, random_state=42, n_jobs=-1, max_depth=8
                        )
                    elif model_name == "gradient_boosting":
                        model = GradientBoostingRegressor(
                            n_estimators=100,
                            random_state=42,
                            max_depth=6,
                            learning_rate=0.1,
                        )

                result = evaluate_model_on_split(
                    train_data, test_data, feature_cols, model_name, model, pollutant
                )

                if result:
                    result.update(
                        {
                            "strategy": strategy_name,
                            "split_idx": split_idx,
                            "model": model_name,
                            "pollutant": pollutant,
                            "train_start": train_data["datetime"].min(),
                            "train_end": train_data["datetime"].max(),
                            "test_start": test_data["datetime"].min(),
                            "test_end": test_data["datetime"].max(),
                        }
                    )
                    results.append(result)

    return pd.DataFrame(results)


def analyze_validation_results(results_df: pd.DataFrame, strategy_name: str) -> None:
    """Analyze and print validation results."""

    print(f"\n{'='*80}")
    print(f"{strategy_name.upper()} VALIDATION RESULTS")
    print(f"{'='*80}")

    # Overall performance by model
    print("\nOVERALL MODEL PERFORMANCE:")
    for model in results_df["model"].unique():
        model_data = results_df[results_df["model"] == model]
        avg_mae = model_data["ensemble_mae"].mean()
        avg_r2 = model_data["ensemble_r2"].mean()
        avg_cams_improvement = model_data["improvement_vs_cams"].mean()
        avg_noaa_improvement = model_data["improvement_vs_noaa"].mean()
        n_evaluations = len(model_data)

        print(f"\n{model.upper().replace('_', ' ')}:")
        print(f"  Average MAE: {avg_mae:.3f} μg/m³")
        print(f"  Average R²: {avg_r2:.3f}")
        print(f"  Improvement vs CAMS: {avg_cams_improvement:+.1f}%")
        print(f"  Improvement vs NOAA: {avg_noaa_improvement:+.1f}%")
        print(f"  Evaluations: {n_evaluations}")

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
        avg_by_model = pollutant_data.groupby("model")["ensemble_mae"].mean()
        best_model = avg_by_model.idxmin()
        best_mae = avg_by_model[best_model]

        pollutant_benchmarks = pollutant_data.groupby("model")[
            ["cams_mae", "noaa_mae"]
        ].mean()
        cams_mae = pollutant_benchmarks.loc[best_model, "cams_mae"]
        noaa_mae = pollutant_benchmarks.loc[best_model, "noaa_mae"]

        cams_imp = (cams_mae - best_mae) / cams_mae * 100
        noaa_imp = (noaa_mae - best_mae) / noaa_mae * 100

        print(f"\n{pollutant.upper()}:")
        print(f"  Best Model: {best_model.replace('_', ' ').title()}")
        print(f"  Best MAE: {best_mae:.3f} μg/m³")
        print(f"  CAMS MAE: {cams_mae:.3f} μg/m³ (improvement: {cams_imp:+.1f}%)")
        print(f"  NOAA MAE: {noaa_mae:.3f} μg/m³ (improvement: {noaa_imp:+.1f}%)")

    # Summary statistics
    total_evaluations = len(results_df)
    total_test_samples = results_df["n_test_samples"].sum()

    print(f"\nSUMMARY:")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Total test samples: {total_test_samples:,}")
    print(
        f"Average test samples per evaluation: {total_test_samples / total_evaluations:.0f}"
    )


def main():
    """Main execution function."""

    # Load 5-year dataset
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    df, feature_cols = load_and_prepare_data(data_path)

    # Run validation strategies
    all_results = []

    # 1. Blocked Time Series Cross-Validation
    blocked_validator = BlockedTimeSeriesValidator(
        train_months=12, test_months=3, gap_months=1
    )
    blocked_results = run_validation_strategy(
        df, feature_cols, "Blocked Time Series", blocked_validator
    )
    all_results.append(blocked_results)

    # 2. Seasonal Split Validation
    seasonal_validator = SeasonalValidator()
    seasonal_results = run_validation_strategy(
        df, feature_cols, "Seasonal Split", seasonal_validator
    )
    all_results.append(seasonal_results)

    # 3. Geographic Cross-Validation
    geographic_validator = GeographicValidator()
    geographic_results = run_validation_strategy(
        df, feature_cols, "Geographic Cross", geographic_validator
    )
    all_results.append(geographic_results)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    results_path = output_dir / "improved_validation_results.csv"
    combined_results.to_csv(results_path, index=False)
    log.info(f"Detailed results saved to {results_path}")

    # Analyze each strategy
    for strategy_results in all_results:
        if len(strategy_results) > 0:
            strategy_name = strategy_results["strategy"].iloc[0]
            analyze_validation_results(strategy_results, strategy_name)

    # Overall hybrid analysis
    print(f"\n{'='*80}")
    print("HYBRID VALIDATION SUMMARY")
    print(f"{'='*80}")

    hybrid_summary = (
        combined_results.groupby(["strategy", "model"])
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

    print("\nAverage Performance by Strategy and Model:")
    print(hybrid_summary)

    return 0


if __name__ == "__main__":
    exit(main())
