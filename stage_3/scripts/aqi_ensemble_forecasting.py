#!/usr/bin/env python3
"""
AQI Ensemble Forecasting

Extends ensemble models to predict Air Quality Index values and categorical classifications.
Focuses on health warning accuracy and minimizing false negatives for public safety.
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
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

# Import our AQI calculation functions
from calculate_aqi import (
    process_dataset_with_aqi,
    calculate_composite_aqi,
    get_aqi_category,
    is_health_warning_required,
    AQI_CATEGORIES,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def load_and_prepare_aqi_data(data_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load dataset and prepare with AQI calculations."""
    log.info(f"Loading dataset and preparing AQI data from {data_path}")

    # Load base dataset
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Sample for reasonable processing time
    df_sampled = df.iloc[::12].copy().reset_index(drop=True)  # Every 12 hours
    log.info(f"Sampled to 12-hourly frequency: {len(df_sampled)} records")

    # Add AQI calculations
    df_aqi = process_dataset_with_aqi(df_sampled)

    # Get feature columns (same as before)
    target_cols = [col for col in df_aqi.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df_aqi.columns if col.startswith("forecast_")]
    aqi_cols = [
        col
        for col in df_aqi.columns
        if col.startswith("aqi_") or col.startswith("health_")
    ]

    exclude_cols = (
        {"city", "datetime", "date", "forecast_made_date"}
        | set(target_cols)
        | set(benchmark_cols)
        | set(aqi_cols)
    )

    feature_cols = []
    for col in df_aqi.columns:
        if col not in exclude_cols:
            if df_aqi[col].dtype in ["int64", "float64", "int32", "float32"]:
                feature_cols.append(col)
            elif col == "week_position":
                df_aqi[col] = (df_aqi[col] == "weekend").astype(int)
                feature_cols.append(col)

    log.info(f"Using {len(feature_cols)} features for AQI forecasting")
    log.info(f"Time range: {df_aqi['datetime'].min()} to {df_aqi['datetime'].max()}")

    return df_aqi, feature_cols


def create_aqi_ensemble_models() -> Dict[str, Any]:
    """Create ensemble models optimized for AQI prediction."""
    return {
        "simple_average": "simple_average",
        "ridge_ensemble": Ridge(alpha=1.0, random_state=42),
        "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=8
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1
        ),
    }


def predict_aqi_from_concentrations(
    predicted_concentrations: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert predicted pollutant concentrations to AQI values and categories.

    Args:
        predicted_concentrations: Dict of {pollutant: predicted_values}

    Returns:
        Tuple of (predicted_aqi_values, predicted_categories)
    """
    n_samples = len(next(iter(predicted_concentrations.values())))
    predicted_aqis = np.zeros(n_samples)
    predicted_categories = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Get concentrations for this sample
        concentrations = {
            pollutant: values[i]
            for pollutant, values in predicted_concentrations.items()
            if not np.isnan(values[i])
        }

        if concentrations:
            # Calculate composite AQI
            composite_aqi, _ = calculate_composite_aqi(concentrations)
            predicted_aqis[i] = composite_aqi if not pd.isna(composite_aqi) else 0

            # Convert to category (0: Good, 1: Moderate, 2: Unhealthy Sensitive, 3: Unhealthy, 4: Very Unhealthy, 5: Hazardous)
            predicted_categories[i] = aqi_to_category_index(composite_aqi)
        else:
            predicted_aqis[i] = 0
            predicted_categories[i] = 0

    return predicted_aqis, predicted_categories


def aqi_to_category_index(aqi_value: float) -> int:
    """Convert AQI value to category index (0-5)."""
    if pd.isna(aqi_value):
        return 0

    if aqi_value <= 50:
        return 0  # Good
    elif aqi_value <= 100:
        return 1  # Moderate
    elif aqi_value <= 150:
        return 2  # Unhealthy for Sensitive Groups
    elif aqi_value <= 200:
        return 3  # Unhealthy
    elif aqi_value <= 300:
        return 4  # Very Unhealthy
    else:
        return 5  # Hazardous


def calculate_health_warning_metrics(
    y_true_aqi: np.ndarray, y_pred_aqi: np.ndarray
) -> Dict[str, float]:
    """
    Calculate health warning specific metrics.

    Args:
        y_true_aqi: Actual AQI values
        y_pred_aqi: Predicted AQI values

    Returns:
        Dictionary of health-focused metrics
    """
    # Convert to warning flags
    true_sensitive_warnings = np.array(
        [is_health_warning_required(aqi, True) for aqi in y_true_aqi]
    )
    pred_sensitive_warnings = np.array(
        [is_health_warning_required(aqi, True) for aqi in y_pred_aqi]
    )

    true_general_warnings = np.array(
        [is_health_warning_required(aqi, False) for aqi in y_true_aqi]
    )
    pred_general_warnings = np.array(
        [is_health_warning_required(aqi, False) for aqi in y_pred_aqi]
    )

    metrics = {}

    # Sensitive group warning metrics
    if np.sum(true_sensitive_warnings) > 0:
        metrics["sensitive_precision"] = precision_score(
            true_sensitive_warnings, pred_sensitive_warnings, zero_division=0
        )
        metrics["sensitive_recall"] = recall_score(
            true_sensitive_warnings, pred_sensitive_warnings, zero_division=0
        )
        metrics["sensitive_f1"] = f1_score(
            true_sensitive_warnings, pred_sensitive_warnings, zero_division=0
        )
    else:
        metrics["sensitive_precision"] = 1.0
        metrics["sensitive_recall"] = 1.0
        metrics["sensitive_f1"] = 1.0

    # General population warning metrics
    if np.sum(true_general_warnings) > 0:
        metrics["general_precision"] = precision_score(
            true_general_warnings, pred_general_warnings, zero_division=0
        )
        metrics["general_recall"] = recall_score(
            true_general_warnings, pred_general_warnings, zero_division=0
        )
        metrics["general_f1"] = f1_score(
            true_general_warnings, pred_general_warnings, zero_division=0
        )
    else:
        metrics["general_precision"] = 1.0
        metrics["general_recall"] = 1.0
        metrics["general_f1"] = 1.0

    # Overall warning accuracy
    metrics["sensitive_accuracy"] = accuracy_score(
        true_sensitive_warnings, pred_sensitive_warnings
    )
    metrics["general_accuracy"] = accuracy_score(
        true_general_warnings, pred_general_warnings
    )

    # False negative rates (critical for public health)
    if np.sum(true_sensitive_warnings) > 0:
        metrics["sensitive_false_negative_rate"] = np.sum(
            (true_sensitive_warnings == 1) & (pred_sensitive_warnings == 0)
        ) / np.sum(true_sensitive_warnings)
    else:
        metrics["sensitive_false_negative_rate"] = 0.0

    if np.sum(true_general_warnings) > 0:
        metrics["general_false_negative_rate"] = np.sum(
            (true_general_warnings == 1) & (pred_general_warnings == 0)
        ) / np.sum(true_general_warnings)
    else:
        metrics["general_false_negative_rate"] = 0.0

    return metrics


def aqi_walk_forward_validation(
    df_aqi: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Walk-forward validation specifically for AQI prediction and health warnings.
    """
    log.info("Starting AQI-focused walk-forward validation...")

    # Define validation period
    validation_start = pd.Timestamp("2024-09-09")
    validation_end = pd.Timestamp("2025-09-08")

    # Get training and validation data
    initial_train_data = df_aqi[df_aqi["datetime"] < validation_start].copy()
    validation_data = (
        df_aqi[
            (df_aqi["datetime"] >= validation_start)
            & (df_aqi["datetime"] <= validation_end)
        ]
        .copy()
        .sort_values("datetime")
    )

    log.info(f"Initial training data: {len(initial_train_data)} records")
    log.info(f"Validation data: {len(validation_data)} records")

    # Sample validation dates (bi-monthly for reasonable runtime)
    validation_dates = sorted(validation_data["date"].dt.date.unique())
    sampled_dates = validation_dates[::60]  # Every 2 months
    log.info(f"Using bi-monthly sampling: {len(sampled_dates)} validation points")

    models = create_aqi_ensemble_models()
    pollutants = ["pm25", "pm10", "no2", "o3"]
    results = []

    for date_idx, current_date in enumerate(sampled_dates):
        log.info(f"Processing {date_idx + 1}/{len(sampled_dates)}: {current_date}")

        # Get current day's data
        current_day_data = validation_data[
            validation_data["date"].dt.date == current_date
        ].copy()

        if len(current_day_data) == 0:
            continue

        # Update training data
        previous_validation_data = validation_data[
            validation_data["date"].dt.date < current_date
        ].copy()

        if len(previous_validation_data) > 0:
            current_train_data = pd.concat(
                [initial_train_data, previous_validation_data], ignore_index=True
            )
        else:
            current_train_data = initial_train_data.copy()

        # Get actual AQI values for testing
        y_test_aqi = current_day_data["aqi_composite"].values

        # Get benchmark AQI predictions
        benchmark_concentrations = {}
        for pollutant in pollutants:
            cams_col = f"forecast_cams_{pollutant}"
            noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"
            if cams_col in current_day_data.columns:
                benchmark_concentrations[f"cams_{pollutant}"] = current_day_data[
                    cams_col
                ].values
            if noaa_col in current_day_data.columns:
                benchmark_concentrations[f"noaa_{pollutant}"] = current_day_data[
                    noaa_col
                ].values

        # Calculate benchmark AQI values
        cams_aqi = np.zeros(len(current_day_data))
        noaa_aqi = np.zeros(len(current_day_data))

        for i in range(len(current_day_data)):
            # CAMS AQI
            cams_conc = {
                p: benchmark_concentrations[f"cams_{p}"][i]
                for p in pollutants
                if f"cams_{p}" in benchmark_concentrations
            }
            if cams_conc:
                cams_aqi_val, _ = calculate_composite_aqi(cams_conc)
                cams_aqi[i] = cams_aqi_val if not pd.isna(cams_aqi_val) else 0

            # NOAA AQI
            noaa_conc = {
                p: benchmark_concentrations[f"noaa_{p}"][i]
                for p in pollutants
                if f"noaa_{p}" in benchmark_concentrations
            }
            if noaa_conc:
                noaa_aqi_val, _ = calculate_composite_aqi(noaa_conc)
                noaa_aqi[i] = noaa_aqi_val if not pd.isna(noaa_aqi_val) else 0

        # Test each ensemble model
        for model_name, model in models.items():
            try:
                # Predict individual pollutant concentrations
                predicted_concentrations = {}

                for pollutant in pollutants:
                    target_col = f"actual_{pollutant}"
                    cams_col = f"forecast_cams_{pollutant}"
                    noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

                    if target_col not in current_train_data.columns:
                        continue

                    # Prepare data for this pollutant
                    X_train = current_train_data[feature_cols].fillna(0).values
                    y_train = current_train_data[target_col].values
                    X_test = current_day_data[feature_cols].fillna(0).values

                    cams_pred = current_day_data[cams_col].values
                    noaa_pred = current_day_data[noaa_col].values

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Make predictions
                    if model_name == "simple_average":
                        ensemble_pred = (cams_pred + noaa_pred) / 2
                    else:
                        # Fresh model instance
                        if model_name == "ridge_ensemble":
                            model_instance = Ridge(alpha=1.0, random_state=42)
                        elif model_name == "elastic_net":
                            model_instance = ElasticNet(
                                alpha=1.0, l1_ratio=0.5, random_state=42
                            )
                        elif model_name == "random_forest":
                            model_instance = RandomForestRegressor(
                                n_estimators=100,
                                random_state=42,
                                n_jobs=-1,
                                max_depth=8,
                            )
                        elif model_name == "gradient_boosting":
                            model_instance = GradientBoostingRegressor(
                                n_estimators=100,
                                random_state=42,
                                max_depth=6,
                                learning_rate=0.1,
                            )

                        model_instance.fit(X_train_scaled, y_train)
                        ensemble_pred = model_instance.predict(X_test_scaled)

                    predicted_concentrations[pollutant] = ensemble_pred

                # Convert predicted concentrations to AQI
                if predicted_concentrations:
                    pred_aqi, pred_categories = predict_aqi_from_concentrations(
                        predicted_concentrations
                    )

                    # Calculate AQI metrics
                    aqi_mae = mean_absolute_error(
                        y_test_aqi[~np.isnan(y_test_aqi)],
                        pred_aqi[~np.isnan(y_test_aqi)],
                    )
                    aqi_r2 = r2_score(
                        y_test_aqi[~np.isnan(y_test_aqi)],
                        pred_aqi[~np.isnan(y_test_aqi)],
                    )

                    # Calculate health warning metrics
                    health_metrics = calculate_health_warning_metrics(
                        y_test_aqi, pred_aqi
                    )

                    # Calculate benchmark AQI metrics
                    cams_aqi_mae = mean_absolute_error(
                        y_test_aqi[~np.isnan(y_test_aqi)],
                        cams_aqi[~np.isnan(y_test_aqi)],
                    )
                    noaa_aqi_mae = mean_absolute_error(
                        y_test_aqi[~np.isnan(y_test_aqi)],
                        noaa_aqi[~np.isnan(y_test_aqi)],
                    )

                    # Store results
                    result = {
                        "date": current_date,
                        "model": model_name,
                        "aqi_mae": aqi_mae,
                        "aqi_r2": aqi_r2,
                        "cams_aqi_mae": cams_aqi_mae,
                        "noaa_aqi_mae": noaa_aqi_mae,
                        "aqi_improvement_vs_cams": (
                            (cams_aqi_mae - aqi_mae) / cams_aqi_mae * 100
                            if cams_aqi_mae > 0
                            else 0
                        ),
                        "aqi_improvement_vs_noaa": (
                            (noaa_aqi_mae - aqi_mae) / noaa_aqi_mae * 100
                            if noaa_aqi_mae > 0
                            else 0
                        ),
                        "n_test_samples": len(y_test_aqi),
                        "n_train_samples": len(current_train_data),
                    }

                    # Add health warning metrics
                    result.update(health_metrics)

                    results.append(result)

            except Exception as e:
                log.warning(f"Error with {model_name} on {current_date}: {e}")
                continue

    return pd.DataFrame(results)


def analyze_aqi_results(results_df: pd.DataFrame) -> None:
    """Analyze AQI forecasting results with focus on health warning accuracy."""

    print("\n" + "=" * 80)
    print("AQI ENSEMBLE FORECASTING RESULTS")
    print("Health Warning Focus - Walk-Forward Validation")
    print("=" * 80)

    # Overall AQI prediction performance
    print("\nAQI PREDICTION PERFORMANCE:")
    overall_summary = (
        results_df.groupby("model")
        .agg(
            {
                "aqi_mae": "mean",
                "aqi_r2": "mean",
                "aqi_improvement_vs_cams": "mean",
                "aqi_improvement_vs_noaa": "mean",
                "n_test_samples": "sum",
            }
        )
        .round(3)
        .sort_values("aqi_mae")
    )

    for rank, (model, data) in enumerate(overall_summary.iterrows(), 1):
        print(f"\n{rank}. {model.upper().replace('_', ' ')}:")
        print(f"   AQI MAE: {data['aqi_mae']:.1f}")
        print(f"   AQI R²: {data['aqi_r2']:.3f}")
        print(f"   Improvement vs CAMS: {data['aqi_improvement_vs_cams']:+.1f}%")
        print(f"   Improvement vs NOAA: {data['aqi_improvement_vs_noaa']:+.1f}%")

    # Health warning performance (CRITICAL METRICS)
    print(f"\nHEALTH WARNING PERFORMANCE (CRITICAL FOR PUBLIC SAFETY):")

    health_metrics = [
        "sensitive_recall",
        "sensitive_precision",
        "sensitive_f1",
        "general_recall",
        "general_precision",
        "general_f1",
        "sensitive_false_negative_rate",
        "general_false_negative_rate",
    ]

    health_summary = results_df.groupby("model")[health_metrics].mean().round(3)

    print(f"\nSENSITIVE GROUPS WARNING (AQI ≥ 101):")
    for model in health_summary.index:
        data = health_summary.loc[model]
        print(f"\n{model.replace('_', ' ').title()}:")
        print(
            f"  Recall (Sensitivity): {data['sensitive_recall']:.3f} (capture {data['sensitive_recall']*100:.1f}% of unhealthy days)"
        )
        print(f"  Precision: {data['sensitive_precision']:.3f} (avoid false alarms)")
        print(f"  F1-Score: {data['sensitive_f1']:.3f}")
        print(
            f"  False Negative Rate: {data['sensitive_false_negative_rate']:.3f} (CRITICAL: missed warnings)"
        )

    print(f"\nGENERAL POPULATION WARNING (AQI ≥ 151):")
    for model in health_summary.index:
        data = health_summary.loc[model]
        print(f"\n{model.replace('_', ' ').title()}:")
        print(f"  Recall (Sensitivity): {data['general_recall']:.3f}")
        print(f"  Precision: {data['general_precision']:.3f}")
        print(f"  F1-Score: {data['general_f1']:.3f}")
        print(f"  False Negative Rate: {data['general_false_negative_rate']:.3f}")

    # Model recommendation based on health safety
    print(f"\nMODEL RECOMMENDATION FOR PUBLIC HEALTH:")

    # Prioritize low false negative rate for sensitive groups
    best_for_health = health_summary.sort_values("sensitive_false_negative_rate").index[
        0
    ]
    best_fnr = health_summary.loc[best_for_health, "sensitive_false_negative_rate"]
    best_recall = health_summary.loc[best_for_health, "sensitive_recall"]

    print(
        f"🏥 RECOMMENDED FOR HEALTH PROTECTION: {best_for_health.replace('_', ' ').title()}"
    )
    print(f"   Captures {best_recall*100:.1f}% of unhealthy days for sensitive groups")
    print(f"   Misses only {best_fnr*100:.1f}% of critical health warnings")

    # Best overall AQI prediction
    best_aqi = overall_summary.index[0]
    best_mae = overall_summary.loc[best_aqi, "aqi_mae"]

    print(f"\n📊 BEST AQI PREDICTION ACCURACY: {best_aqi.replace('_', ' ').title()}")
    print(f"   AQI MAE: {best_mae:.1f}")

    print(f"\nValidation Summary:")
    print(f"Total evaluations: {len(results_df)}")
    print(f"Total test samples: {results_df['n_test_samples'].sum():,}")
    print(f"Models tested: {results_df['model'].nunique()}")

    print("\n" + "=" * 80)
    print("CONCLUSION: Prioritize model with lowest false negative rate")
    print(
        "for public health protection, even if AQI prediction is slightly less accurate."
    )
    print("=" * 80)


def main():
    """Main execution function for AQI ensemble forecasting."""

    # Load data with AQI calculations
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    df_aqi, feature_cols = load_and_prepare_aqi_data(data_path)

    # Generate AQI summary for the dataset
    from calculate_aqi import generate_aqi_summary_report

    generate_aqi_summary_report(df_aqi)

    # Run AQI-focused walk-forward validation
    log.info("Starting AQI ensemble forecasting validation...")
    results_df = aqi_walk_forward_validation(df_aqi, feature_cols)

    if len(results_df) == 0:
        log.error("No AQI validation results generated")
        return 1

    # Save results
    results_path = output_dir / "aqi_ensemble_forecasting_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"AQI results saved to {results_path}")

    # Analyze results
    analyze_aqi_results(results_df)

    # Save processed dataset with AQI
    aqi_dataset_path = output_dir / "dataset_with_aqi.csv"
    df_aqi.to_csv(aqi_dataset_path, index=False)
    log.info(f"Dataset with AQI saved to {aqi_dataset_path}")

    return 0


if __name__ == "__main__":
    exit(main())
