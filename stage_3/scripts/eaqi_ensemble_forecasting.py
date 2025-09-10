#!/usr/bin/env python3
"""
EAQI Ensemble Forecasting

Extends ensemble models to predict European Air Quality Index (EAQI) values and categorical classifications.
Focuses on health warning accuracy using European standards and minimizing false negatives for public safety.
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

# Import our dual AQI calculation functions
from calculate_aqi_dual_standard import (
    process_dataset_with_dual_aqi,
    calculate_composite_aqi,
    get_aqi_category,
    is_health_warning_required,
    convert_units_for_eaqi,
    EAQI_CATEGORIES,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def load_and_prepare_eaqi_data(data_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load dataset and prepare with EAQI calculations."""
    log.info(f"Loading dataset and preparing EAQI data from {data_path}")

    # Load base dataset
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Sample for reasonable processing time
    df_sampled = df.iloc[::12].copy().reset_index(drop=True)  # Every 12 hours
    log.info(f"Sampled to 12-hourly frequency: {len(df_sampled)} records")

    # Add EAQI calculations (European standard only)
    df_eaqi = process_dataset_with_dual_aqi(df_sampled, standards=["EAQI"])

    # Get feature columns (same as before)
    target_cols = [col for col in df_eaqi.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df_eaqi.columns if col.startswith("forecast_")]
    eaqi_cols = [
        col
        for col in df_eaqi.columns
        if col.startswith("aqi_") or col.startswith("health_")
    ]

    exclude_cols = (
        {"city", "datetime", "date", "forecast_made_date"}
        | set(target_cols)
        | set(benchmark_cols)
        | set(eaqi_cols)
    )

    feature_cols = []
    for col in df_eaqi.columns:
        if col not in exclude_cols:
            if df_eaqi[col].dtype in ["int64", "float64", "int32", "float32"]:
                feature_cols.append(col)
            elif col == "week_position":
                df_eaqi[col] = (df_eaqi[col] == "weekend").astype(int)
                feature_cols.append(col)

    log.info(f"Using {len(feature_cols)} features for EAQI forecasting")
    log.info(f"Time range: {df_eaqi['datetime'].min()} to {df_eaqi['datetime'].max()}")

    return df_eaqi, feature_cols


def create_eaqi_ensemble_models() -> Dict[str, Any]:
    """Create ensemble models optimized for EAQI prediction."""
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


def predict_eaqi_from_concentrations(
    predicted_concentrations: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert predicted pollutant concentrations to EAQI values and categories.

    Args:
        predicted_concentrations: Dict of {pollutant: predicted_values}

    Returns:
        Tuple of (predicted_eaqi_values, predicted_categories)
    """
    n_samples = len(next(iter(predicted_concentrations.values())))
    predicted_eaqis = np.zeros(n_samples)
    predicted_categories = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Get concentrations for this sample and convert units for EAQI
        concentrations = {}
        for pollutant, values in predicted_concentrations.items():
            if not np.isnan(values[i]):
                if pollutant in ["no2", "o3"]:
                    # Convert from ppb to ug/m3 for EAQI
                    conc = convert_units_for_eaqi(values[i], pollutant, "ppb")
                else:
                    conc = values[i]
                concentrations[pollutant] = conc

        if concentrations:
            # Calculate composite EAQI
            composite_eaqi, _ = calculate_composite_aqi(concentrations, "EAQI")
            predicted_eaqis[i] = composite_eaqi if not pd.isna(composite_eaqi) else 1

            # EAQI categories are already 1-6
            predicted_categories[i] = (
                int(composite_eaqi) if not pd.isna(composite_eaqi) else 1
            )
        else:
            predicted_eaqis[i] = 1  # Very Good (default for EAQI)
            predicted_categories[i] = 1

    return predicted_eaqis, predicted_categories


def calculate_eaqi_health_warning_metrics(
    y_true_eaqi: np.ndarray, y_pred_eaqi: np.ndarray
) -> Dict[str, float]:
    """
    Calculate health warning specific metrics for EAQI.

    Args:
        y_true_eaqi: Actual EAQI values (1-6)
        y_pred_eaqi: Predicted EAQI values (1-6)

    Returns:
        Dictionary of health-focused metrics
    """
    # Convert to warning flags using EAQI thresholds
    true_sensitive_warnings = np.array(
        [is_health_warning_required(eaqi, True, "EAQI") for eaqi in y_true_eaqi]
    )
    pred_sensitive_warnings = np.array(
        [is_health_warning_required(eaqi, True, "EAQI") for eaqi in y_pred_eaqi]
    )

    true_general_warnings = np.array(
        [is_health_warning_required(eaqi, False, "EAQI") for eaqi in y_true_eaqi]
    )
    pred_general_warnings = np.array(
        [is_health_warning_required(eaqi, False, "EAQI") for eaqi in y_pred_eaqi]
    )

    metrics = {}

    # Sensitive group warning metrics (Level >= 4: Poor)
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

    # General population warning metrics (Level >= 5: Very Poor)
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


def eaqi_walk_forward_validation(
    df_eaqi: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Walk-forward validation specifically for EAQI prediction and health warnings.
    """
    log.info("Starting EAQI-focused walk-forward validation...")

    # Define validation period
    validation_start = pd.Timestamp("2024-09-09")
    validation_end = pd.Timestamp("2025-09-08")

    # Get training and validation data
    initial_train_data = df_eaqi[df_eaqi["datetime"] < validation_start].copy()
    validation_data = (
        df_eaqi[
            (df_eaqi["datetime"] >= validation_start)
            & (df_eaqi["datetime"] <= validation_end)
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

    models = create_eaqi_ensemble_models()
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

        # Get actual EAQI values for testing
        y_test_eaqi = current_day_data["aqi_composite_eaqi"].values

        # Get benchmark EAQI predictions
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

        # Calculate benchmark EAQI values
        cams_eaqi = np.ones(len(current_day_data))  # Default to "Very Good"
        noaa_eaqi = np.ones(len(current_day_data))

        for i in range(len(current_day_data)):
            # CAMS EAQI
            cams_conc = {}
            for p in pollutants:
                if f"cams_{p}" in benchmark_concentrations:
                    conc = benchmark_concentrations[f"cams_{p}"][i]
                    if p in ["no2", "o3"]:
                        # Convert ppb to ug/m3 for EAQI
                        conc = convert_units_for_eaqi(conc, p, "ppb")
                    cams_conc[p] = conc

            if cams_conc:
                cams_eaqi_val, _ = calculate_composite_aqi(cams_conc, "EAQI")
                cams_eaqi[i] = cams_eaqi_val if not pd.isna(cams_eaqi_val) else 1

            # NOAA EAQI
            noaa_conc = {}
            for p in pollutants:
                if f"noaa_{p}" in benchmark_concentrations:
                    conc = benchmark_concentrations[f"noaa_{p}"][i]
                    if p in ["no2", "o3"]:
                        # Convert ppb to ug/m3 for EAQI
                        conc = convert_units_for_eaqi(conc, p, "ppb")
                    noaa_conc[p] = conc

            if noaa_conc:
                noaa_eaqi_val, _ = calculate_composite_aqi(noaa_conc, "EAQI")
                noaa_eaqi[i] = noaa_eaqi_val if not pd.isna(noaa_eaqi_val) else 1

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

                # Convert predicted concentrations to EAQI
                if predicted_concentrations:
                    pred_eaqi, pred_categories = predict_eaqi_from_concentrations(
                        predicted_concentrations
                    )

                    # Calculate EAQI metrics
                    eaqi_mae = mean_absolute_error(
                        y_test_eaqi[~np.isnan(y_test_eaqi)],
                        pred_eaqi[~np.isnan(y_test_eaqi)],
                    )

                    # Calculate health warning metrics
                    health_metrics = calculate_eaqi_health_warning_metrics(
                        y_test_eaqi, pred_eaqi
                    )

                    # Calculate benchmark EAQI metrics
                    cams_eaqi_mae = mean_absolute_error(
                        y_test_eaqi[~np.isnan(y_test_eaqi)],
                        cams_eaqi[~np.isnan(y_test_eaqi)],
                    )
                    noaa_eaqi_mae = mean_absolute_error(
                        y_test_eaqi[~np.isnan(y_test_eaqi)],
                        noaa_eaqi[~np.isnan(y_test_eaqi)],
                    )

                    # Store results
                    result = {
                        "date": current_date,
                        "model": model_name,
                        "eaqi_mae": eaqi_mae,
                        "cams_eaqi_mae": cams_eaqi_mae,
                        "noaa_eaqi_mae": noaa_eaqi_mae,
                        "eaqi_improvement_vs_cams": (
                            (cams_eaqi_mae - eaqi_mae) / cams_eaqi_mae * 100
                            if cams_eaqi_mae > 0
                            else 0
                        ),
                        "eaqi_improvement_vs_noaa": (
                            (noaa_eaqi_mae - eaqi_mae) / noaa_eaqi_mae * 100
                            if noaa_eaqi_mae > 0
                            else 0
                        ),
                        "n_test_samples": len(y_test_eaqi),
                        "n_train_samples": len(current_train_data),
                    }

                    # Add health warning metrics
                    result.update(health_metrics)

                    results.append(result)

            except Exception as e:
                log.warning(f"Error with {model_name} on {current_date}: {e}")
                continue

    return pd.DataFrame(results)


def analyze_eaqi_results(results_df: pd.DataFrame) -> None:
    """Analyze EAQI forecasting results with focus on health warning accuracy."""

    print("\n" + "=" * 80)
    print("EAQI ENSEMBLE FORECASTING RESULTS")
    print("European Air Quality Index - Health Warning Focus")
    print("Walk-Forward Validation (2024-09-09 to 2025-09-08)")
    print("=" * 80)

    # Overall EAQI prediction performance
    print("\nEAQI PREDICTION PERFORMANCE:")
    overall_summary = (
        results_df.groupby("model")
        .agg(
            {
                "eaqi_mae": "mean",
                "eaqi_improvement_vs_cams": "mean",
                "eaqi_improvement_vs_noaa": "mean",
                "n_test_samples": "sum",
            }
        )
        .round(3)
        .sort_values("eaqi_mae")
    )

    for rank, (model, data) in enumerate(overall_summary.iterrows(), 1):
        print(f"\n{rank}. {model.upper().replace('_', ' ')}:")
        print(f"   EAQI MAE: {data['eaqi_mae']:.2f}")
        print(f"   Improvement vs CAMS: {data['eaqi_improvement_vs_cams']:+.1f}%")
        print(f"   Improvement vs NOAA: {data['eaqi_improvement_vs_noaa']:+.1f}%")

    # Health warning performance (CRITICAL METRICS)
    print(f"\nHEALTH WARNING PERFORMANCE (EUROPEAN STANDARDS):")

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

    print(f"\nSENSITIVE GROUPS WARNING (EAQI >= 4: Poor):")
    for model in health_summary.index:
        data = health_summary.loc[model]
        print(f"\n{model.replace('_', ' ').title()}:")
        print(
            f"  Recall (Sensitivity): {data['sensitive_recall']:.3f} (capture {data['sensitive_recall']*100:.1f}% of poor air days)"
        )
        print(f"  Precision: {data['sensitive_precision']:.3f} (avoid false alarms)")
        print(f"  F1-Score: {data['sensitive_f1']:.3f}")
        print(
            f"  False Negative Rate: {data['sensitive_false_negative_rate']:.3f} (CRITICAL: missed warnings)"
        )

    print(f"\nGENERAL POPULATION WARNING (EAQI >= 5: Very Poor):")
    for model in health_summary.index:
        data = health_summary.loc[model]
        print(f"\n{model.replace('_', ' ').title()}:")
        print(f"  Recall (Sensitivity): {data['general_recall']:.3f}")
        print(f"  Precision: {data['general_precision']:.3f}")
        print(f"  F1-Score: {data['general_f1']:.3f}")
        print(f"  False Negative Rate: {data['general_false_negative_rate']:.3f}")

    # Model recommendation based on health safety
    print(f"\nMODEL RECOMMENDATION FOR EUROPEAN PUBLIC HEALTH:")

    # Prioritize low false negative rate for sensitive groups
    best_for_health = health_summary.sort_values("sensitive_false_negative_rate").index[
        0
    ]
    best_fnr = health_summary.loc[best_for_health, "sensitive_false_negative_rate"]
    best_recall = health_summary.loc[best_for_health, "sensitive_recall"]

    print(
        f"RECOMMENDED FOR HEALTH PROTECTION: {best_for_health.replace('_', ' ').title()}"
    )
    print(f"   Captures {best_recall*100:.1f}% of poor air quality days")
    print(f"   Misses only {best_fnr*100:.1f}% of critical health warnings")

    # Best overall EAQI prediction
    best_eaqi = overall_summary.index[0]
    best_mae = overall_summary.loc[best_eaqi, "eaqi_mae"]

    print(f"\nBEST EAQI PREDICTION ACCURACY: {best_eaqi.replace('_', ' ').title()}")
    print(f"   EAQI MAE: {best_mae:.2f}")

    print(f"\nValidation Summary:")
    print(f"Total evaluations: {len(results_df)}")
    print(f"Total test samples: {results_df['n_test_samples'].sum():,}")
    print(f"Models tested: {results_df['model'].nunique()}")

    print("\n" + "=" * 80)
    print("EAQI CONCLUSION: European standards provide more conservative")
    print("health warnings. Prioritize model with lowest false negative rate.")
    print("=" * 80)


def main():
    """Main execution function for EAQI ensemble forecasting."""

    # Load data with EAQI calculations
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    df_eaqi, feature_cols = load_and_prepare_eaqi_data(data_path)

    # Generate EAQI summary for the dataset
    from calculate_aqi_dual_standard import compare_aqi_standards

    # Show EAQI distribution
    print("\n" + "=" * 80)
    print("EUROPEAN AIR QUALITY INDEX (EAQI) DATASET SUMMARY")
    print("=" * 80)

    if "aqi_composite_eaqi" in df_eaqi.columns:
        eaqi_stats = df_eaqi["aqi_composite_eaqi"].describe()
        print(f"\nEAQI Statistics ({int(eaqi_stats['count']):,} observations):")
        print(f"Mean EAQI: {eaqi_stats['mean']:.2f}")
        print(f"Min EAQI: {eaqi_stats['min']:.0f}")
        print(f"Max EAQI: {eaqi_stats['max']:.0f}")

        # Category distribution
        if "aqi_level_eaqi" in df_eaqi.columns:
            print(f"\nEAQI Category Distribution:")
            category_counts = df_eaqi["aqi_level_eaqi"].value_counts()
            for category, count in category_counts.items():
                pct = (count / len(df_eaqi)) * 100
                print(f"  {category}: {count:,} ({pct:.1f}%)")

        # Health warning analysis
        if "health_warning_sensitive_eaqi" in df_eaqi.columns:
            sensitive_warnings = df_eaqi["health_warning_sensitive_eaqi"].sum()
            sensitive_pct = (sensitive_warnings / len(df_eaqi)) * 100
            print(f"\nEuropean Health Warnings:")
            print(
                f"  Sensitive Groups (EAQI >= 4): {sensitive_warnings:,} days ({sensitive_pct:.1f}%)"
            )

            general_warnings = df_eaqi["health_warning_general_eaqi"].sum()
            general_pct = (general_warnings / len(df_eaqi)) * 100
            print(
                f"  General Population (EAQI >= 5): {general_warnings:,} days ({general_pct:.1f}%)"
            )

    # Run EAQI-focused walk-forward validation
    log.info("Starting EAQI ensemble forecasting validation...")
    results_df = eaqi_walk_forward_validation(df_eaqi, feature_cols)

    if len(results_df) == 0:
        log.error("No EAQI validation results generated")
        return 1

    # Save results
    results_path = output_dir / "eaqi_ensemble_forecasting_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"EAQI results saved to {results_path}")

    # Analyze results
    analyze_eaqi_results(results_df)

    # Save processed dataset with EAQI
    eaqi_dataset_path = output_dir / "dataset_with_eaqi.csv"
    df_eaqi.to_csv(eaqi_dataset_path, index=False)
    log.info(f"Dataset with EAQI saved to {eaqi_dataset_path}")

    return 0


if __name__ == "__main__":
    exit(main())
