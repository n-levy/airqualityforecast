#!/usr/bin/env python3
"""
Quick EAQI Model Comparison

Fast comparison of ensemble models using European Air Quality Index (EAQI) standard.
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import dual AQI calculation functions
from calculate_aqi_dual_standard import (
    process_dataset_with_dual_aqi,
    calculate_composite_aqi,
    get_aqi_category,
    is_health_warning_required,
    convert_units_for_eaqi,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def quick_eaqi_model_comparison():
    """Quick ensemble comparison using EAQI with simple train/test split."""

    # Load and sample data heavily for speed
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    log.info("Loading and heavily sampling dataset for EAQI comparison...")

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Heavy sampling - every 48 hours for very fast processing
    df_sampled = df.iloc[::48].copy().reset_index(drop=True)
    log.info(f"Heavily sampled dataset: {len(df_sampled)} records")

    # Add EAQI calculations
    df_eaqi = process_dataset_with_dual_aqi(df_sampled, standards=["EAQI"])

    # Get features
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

    # Use subset of features for speed
    feature_cols = feature_cols[:20]
    log.info(f"Using {len(feature_cols)} features for speed")

    # Models with reduced complexity
    models = {
        "simple_average": "simple_average",
        "ridge_ensemble": Ridge(alpha=1.0, random_state=42),
        "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=20, random_state=42, max_depth=4, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=20, random_state=42, max_depth=3, learning_rate=0.1
        ),
    }

    pollutants = ["pm25", "pm10"]  # Just 2 pollutants for speed
    results = []

    # Get actual EAQI values
    y_eaqi_all = df_eaqi["aqi_composite_eaqi"].values

    for pollutant in pollutants:
        log.info(f"Processing {pollutant} for EAQI prediction...")

        target_col = f"actual_{pollutant}"
        cams_col = f"forecast_cams_{pollutant}"
        noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

        # Prepare data
        X = df_eaqi[feature_cols].fillna(0).values
        y = df_eaqi[target_col].values
        cams_pred_all = df_eaqi[cams_col].values
        noaa_pred_all = df_eaqi[noaa_col].values

        # Simple train/test split (70/30)
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        cams_test = cams_pred_all[test_idx]
        noaa_test = noaa_pred_all[test_idx]

        # Get actual EAQI for test set
        y_eaqi_test = y_eaqi_all[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for model_name, model in models.items():
            try:
                if model_name == "simple_average":
                    ensemble_pred = (cams_test + noaa_test) / 2
                else:
                    # Fresh model instance
                    if model_name == "ridge_ensemble":
                        model = Ridge(alpha=1.0, random_state=42)
                    elif model_name == "elastic_net":
                        model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
                    elif model_name == "random_forest":
                        model = RandomForestRegressor(
                            n_estimators=20, random_state=42, max_depth=4, n_jobs=-1
                        )
                    elif model_name == "gradient_boosting":
                        model = GradientBoostingRegressor(
                            n_estimators=20,
                            random_state=42,
                            max_depth=3,
                            learning_rate=0.1,
                        )

                    model.fit(X_train_scaled, y_train)
                    ensemble_pred = model.predict(X_test_scaled)

                # Calculate individual pollutant metrics
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                cams_mae = mean_absolute_error(y_test, cams_test)
                noaa_mae = mean_absolute_error(y_test, noaa_test)

                # Calculate improvements
                cams_improvement = (
                    (cams_mae - ensemble_mae) / cams_mae * 100 if cams_mae > 0 else 0
                )
                noaa_improvement = (
                    (noaa_mae - ensemble_mae) / noaa_mae * 100 if noaa_mae > 0 else 0
                )

                results.append(
                    {
                        "model": model_name,
                        "pollutant": pollutant,
                        "ensemble_mae": ensemble_mae,
                        "cams_mae": cams_mae,
                        "noaa_mae": noaa_mae,
                        "improvement_vs_cams": cams_improvement,
                        "improvement_vs_noaa": noaa_improvement,
                        "n_test_samples": len(y_test),
                    }
                )

            except Exception as e:
                log.warning(f"Error with {model_name} for {pollutant}: {e}")
                continue

    # Now test EAQI prediction directly
    log.info("Testing direct EAQI prediction...")

    # Split EAQI data
    eaqi_train_idx, eaqi_test_idx = train_test_split(
        np.arange(len(df_eaqi)), test_size=0.3, random_state=42
    )

    X_eaqi_train = df_eaqi.iloc[eaqi_train_idx][feature_cols].fillna(0).values
    X_eaqi_test = df_eaqi.iloc[eaqi_test_idx][feature_cols].fillna(0).values
    y_eaqi_train = df_eaqi.iloc[eaqi_train_idx]["aqi_composite_eaqi"].values
    y_eaqi_test = df_eaqi.iloc[eaqi_test_idx]["aqi_composite_eaqi"].values

    # Scale features for EAQI prediction
    eaqi_scaler = StandardScaler()
    X_eaqi_train_scaled = eaqi_scaler.fit_transform(X_eaqi_train)
    X_eaqi_test_scaled = eaqi_scaler.transform(X_eaqi_test)

    eaqi_results = []

    for model_name, model in models.items():
        if model_name == "simple_average":
            continue  # Skip for direct EAQI prediction

        try:
            # Fresh model instance for EAQI prediction
            if model_name == "ridge_ensemble":
                eaqi_model = Ridge(alpha=1.0, random_state=42)
            elif model_name == "elastic_net":
                eaqi_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
            elif model_name == "random_forest":
                eaqi_model = RandomForestRegressor(
                    n_estimators=20, random_state=42, max_depth=4, n_jobs=-1
                )
            elif model_name == "gradient_boosting":
                eaqi_model = GradientBoostingRegressor(
                    n_estimators=20,
                    random_state=42,
                    max_depth=3,
                    learning_rate=0.1,
                )

            eaqi_model.fit(X_eaqi_train_scaled, y_eaqi_train)
            eaqi_pred = eaqi_model.predict(X_eaqi_test_scaled)

            # Round predictions to nearest EAQI level (1-6)
            eaqi_pred_rounded = np.clip(np.round(eaqi_pred), 1, 6)

            # Calculate EAQI metrics
            eaqi_mae = mean_absolute_error(y_eaqi_test, eaqi_pred_rounded)

            # Health warning metrics
            true_sensitive = [
                is_health_warning_required(val, True, "EAQI") for val in y_eaqi_test
            ]
            pred_sensitive = [
                is_health_warning_required(val, True, "EAQI")
                for val in eaqi_pred_rounded
            ]

            sensitive_accuracy = accuracy_score(true_sensitive, pred_sensitive)
            sensitive_precision = precision_score(
                true_sensitive, pred_sensitive, zero_division=0
            )
            sensitive_recall = recall_score(
                true_sensitive, pred_sensitive, zero_division=0
            )

            # False negative rate (critical)
            if sum(true_sensitive) > 0:
                false_negatives = sum(
                    [
                        (actual and not pred)
                        for actual, pred in zip(true_sensitive, pred_sensitive)
                    ]
                )
                false_negative_rate = false_negatives / sum(true_sensitive)
            else:
                false_negative_rate = 0.0

            eaqi_results.append(
                {
                    "model": model_name,
                    "eaqi_mae": eaqi_mae,
                    "sensitive_accuracy": sensitive_accuracy,
                    "sensitive_precision": sensitive_precision,
                    "sensitive_recall": sensitive_recall,
                    "false_negative_rate": false_negative_rate,
                    "n_test_samples": len(y_eaqi_test),
                }
            )

        except Exception as e:
            log.warning(f"Error with {model_name} for EAQI prediction: {e}")
            continue

    return pd.DataFrame(results), pd.DataFrame(eaqi_results), df_eaqi


def analyze_eaqi_comparison_results(
    results_df: pd.DataFrame, eaqi_results_df: pd.DataFrame, df_eaqi: pd.DataFrame
):
    """Analyze EAQI comparison results."""

    print("\n" + "=" * 80)
    print("QUICK EAQI ENSEMBLE MODEL COMPARISON")
    print("European Air Quality Index Standard")
    print("=" * 80)

    # Show EAQI dataset summary first
    print(f"\nEAQI DATASET SUMMARY:")
    if "aqi_composite_eaqi" in df_eaqi.columns:
        eaqi_stats = df_eaqi["aqi_composite_eaqi"].describe()
        print(f"Mean EAQI: {eaqi_stats['mean']:.2f}")
        print(f"Range: {eaqi_stats['min']:.0f} - {eaqi_stats['max']:.0f}")

        if "aqi_level_eaqi" in df_eaqi.columns:
            category_counts = df_eaqi["aqi_level_eaqi"].value_counts()
            print(f"Category Distribution:")
            for category, count in category_counts.items():
                pct = (count / len(df_eaqi)) * 100
                print(f"  {category}: {count:,} ({pct:.1f}%)")

    # Individual pollutant performance
    print(f"\nINDIVIDUAL POLLUTANT FORECASTING PERFORMANCE:")
    if len(results_df) > 0:
        overall_summary = (
            results_df.groupby("model")
            .agg(
                {
                    "ensemble_mae": "mean",
                    "improvement_vs_cams": "mean",
                    "improvement_vs_noaa": "mean",
                    "n_test_samples": "sum",
                }
            )
            .round(3)
            .sort_values("ensemble_mae")
        )

        for rank, (model, data) in enumerate(overall_summary.iterrows(), 1):
            print(f"\n{rank}. {model.upper().replace('_', ' ')}:")
            print(f"   MAE: {data['ensemble_mae']:.3f} ug/m3")
            print(f"   Improvement vs CAMS: {data['improvement_vs_cams']:+.1f}%")
            print(f"   Improvement vs NOAA: {data['improvement_vs_noaa']:+.1f}%")

    # EAQI direct prediction performance
    print(f"\nDIRECT EAQI PREDICTION PERFORMANCE:")
    if len(eaqi_results_df) > 0:
        eaqi_summary = eaqi_results_df.sort_values("eaqi_mae")

        for rank, (_, data) in enumerate(eaqi_summary.iterrows(), 1):
            print(f"\n{rank}. {data['model'].upper().replace('_', ' ')}:")
            print(f"   EAQI MAE: {data['eaqi_mae']:.3f}")
            print(f"   Health Warning Accuracy: {data['sensitive_accuracy']:.3f}")
            print(f"   Precision: {data['sensitive_precision']:.3f}")
            print(f"   Recall: {data['sensitive_recall']:.3f}")
            print(
                f"   False Negative Rate: {data['false_negative_rate']:.3f} (CRITICAL)"
            )

    # Performance by pollutant
    if len(results_df) > 0:
        print(f"\nPERFORMANCE BY POLLUTANT:")
        for pollutant in results_df["pollutant"].unique():
            print(f"\n{pollutant.upper()}:")
            pollutant_data = results_df[results_df["pollutant"] == pollutant]
            model_performance = pollutant_data.set_index("model")[
                "ensemble_mae"
            ].sort_values()

            for rank, (model, mae) in enumerate(model_performance.items(), 1):
                improvement = pollutant_data[pollutant_data["model"] == model][
                    "improvement_vs_cams"
                ].iloc[0]
                print(
                    f"  {rank}. {model.replace('_', ' ').title()}: {mae:.3f} ug/m3 ({improvement:+.1f}%)"
                )

    # Health warning focus
    print(f"\nHEALTH WARNING ANALYSIS (EUROPEAN STANDARDS):")
    print("EAQI Thresholds:")
    print("  - Sensitive Groups: Level >= 4 (Poor)")
    print("  - General Population: Level >= 5 (Very Poor)")

    if len(eaqi_results_df) > 0:
        best_health_model = eaqi_results_df.sort_values("false_negative_rate").iloc[0]
        print(f"\nBEST FOR HEALTH PROTECTION:")
        print(f"  Model: {best_health_model['model'].replace('_', ' ').title()}")
        print(
            f"  Misses only {best_health_model['false_negative_rate']*100:.1f}% of health warnings"
        )
        print(f"  Overall accuracy: {best_health_model['sensitive_accuracy']*100:.1f}%")

        best_accuracy_model = eaqi_results_df.sort_values("eaqi_mae").iloc[0]
        print(f"\nBEST EAQI PREDICTION:")
        print(f"  Model: {best_accuracy_model['model'].replace('_', ' ').title()}")
        print(f"  EAQI MAE: {best_accuracy_model['eaqi_mae']:.3f}")

    print("\n" + "=" * 80)
    print("EAQI COMPARISON COMPLETE")
    print("=" * 80)
    print("Key Findings:")
    print("- European EAQI uses 1-6 scale (more conservative than EPA)")
    print("- Focus on minimizing false negatives for health warnings")
    print("- EAQI Level 4+ (Poor) triggers sensitive group warnings")
    print("- Choose model based on health warning accuracy")
    print("=" * 80)


def main():
    """Main execution function."""

    log.info("Starting quick EAQI model comparison...")
    results_df, eaqi_results_df, df_eaqi = quick_eaqi_model_comparison()

    # Save results
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)

    if len(results_df) > 0:
        results_path = output_dir / "quick_eaqi_comparison_results.csv"
        results_df.to_csv(results_path, index=False)
        log.info(f"Pollutant results saved to {results_path}")

    if len(eaqi_results_df) > 0:
        eaqi_path = output_dir / "quick_eaqi_prediction_results.csv"
        eaqi_results_df.to_csv(eaqi_path, index=False)
        log.info(f"EAQI prediction results saved to {eaqi_path}")

    # Analyze results
    analyze_eaqi_comparison_results(results_df, eaqi_results_df, df_eaqi)

    return 0


if __name__ == "__main__":
    exit(main())
