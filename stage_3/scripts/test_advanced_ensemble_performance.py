#!/usr/bin/env python3
"""
Test Advanced Ensemble Performance with Comprehensive Features

This script tests the performance improvement achieved by using the comprehensive
feature set with advanced machine learning ensemble methods.

Compares:
1. Original ensemble methods (using only CAMS/NOAA forecasts)
2. Advanced ensemble methods using all 160 features
3. Feature importance analysis
4. Performance improvement quantification
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def prepare_features_and_targets(df: pd.DataFrame) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Prepare feature sets and target variables for modeling.
    Handle categorical variables by encoding them.
    """
    df_processed = df.copy()

    # Identify feature groups
    basic_features = [
        "forecast_cams_pm25",
        "forecast_cams_pm10",
        "forecast_cams_no2",
        "forecast_cams_o3",
        "forecast_noaa_gefs_aerosol_pm25",
        "forecast_noaa_gefs_aerosol_pm10",
        "forecast_noaa_gefs_aerosol_no2",
        "forecast_noaa_gefs_aerosol_o3",
    ]

    # All features except identity, actuals, and existing ensemble forecasts
    exclude_patterns = [
        "city",
        "date",
        "forecast_made_date",
        "forecast_lead_hours",
        "actual_",
        "forecast_simple_avg_",
        "forecast_weighted_avg_",
        "forecast_ridge_",
        "forecast_xgboost_",
        "forecast_bias_corrected_",
    ]

    all_features = []
    for col in df.columns:
        if not any(pattern in col for pattern in exclude_patterns):
            all_features.append(col)

    # Handle categorical variables
    categorical_columns = []
    for col in all_features:
        if col in df_processed.columns and df_processed[col].dtype == "object":
            categorical_columns.append(col)

    log.info(
        f"Found {len(categorical_columns)} categorical columns: {categorical_columns}"
    )

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            # Handle NaN values by filling with 'missing' first
            df_processed[col] = df_processed[col].fillna("missing")
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le

    # Remove categorical columns that couldn't be processed
    all_features = [col for col in all_features if col in df_processed.columns]

    # Feature sets
    feature_sets = {"basic": basic_features, "comprehensive": all_features}

    # Target variables
    targets = {
        "pm25": "actual_pm25",
        "pm10": "actual_pm10",
        "no2": "actual_no2",
        "o3": "actual_o3",
    }

    log.info(f"Basic features: {len(basic_features)}")
    log.info(f"Comprehensive features: {len(all_features)}")

    return feature_sets, targets, df_processed


def train_advanced_ensemble_models(
    df: pd.DataFrame, feature_sets: Dict, targets: Dict
) -> Dict:
    """
    Train advanced ensemble models using different feature sets.
    """
    results = {}
    pollutants = list(targets.keys())

    # Model configurations
    models = {
        "ridge_advanced": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }

    # Sort data by city and date for time series CV
    df_sorted = df.sort_values(["city", "date"]).copy()

    log.info("Training advanced ensemble models...")

    for feature_set_name, feature_list in feature_sets.items():
        log.info(
            f"Training models with {feature_set_name} features ({len(feature_list)} features)"
        )

        for pollutant in pollutants:
            target_col = targets[pollutant]

            # Prepare data
            valid_mask = df_sorted[target_col].notna()
            for feature in feature_list:
                if feature in df_sorted.columns:
                    valid_mask &= df_sorted[feature].notna()

            if not valid_mask.any():
                log.warning(
                    f"No valid data for {pollutant} with {feature_set_name} features"
                )
                continue

            # Extract features and targets
            available_features = [f for f in feature_list if f in df_sorted.columns]
            X = df_sorted.loc[valid_mask, available_features].values
            y = df_sorted.loc[valid_mask, target_col].values

            if len(available_features) == 0 or len(y) < 10:
                log.warning(
                    f"Insufficient data for {pollutant} with {feature_set_name} features"
                )
                continue

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)

            for model_name, model in models.items():
                try:
                    # Cross-validation scores
                    cv_scores = cross_val_score(
                        model, X_scaled, y, cv=tscv, scoring="neg_mean_absolute_error"
                    )
                    mean_cv_mae = -cv_scores.mean()

                    # Train final model on all data
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)

                    # Calculate metrics
                    mae = mean_absolute_error(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    r2 = r2_score(y, y_pred)
                    correlation = np.corrcoef(y, y_pred)[0, 1]

                    # Store results
                    key = f"{model_name}_{feature_set_name}_{pollutant}"
                    results[key] = {
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "pollutant": pollutant,
                        "n_features": len(available_features),
                        "n_samples": len(y),
                        "cv_mae": mean_cv_mae,
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2,
                        "correlation": correlation,
                        "model": model,
                        "scaler": scaler,
                        "features": available_features,
                    }

                    log.info(
                        f"{model_name} {feature_set_name} {pollutant}: MAE={mae:.3f}, R²={r2:.3f}"
                    )

                except Exception as e:
                    log.warning(
                        f"Failed to train {model_name} for {pollutant} with {feature_set_name}: {e}"
                    )

    return results


def analyze_feature_importance(results: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze feature importance for models that support it.
    """
    importance_data = []

    log.info("Analyzing feature importance...")

    for key, result in results.items():
        model = result["model"]
        features = result["features"]

        # Extract importance for tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            for feature, importance in zip(features, importances):
                importance_data.append(
                    {
                        "model": result["model_name"],
                        "feature_set": result["feature_set"],
                        "pollutant": result["pollutant"],
                        "feature": feature,
                        "importance": importance,
                    }
                )

    if importance_data:
        return pd.DataFrame(importance_data)
    else:
        return pd.DataFrame()


def compare_performance_improvements(results: Dict) -> pd.DataFrame:
    """
    Compare performance between basic and comprehensive feature sets.
    """
    comparison_data = []

    # Group results by model and pollutant
    model_pollutant_groups = {}
    for key, result in results.items():
        group_key = f"{result['model_name']}_{result['pollutant']}"
        if group_key not in model_pollutant_groups:
            model_pollutant_groups[group_key] = {}
        model_pollutant_groups[group_key][result["feature_set"]] = result

    # Compare basic vs comprehensive
    for group_key, group_results in model_pollutant_groups.items():
        if "basic" in group_results and "comprehensive" in group_results:
            basic = group_results["basic"]
            comprehensive = group_results["comprehensive"]

            mae_improvement = (basic["mae"] - comprehensive["mae"]) / basic["mae"] * 100
            r2_improvement = (
                (comprehensive["r2"] - basic["r2"]) / abs(basic["r2"]) * 100
                if basic["r2"] != 0
                else 0
            )

            comparison_data.append(
                {
                    "model": basic["model_name"],
                    "pollutant": basic["pollutant"],
                    "basic_mae": basic["mae"],
                    "comprehensive_mae": comprehensive["mae"],
                    "mae_improvement_pct": mae_improvement,
                    "basic_r2": basic["r2"],
                    "comprehensive_r2": comprehensive["r2"],
                    "r2_improvement_pct": r2_improvement,
                    "basic_features": basic["n_features"],
                    "comprehensive_features": comprehensive["n_features"],
                }
            )

    return pd.DataFrame(comparison_data)


def print_performance_analysis(
    comparison_df: pd.DataFrame, feature_importance_df: pd.DataFrame
):
    """
    Print comprehensive performance analysis.
    """
    print("\n" + "=" * 120)
    print("ADVANCED ENSEMBLE PERFORMANCE ANALYSIS")
    print("=" * 120)

    if comparison_df.empty:
        print("No comparison data available")
        return

    print("\nPERFORMANCE IMPROVEMENT SUMMARY:")
    print("-" * 60)

    # Overall improvements
    avg_mae_improvement = comparison_df["mae_improvement_pct"].mean()
    avg_r2_improvement = comparison_df["r2_improvement_pct"].mean()

    print(f"Average MAE improvement: {avg_mae_improvement:+.1f}%")
    print(f"Average R² improvement: {avg_r2_improvement:+.1f}%")
    print()

    # Best performing models
    print("BEST MODELS BY POLLUTANT (Comprehensive Features):")
    print("-" * 60)

    for pollutant in comparison_df["pollutant"].unique():
        pollutant_data = comparison_df[comparison_df["pollutant"] == pollutant]
        best_model = pollutant_data.loc[pollutant_data["comprehensive_mae"].idxmin()]

        print(f"{pollutant.upper()}:")
        print(f"  Best Model: {best_model['model']}")
        print(
            f"  MAE: {best_model['comprehensive_mae']:.3f} (vs {best_model['basic_mae']:.3f} basic)"
        )
        print(f"  Improvement: {best_model['mae_improvement_pct']:+.1f}%")
        print(f"  R²: {best_model['comprehensive_r2']:.3f}")
        print()

    # Model ranking
    print("MODEL RANKING BY AVERAGE PERFORMANCE:")
    print("-" * 60)

    model_avg = (
        comparison_df.groupby("model")
        .agg(
            {
                "comprehensive_mae": "mean",
                "mae_improvement_pct": "mean",
                "comprehensive_r2": "mean",
            }
        )
        .round(3)
    )

    model_avg_sorted = model_avg.sort_values("comprehensive_mae")
    print(model_avg_sorted.to_string())
    print()

    # Feature importance analysis
    if not feature_importance_df.empty:
        print("TOP FEATURE IMPORTANCE (Comprehensive Models):")
        print("-" * 60)

        # Average importance across all models
        avg_importance = (
            feature_importance_df.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )

        print("Top 15 Most Important Features:")
        for i, (feature, importance) in enumerate(avg_importance.items(), 1):
            print(f"  {i:2d}. {feature}: {importance:.4f}")
        print()

        # Feature category analysis
        print("FEATURE CATEGORY IMPORTANCE:")
        print("-" * 40)

        feature_categories = {
            "Meteorological": [
                "wind",
                "temp",
                "humid",
                "precip",
                "pressure",
                "solar",
                "boundary",
            ],
            "Temporal": [
                "rush",
                "holiday",
                "weekend",
                "week_",
                "august_",
                "sin",
                "cos",
            ],
            "Cross-Pollutant": [
                "ratio",
                "load",
                "secondary",
                "fresh",
                "aged",
                "photochem",
            ],
            "Spatial": [
                "spatial",
                "gradient",
                "upwind",
                "urban",
                "coastal",
                "industrial",
            ],
            "Uncertainty": ["spread", "confidence", "variance", "consensus", "extreme"],
            "External": [
                "traffic",
                "construction",
                "fire",
                "economic",
                "anthropogenic",
            ],
            "Interactions": ["interaction", "stagnation"],
        }

        category_importance = {}
        for category, keywords in feature_categories.items():
            category_features = feature_importance_df[
                feature_importance_df["feature"].str.contains(
                    "|".join(keywords), case=False
                )
            ]
            if not category_features.empty:
                category_importance[category] = category_features["importance"].mean()

        for category, importance in sorted(
            category_importance.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {category}: {importance:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test advanced ensemble performance with comprehensive features"
    )
    parser.add_argument(
        "--input", required=True, help="Input comprehensive enhanced dataset path"
    )
    parser.add_argument(
        "--output-dir", default="data/analysis", help="Output directory for results"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        log.error(f"Input file does not exist: {input_path}")
        return 1

    # Load comprehensive enhanced dataset
    log.info(f"Loading comprehensive enhanced dataset from {input_path}")
    df = pd.read_csv(input_path)
    log.info(f"Dataset shape: {df.shape}")

    # Prepare features and targets
    feature_sets, targets, df_processed = prepare_features_and_targets(df)

    # Train advanced ensemble models
    results = train_advanced_ensemble_models(df_processed, feature_sets, targets)

    if not results:
        log.error("No models were successfully trained")
        return 1

    # Analyze feature importance
    feature_importance_df = analyze_feature_importance(results, df_processed)

    # Compare performance improvements
    comparison_df = compare_performance_improvements(results)

    # Save results
    if not comparison_df.empty:
        comparison_path = output_dir / "advanced_ensemble_performance_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        log.info(f"Saved performance comparison to {comparison_path}")

    if not feature_importance_df.empty:
        importance_path = output_dir / "feature_importance_analysis.csv"
        feature_importance_df.to_csv(importance_path, index=False)
        log.info(f"Saved feature importance analysis to {importance_path}")

    # Print analysis
    print_performance_analysis(comparison_df, feature_importance_df)

    return 0


if __name__ == "__main__":
    exit(main())
