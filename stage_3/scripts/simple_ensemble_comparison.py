#!/usr/bin/env python3
"""
Simple Ensemble Comparison

Quick comparison using minimal data and simple validation approach.
"""

import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def simple_ensemble_validation():
    """Quick ensemble comparison using simple train/test split."""

    # Load and sample data heavily for speed
    data_path = Path("data/analysis/5year_hourly_comprehensive_dataset.csv")
    log.info("Loading and heavily sampling dataset for speed...")

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = pd.to_datetime(df["date"])

    # Heavy sampling - every 48 hours for very fast processing
    df_sampled = df.iloc[::48].copy().reset_index(drop=True)
    log.info(f"Heavily sampled dataset: {len(df_sampled)} records")

    # Get features
    target_cols = [col for col in df_sampled.columns if col.startswith("actual_")]
    benchmark_cols = [col for col in df_sampled.columns if col.startswith("forecast_")]

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
            elif col == "week_position":
                df_sampled[col] = (df_sampled[col] == "weekend").astype(int)
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

    for pollutant in pollutants:
        log.info(f"Processing {pollutant}...")

        target_col = f"actual_{pollutant}"
        cams_col = f"forecast_cams_{pollutant}"
        noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

        # Prepare data
        X = df_sampled[feature_cols].fillna(0).values
        y = df_sampled[target_col].values
        cams_pred_all = df_sampled[cams_col].values
        noaa_pred_all = df_sampled[noaa_col].values

        # Simple train/test split (70/30)
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        cams_test = cams_pred_all[test_idx]
        noaa_test = noaa_pred_all[test_idx]

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

                # Calculate metrics
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                ensemble_r2 = r2_score(y_test, ensemble_pred)
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
                        "ensemble_r2": ensemble_r2,
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

    return pd.DataFrame(results)


def analyze_simple_results(results_df: pd.DataFrame):
    """Analyze simple validation results."""

    print("\n" + "=" * 80)
    print("SIMPLE ENSEMBLE COMPARISON RESULTS")
    print("Quick Validation with Train/Test Split")
    print("=" * 80)

    # Overall performance by model
    print("\nMODEL RANKING (by average MAE):")
    overall_summary = (
        results_df.groupby("model")
        .agg(
            {
                "ensemble_mae": "mean",
                "ensemble_r2": "mean",
                "improvement_vs_cams": "mean",
                "improvement_vs_noaa": "mean",
                "n_test_samples": "sum",
            }
        )
        .round(3)
    )

    # Sort by MAE (best first)
    overall_summary = overall_summary.sort_values("ensemble_mae")

    for rank, (model, data) in enumerate(overall_summary.iterrows(), 1):
        print(f"\n{rank}. {model.upper().replace('_', ' ')}:")
        print(f"   MAE: {data['ensemble_mae']:.3f} ug/m3")
        print(f"   R²: {data['ensemble_r2']:.3f}")
        print(f"   Improvement vs CAMS: {data['improvement_vs_cams']:+.1f}%")
        print(f"   Improvement vs NOAA: {data['improvement_vs_noaa']:+.1f}%")

    # Benchmark performance
    print("\nBENCHMARK COMPARISON:")
    cams_avg_mae = results_df["cams_mae"].mean()
    noaa_avg_mae = results_df["noaa_mae"].mean()
    print(f"CAMS Average MAE: {cams_avg_mae:.3f} ug/m3")
    print(f"NOAA Average MAE: {noaa_avg_mae:.3f} ug/m3")

    # Performance by pollutant
    print("\nPERFORMANCE BY POLLUTANT:")
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

    # Summary table
    print("\nSUMMARY TABLE (MAE in ug/m3):")
    pivot_table = results_df.pivot_table(
        index="pollutant", columns="model", values="ensemble_mae", aggfunc="mean"
    ).round(3)

    # Reorder by performance
    model_order = overall_summary.index.tolist()
    pivot_table = pivot_table[model_order]
    print(pivot_table.to_string())

    # Best model
    best_model = overall_summary.index[0]
    best_mae = overall_summary.loc[best_model, "ensemble_mae"]
    best_improvement = overall_summary.loc[best_model, "improvement_vs_cams"]

    print(f"\n🏆 BEST PERFORMING MODEL:")
    print(f"   {best_model.replace('_', ' ').title()}")
    print(f"   Average MAE: {best_mae:.3f} ug/m3")
    print(f"   Average improvement: {best_improvement:+.1f}% vs CAMS")

    print("\n" + "=" * 80)
    print("NOTE: This is a simplified comparison. Full walk-forward validation")
    print("running in background will provide more comprehensive results.")
    print("=" * 80)


def main():
    """Main execution function."""

    log.info("Starting simple ensemble comparison...")
    results_df = simple_ensemble_validation()

    if len(results_df) == 0:
        log.error("No validation results generated")
        return 1

    # Save results
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)
    results_path = output_dir / "simple_ensemble_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Results saved to {results_path}")

    # Analyze results
    analyze_simple_results(results_df)

    return 0


if __name__ == "__main__":
    exit(main())
