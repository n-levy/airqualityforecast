#!/usr/bin/env python3
"""
Enhanced Evaluation Analysis with Realistic Benchmarks

Re-run comprehensive evaluation analysis using enhanced realistic CAMS
and NOAA benchmarks based on scientific literature performance data.
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_enhanced_results():
    """Load enhanced results with realistic benchmarks."""
    # Load the most recent enhanced results
    results_file = "../final_dataset/enhanced_realistic_results_20250911_124055.json"
    with open(results_file, "r") as f:
        return json.load(f)


def comprehensive_performance_analysis(results):
    """Perform comprehensive performance analysis."""

    print("ENHANCED EVALUATION ANALYSIS")
    print("=" * 50)
    print(f"Total cities analyzed: {len(results)}")

    # Collect all performance data
    all_data = []
    methods = ["ridge", "cams", "noaa"]
    pollutants = ["PM25", "PM10", "NO2", "O3", "SO2", "CO", "AQI"]

    for city_name, city_data in results.items():
        continent = city_data["continent"]

        for pollutant in pollutants:
            if pollutant in city_data["results"]:
                pollutant_data = city_data["results"][pollutant]

                row = {
                    "City": city_name,
                    "Continent": continent,
                    "Pollutant": pollutant,
                }

                for method in methods:
                    if method in pollutant_data:
                        row[f"{method}_MAE"] = pollutant_data[method].get("MAE", np.nan)
                        row[f"{method}_RMSE"] = pollutant_data[method].get(
                            "RMSE", np.nan
                        )
                        row[f"{method}_R2"] = pollutant_data[method].get("R2", np.nan)
                        row[f"{method}_MPE"] = pollutant_data[method].get("MPE", np.nan)

                all_data.append(row)

    df = pd.DataFrame(all_data)

    # Calculate improvements for each method comparison
    df["ridge_vs_cams_mae_improvement"] = (
        (df["cams_MAE"] - df["ridge_MAE"]) / df["cams_MAE"] * 100
    )
    df["ridge_vs_noaa_mae_improvement"] = (
        (df["noaa_MAE"] - df["ridge_MAE"]) / df["noaa_MAE"] * 100
    )
    df["ridge_vs_best_benchmark_improvement"] = np.maximum(
        df["ridge_vs_cams_mae_improvement"], df["ridge_vs_noaa_mae_improvement"]
    )

    return df


def analyze_continental_performance(df):
    """Analyze performance by continent."""

    print("\nCONTINENTAL PERFORMANCE ANALYSIS")
    print("=" * 40)

    continental_summary = []

    for continent in df["Continent"].unique():
        continent_data = df[df["Continent"] == continent]

        summary = {
            "Continent": continent,
            "Cities": continent_data["City"].nunique(),
            "Ridge_MAE_Avg": continent_data["ridge_MAE"].mean(),
            "CAMS_MAE_Avg": continent_data["cams_MAE"].mean(),
            "NOAA_MAE_Avg": continent_data["noaa_MAE"].mean(),
            "Ridge_vs_CAMS_Improvement": continent_data[
                "ridge_vs_cams_mae_improvement"
            ].mean(),
            "Ridge_vs_NOAA_Improvement": continent_data[
                "ridge_vs_noaa_mae_improvement"
            ].mean(),
            "Ridge_vs_Best_Improvement": continent_data[
                "ridge_vs_best_benchmark_improvement"
            ].mean(),
            "Ridge_R2_Avg": continent_data["ridge_R2"].mean(),
            "CAMS_R2_Avg": continent_data["cams_R2"].mean(),
            "NOAA_R2_Avg": continent_data["noaa_R2"].mean(),
        }

        continental_summary.append(summary)

    continental_df = pd.DataFrame(continental_summary)

    print(
        f"{'Continent':<15} {'Cities':<8} {'Ridge MAE':<10} {'CAMS MAE':<10} {'NOAA MAE':<10} {'Best Improve%':<12}"
    )
    print("-" * 75)

    for _, row in continental_df.iterrows():
        print(
            f"{row['Continent']:<15} {row['Cities']:<8} {row['Ridge_MAE_Avg']:<10.3f} {row['CAMS_MAE_Avg']:<10.3f} {row['NOAA_MAE_Avg']:<10.3f} {row['Ridge_vs_Best_Improvement']:<12.1f}"
        )

    return continental_df


def analyze_pollutant_performance(df):
    """Analyze performance by pollutant."""

    print("\nPOLLUTANT PERFORMANCE ANALYSIS")
    print("=" * 35)

    pollutant_summary = []

    for pollutant in df["Pollutant"].unique():
        pollutant_data = df[df["Pollutant"] == pollutant]

        summary = {
            "Pollutant": pollutant,
            "Cities": len(pollutant_data),
            "Ridge_MAE_Avg": pollutant_data["ridge_MAE"].mean(),
            "CAMS_MAE_Avg": pollutant_data["cams_MAE"].mean(),
            "NOAA_MAE_Avg": pollutant_data["noaa_MAE"].mean(),
            "Ridge_vs_CAMS_Improvement": pollutant_data[
                "ridge_vs_cams_mae_improvement"
            ].mean(),
            "Ridge_vs_NOAA_Improvement": pollutant_data[
                "ridge_vs_noaa_mae_improvement"
            ].mean(),
            "Ridge_vs_Best_Improvement": pollutant_data[
                "ridge_vs_best_benchmark_improvement"
            ].mean(),
            "Ridge_R2_Avg": pollutant_data["ridge_R2"].mean(),
            "CAMS_R2_Avg": pollutant_data["cams_R2"].mean(),
            "NOAA_R2_Avg": pollutant_data["noaa_R2"].mean(),
        }

        pollutant_summary.append(summary)

    pollutant_df = pd.DataFrame(pollutant_summary)

    print(
        f"{'Pollutant':<10} {'Cities':<8} {'Ridge MAE':<10} {'CAMS MAE':<10} {'NOAA MAE':<10} {'Best Improve%':<12}"
    )
    print("-" * 70)

    for _, row in pollutant_df.iterrows():
        print(
            f"{row['Pollutant']:<10} {row['Cities']:<8} {row['Ridge_MAE_Avg']:<10.3f} {row['CAMS_MAE_Avg']:<10.3f} {row['NOAA_MAE_Avg']:<10.3f} {row['Ridge_vs_Best_Improvement']:<12.1f}"
        )

    return pollutant_df


def analyze_health_warnings_enhanced(results):
    """Analyze health warning performance with enhanced benchmarks."""

    print("\nENHANCED HEALTH WARNING ANALYSIS")
    print("=" * 40)

    methods = ["ridge", "cams", "noaa"]
    global_totals = {}

    for method in methods:
        global_totals[method] = {
            "sensitive": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "general": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        }

    # Aggregate health warning data
    for city_name, city_data in results.items():
        if "health_warnings" in city_data["results"]:
            health_data = city_data["results"]["health_warnings"]

            for method in methods:
                if method in health_data:
                    for alert_type in ["sensitive", "general"]:
                        if alert_type in health_data[method]:
                            alert = health_data[method][alert_type]

                            global_totals[method][alert_type]["tp"] += alert.get(
                                "true_positives", 0
                            )
                            global_totals[method][alert_type]["fp"] += alert.get(
                                "false_positives", 0
                            )
                            global_totals[method][alert_type]["tn"] += alert.get(
                                "true_negatives", 0
                            )
                            global_totals[method][alert_type]["fn"] += alert.get(
                                "false_negatives", 0
                            )

    # Calculate and display metrics
    print(
        f"{'Method':<8} {'Alert Type':<12} {'Precision':<10} {'Recall':<8} {'FNR':<8} {'Rating':<12}"
    )
    print("-" * 68)

    method_ratings = {}

    for method in methods:
        method_ratings[method] = {}
        for alert_type in ["sensitive", "general"]:
            stats = global_totals[method][alert_type]

            precision = (
                stats["tp"] / (stats["tp"] + stats["fp"])
                if (stats["tp"] + stats["fp"]) > 0
                else 0
            )
            recall = (
                stats["tp"] / (stats["tp"] + stats["fn"])
                if (stats["tp"] + stats["fn"]) > 0
                else 0
            )
            fnr = (
                stats["fn"] / (stats["tp"] + stats["fn"])
                if (stats["tp"] + stats["fn"]) > 0
                else 0
            )

            if fnr < 0.05:
                rating = "EXCELLENT"
            elif fnr < 0.10:
                rating = "GOOD"
            elif fnr < 0.15:
                rating = "ACCEPTABLE"
            else:
                rating = "POOR"

            method_ratings[method][alert_type] = {
                "precision": precision,
                "recall": recall,
                "fnr": fnr,
                "rating": rating,
            }

            print(
                f"{method.upper():<8} {alert_type:<12} {precision:<10.3f} {recall:<8.3f} {fnr:<8.3f} {rating:<12}"
            )

    return method_ratings


def generate_summary_report(df, continental_df, pollutant_df, health_ratings):
    """Generate comprehensive summary report."""

    print("\nEVALUATION SUMMARY REPORT")
    print("=" * 30)

    # Overall performance
    overall_ridge_improvement = df["ridge_vs_best_benchmark_improvement"].mean()
    print(
        f"Overall Ridge Improvement vs Best Benchmark: {overall_ridge_improvement:.1f}%"
    )

    # Best and worst performing continents for Ridge
    best_continent = continental_df.loc[
        continental_df["Ridge_vs_Best_Improvement"].idxmax()
    ]
    worst_continent = continental_df.loc[
        continental_df["Ridge_vs_Best_Improvement"].idxmin()
    ]

    print(
        f"Best Ridge Performance: {best_continent['Continent']} ({best_continent['Ridge_vs_Best_Improvement']:.1f}% improvement)"
    )
    print(
        f"Worst Ridge Performance: {worst_continent['Continent']} ({worst_continent['Ridge_vs_Best_Improvement']:.1f}% improvement)"
    )

    # Best and worst performing pollutants for Ridge
    best_pollutant = pollutant_df.loc[
        pollutant_df["Ridge_vs_Best_Improvement"].idxmax()
    ]
    worst_pollutant = pollutant_df.loc[
        pollutant_df["Ridge_vs_Best_Improvement"].idxmin()
    ]

    print(
        f"Best Pollutant Performance: {best_pollutant['Pollutant']} ({best_pollutant['Ridge_vs_Best_Improvement']:.1f}% improvement)"
    )
    print(
        f"Worst Pollutant Performance: {worst_pollutant['Pollutant']} ({worst_pollutant['Ridge_vs_Best_Improvement']:.1f}% improvement)"
    )

    # Health warning performance
    ridge_avg_fnr = (
        health_ratings["ridge"]["sensitive"]["fnr"]
        + health_ratings["ridge"]["general"]["fnr"]
    ) / 2
    cams_avg_fnr = (
        health_ratings["cams"]["sensitive"]["fnr"]
        + health_ratings["cams"]["general"]["fnr"]
    ) / 2
    noaa_avg_fnr = (
        health_ratings["noaa"]["sensitive"]["fnr"]
        + health_ratings["noaa"]["general"]["fnr"]
    ) / 2

    print(f"Health Warning False Negative Rates:")
    print(
        f"  Ridge: {ridge_avg_fnr:.1%} | CAMS: {cams_avg_fnr:.1%} | NOAA: {noaa_avg_fnr:.1%}"
    )

    # Determine best method
    if ridge_avg_fnr <= min(cams_avg_fnr, noaa_avg_fnr):
        best_health_method = "Ridge"
    elif cams_avg_fnr <= noaa_avg_fnr:
        best_health_method = "CAMS"
    else:
        best_health_method = "NOAA"

    print(f"Best Health Warning Performance: {best_health_method}")

    return {
        "overall_improvement": overall_ridge_improvement,
        "best_continent": best_continent["Continent"],
        "worst_continent": worst_continent["Continent"],
        "best_pollutant": best_pollutant["Pollutant"],
        "worst_pollutant": worst_pollutant["Pollutant"],
        "best_health_method": best_health_method,
    }


def save_enhanced_analysis_results(
    df, continental_df, pollutant_df, health_ratings, summary
):
    """Save enhanced analysis results to files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    output_file = f"../final_dataset/enhanced_evaluation_analysis_{timestamp}.json"

    results_data = {
        "timestamp": timestamp,
        "analysis_type": "Enhanced Realistic Benchmark Evaluation",
        "summary": summary,
        "continental_performance": continental_df.to_dict("records"),
        "pollutant_performance": pollutant_df.to_dict("records"),
        "health_warning_performance": health_ratings,
        "detailed_data": df.to_dict("records"),
    }

    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nEnhanced analysis results saved to: {output_file}")

    # Save CSV summaries
    continental_df.to_csv(
        f"../final_dataset/continental_performance_{timestamp}.csv", index=False
    )
    pollutant_df.to_csv(
        f"../final_dataset/pollutant_performance_{timestamp}.csv", index=False
    )

    print(f"CSV summaries saved with timestamp: {timestamp}")

    return output_file


def main():
    """Main execution function."""

    print("ENHANCED EVALUATION ANALYSIS WITH REALISTIC BENCHMARKS")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65)

    # Load enhanced results
    results = load_enhanced_results()
    print(f"Loaded enhanced results for {len(results)} cities")

    # Perform comprehensive analysis
    df = comprehensive_performance_analysis(results)

    # Analyze by continent
    continental_df = analyze_continental_performance(df)

    # Analyze by pollutant
    pollutant_df = analyze_pollutant_performance(df)

    # Analyze health warnings
    health_ratings = analyze_health_warnings_enhanced(results)

    # Generate summary
    summary = generate_summary_report(df, continental_df, pollutant_df, health_ratings)

    # Save results
    output_file = save_enhanced_analysis_results(
        df, continental_df, pollutant_df, health_ratings, summary
    )

    print(f"\nFINAL SUMMARY:")
    print(f"✓ Enhanced evaluation analysis completed for {len(results)} cities")
    print(
        f"✓ Ridge regression shows {summary['overall_improvement']:.1f}% average improvement"
    )
    print(
        f"✓ Best performance in {summary['best_continent']} for {summary['best_pollutant']}"
    )
    print(f"✓ {summary['best_health_method']} provides best health warning performance")
    print(f"✓ Results saved to: {output_file}")

    return results_data, output_file


if __name__ == "__main__":
    results, file_path = main()
