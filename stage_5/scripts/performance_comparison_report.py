#!/usr/bin/env python3
"""
Performance Comparison Report Generator

Generate a comprehensive text report comparing Ridge regression vs benchmarks.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd


def load_results(
    results_file="../final_dataset/full_100_city_results_20250911_121246.json",
):
    """Load evaluation results."""
    with open(results_file, "r") as f:
        return json.load(f)


def analyze_pollutant_comparison(results, pollutant):
    """Compare Ridge vs CAMS vs NOAA for a specific pollutant."""

    report_lines = []
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"POLLUTANT ANALYSIS: {pollutant}")
    report_lines.append(f"{'='*80}")

    city_data = []

    for city_name, city_info in results.items():
        continent = city_info["continent"]
        country = city_info["country"]

        if pollutant in city_info["results"]:
            data = city_info["results"][pollutant]

            ridge_mae = data["ridge"]["MAE"]
            cams_mae = data["cams"]["MAE"]
            noaa_mae = data["noaa"]["MAE"]

            ridge_r2 = data["ridge"]["R2"]
            cams_r2 = data["cams"]["R2"]
            noaa_r2 = data["noaa"]["R2"]

            # Calculate improvements
            best_benchmark_mae = min(cams_mae, noaa_mae)
            ridge_improvement = (
                ((best_benchmark_mae - ridge_mae) / best_benchmark_mae) * 100
                if best_benchmark_mae > 0
                else 0
            )

            city_data.append(
                {
                    "City": city_name,
                    "Country": country,
                    "Continent": continent,
                    "Ridge_MAE": ridge_mae,
                    "CAMS_MAE": cams_mae,
                    "NOAA_MAE": noaa_mae,
                    "Best_Benchmark_MAE": best_benchmark_mae,
                    "Ridge_R2": ridge_r2,
                    "CAMS_R2": cams_r2,
                    "NOAA_R2": noaa_r2,
                    "Ridge_Improvement_%": ridge_improvement,
                }
            )

    df = pd.DataFrame(city_data)

    # Overall statistics
    report_lines.append(f"\nOVERALL PERFORMANCE STATISTICS:")
    report_lines.append(f"Cities Analyzed: {len(df)}")
    report_lines.append(
        f"Average MAE - Ridge: {df['Ridge_MAE'].mean():.3f}, CAMS: {df['CAMS_MAE'].mean():.3f}, NOAA: {df['NOAA_MAE'].mean():.3f}"
    )
    report_lines.append(
        f"Average R2 - Ridge: {df['Ridge_R2'].mean():.3f}, CAMS: {df['CAMS_R2'].mean():.3f}, NOAA: {df['NOAA_R2'].mean():.3f}"
    )
    report_lines.append(
        f"Ridge Improvement over Best Benchmark: {df['Ridge_Improvement_%'].mean():.1f}% average"
    )

    # Performance by continent
    report_lines.append(f"\nPERFORMANCE BY CONTINENT:")
    continental_stats = (
        df.groupby("Continent")
        .agg(
            {
                "Ridge_MAE": "mean",
                "CAMS_MAE": "mean",
                "NOAA_MAE": "mean",
                "Ridge_Improvement_%": "mean",
            }
        )
        .round(3)
    )

    for continent in continental_stats.index:
        stats = continental_stats.loc[continent]
        report_lines.append(
            f"{continent}: Ridge {stats['Ridge_MAE']:.3f}, CAMS {stats['CAMS_MAE']:.3f}, NOAA {stats['NOAA_MAE']:.3f}, Improvement {stats['Ridge_Improvement_%']:.1f}%"
        )

    # Top and bottom performers
    df_sorted = df.sort_values("Ridge_Improvement_%", ascending=False)

    report_lines.append(f"\nTOP 10 RIDGE IMPROVEMENTS:")
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
        report_lines.append(
            f"{i+1:2d}. {row['City']:<15} ({row['Continent']:<15}): {row['Ridge_Improvement_%']:6.1f}%"
        )

    # Cities where Ridge underperformed
    underperformed = df[df["Ridge_Improvement_%"] < 0]
    if len(underperformed) > 0:
        report_lines.append(
            f"\nCITIES WHERE RIDGE UNDERPERFORMED ({len(underperformed)} cities):"
        )
        for _, row in underperformed.iterrows():
            report_lines.append(
                f"  {row['City']:<15} ({row['Continent']:<15}): {row['Ridge_Improvement_%']:6.1f}%"
            )
    else:
        report_lines.append(f"\nRidge outperformed benchmarks in ALL {len(df)} cities!")

    return "\n".join(report_lines), df


def analyze_aqi_health_warnings(results):
    """Analyze AQI and health warning performance."""

    report_lines = []
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"HEALTH WARNING CONFUSION MATRIX ANALYSIS")
    report_lines.append(f"{'='*80}")

    # Aggregate confusion matrices
    methods = ["ridge", "cams", "noaa"]
    method_totals = {}
    for method in methods:
        method_totals[method] = {
            "sensitive": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "general": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        }

    continental_totals = {}

    for city_name, city_data in results.items():
        continent = city_data["continent"]

        if continent not in continental_totals:
            continental_totals[continent] = {}
            for method in methods:
                continental_totals[continent][method] = {
                    "sensitive": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                    "general": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                }

        if "health_warnings" in city_data["results"]:
            health_data = city_data["results"]["health_warnings"]

            for method in methods:
                if method in health_data:
                    for alert_type in ["sensitive", "general"]:
                        if alert_type in health_data[method]:
                            alert_data = health_data[method][alert_type]

                            # Global totals
                            method_totals[method][alert_type]["tp"] += alert_data.get(
                                "true_positives", 0
                            )
                            method_totals[method][alert_type]["fp"] += alert_data.get(
                                "false_positives", 0
                            )
                            method_totals[method][alert_type]["tn"] += alert_data.get(
                                "true_negatives", 0
                            )
                            method_totals[method][alert_type]["fn"] += alert_data.get(
                                "false_negatives", 0
                            )

                            # Continental totals
                            continental_totals[continent][method][alert_type][
                                "tp"
                            ] += alert_data.get("true_positives", 0)
                            continental_totals[continent][method][alert_type][
                                "fp"
                            ] += alert_data.get("false_positives", 0)
                            continental_totals[continent][method][alert_type][
                                "tn"
                            ] += alert_data.get("true_negatives", 0)
                            continental_totals[continent][method][alert_type][
                                "fn"
                            ] += alert_data.get("false_negatives", 0)

    # Calculate and report metrics
    report_lines.append(f"\nGLOBAL HEALTH WARNING PERFORMANCE:")
    report_lines.append(
        f"{'Method':<8} {'Alert':<10} {'Precision':<10} {'Recall':<8} {'FPR':<8} {'FNR':<8} {'F1':<8}"
    )
    report_lines.append(f"{'-'*68}")

    for method in methods:
        for alert_type in ["sensitive", "general"]:
            stats = method_totals[method][alert_type]

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
            fpr = (
                stats["fp"] / (stats["fp"] + stats["tn"])
                if (stats["fp"] + stats["tn"]) > 0
                else 0
            )
            fnr = (
                stats["fn"] / (stats["tp"] + stats["fn"])
                if (stats["tp"] + stats["fn"]) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            report_lines.append(
                f"{method.upper():<8} {alert_type:<10} {precision:<10.3f} {recall:<8.3f} {fpr:<8.3f} {fnr:<8.3f} {f1:<8.3f}"
            )

    # Ridge performance by continent
    report_lines.append(f"\nRIDGE HEALTH WARNING PERFORMANCE BY CONTINENT:")
    report_lines.append(
        f"{'Continent':<15} {'Alert':<10} {'Precision':<10} {'Recall':<8} {'FPR':<8} {'FNR':<8}"
    )
    report_lines.append(f"{'-'*62}")

    for continent in sorted(continental_totals.keys()):
        for alert_type in ["sensitive", "general"]:
            stats = continental_totals[continent]["ridge"][alert_type]

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
            fpr = (
                stats["fp"] / (stats["fp"] + stats["tn"])
                if (stats["fp"] + stats["tn"]) > 0
                else 0
            )
            fnr = (
                stats["fn"] / (stats["tp"] + stats["fn"])
                if (stats["tp"] + stats["fn"]) > 0
                else 0
            )

            report_lines.append(
                f"{continent:<15} {alert_type:<10} {precision:<10.3f} {recall:<8.3f} {fpr:<8.3f} {fnr:<8.3f}"
            )

    return "\n".join(report_lines), method_totals


def generate_city_rankings(results):
    """Generate city rankings by overall improvement."""

    report_lines = []
    report_lines.append(f"\n{'='*80}")
    report_lines.append(f"CITY RANKINGS BY OVERALL IMPROVEMENT")
    report_lines.append(f"{'='*80}")

    pollutants = ["PM25", "PM10", "NO2", "O3", "SO2", "CO", "AQI"]
    city_improvements = []

    for city_name, city_data in results.items():
        continent = city_data["continent"]
        country = city_data["country"]

        total_improvement = 0
        valid_pollutants = 0

        for pollutant in pollutants:
            if pollutant in city_data["results"]:
                data = city_data["results"][pollutant]

                ridge_mae = data["ridge"]["MAE"]
                cams_mae = data["cams"]["MAE"]
                noaa_mae = data["noaa"]["MAE"]

                best_benchmark_mae = min(cams_mae, noaa_mae)
                improvement = (
                    ((best_benchmark_mae - ridge_mae) / best_benchmark_mae) * 100
                    if best_benchmark_mae > 0
                    else 0
                )

                total_improvement += improvement
                valid_pollutants += 1

        avg_improvement = (
            total_improvement / valid_pollutants if valid_pollutants > 0 else 0
        )

        city_improvements.append(
            {
                "City": city_name,
                "Country": country,
                "Continent": continent,
                "Avg_Improvement_%": avg_improvement,
            }
        )

    df = pd.DataFrame(city_improvements)
    df = df.sort_values("Avg_Improvement_%", ascending=False)

    report_lines.append(f"\nTOP 20 CITIES BY AVERAGE IMPROVEMENT:")
    for i, (_, row) in enumerate(df.head(20).iterrows()):
        report_lines.append(
            f"{i+1:2d}. {row['City']:<15} ({row['Country']:<15}, {row['Continent']:<15}): {row['Avg_Improvement_%']:6.1f}%"
        )

    report_lines.append(f"\nBOTTOM 10 CITIES BY AVERAGE IMPROVEMENT:")
    for i, (_, row) in enumerate(df.tail(10).iterrows()):
        report_lines.append(
            f"{i+1:2d}. {row['City']:<15} ({row['Country']:<15}, {row['Continent']:<15}): {row['Avg_Improvement_%']:6.1f}%"
        )

    # Continental averages
    report_lines.append(f"\nAVERAGE IMPROVEMENT BY CONTINENT:")
    continental_avg = (
        df.groupby("Continent")["Avg_Improvement_%"].mean().sort_values(ascending=False)
    )
    for continent, avg in continental_avg.items():
        report_lines.append(f"{continent:<15}: {avg:6.1f}%")

    return "\n".join(report_lines), df


def main():
    """Generate comprehensive performance comparison report."""

    print("Generating comprehensive performance comparison report...")

    # Load results
    results = load_results()

    report_content = []
    report_content.append(f"COMPREHENSIVE PERFORMANCE COMPARISON REPORT")
    report_content.append(f"Ridge Regression vs CAMS vs NOAA Benchmarks")
    report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Dataset: {len(results)} cities evaluated")
    report_content.append(f"{'='*80}")

    # Analyze each pollutant
    pollutants = ["PM25", "PM10", "NO2", "O3", "SO2", "CO", "AQI"]
    pollutant_dfs = {}

    for pollutant in pollutants:
        print(f"Analyzing {pollutant}...")
        pollutant_report, pollutant_df = analyze_pollutant_comparison(
            results, pollutant
        )
        report_content.append(pollutant_report)
        pollutant_dfs[pollutant] = pollutant_df

    # Health warning analysis
    print("Analyzing health warnings...")
    health_report, health_totals = analyze_aqi_health_warnings(results)
    report_content.append(health_report)

    # City rankings
    print("Generating city rankings...")
    rankings_report, rankings_df = generate_city_rankings(results)
    report_content.append(rankings_report)

    # Overall summary
    report_content.append(f"\n{'='*80}")
    report_content.append(f"OVERALL PERFORMANCE SUMMARY")
    report_content.append(f"{'='*80}")

    overall_improvements = []
    for pollutant in pollutants:
        if pollutant in pollutant_dfs:
            avg_improvement = pollutant_dfs[pollutant]["Ridge_Improvement_%"].mean()
            overall_improvements.append(avg_improvement)
            report_content.append(
                f"{pollutant:<8}: {avg_improvement:6.1f}% average improvement over best benchmark"
            )

    total_avg = np.mean(overall_improvements)
    report_content.append(f"\nOVERALL AVERAGE IMPROVEMENT: {total_avg:.1f}%")

    # Ridge health warning summary
    ridge_sensitive = health_totals["ridge"]["sensitive"]
    ridge_general = health_totals["ridge"]["general"]

    sens_precision = (
        ridge_sensitive["tp"] / (ridge_sensitive["tp"] + ridge_sensitive["fp"])
        if (ridge_sensitive["tp"] + ridge_sensitive["fp"]) > 0
        else 0
    )
    sens_fnr = (
        ridge_sensitive["fn"] / (ridge_sensitive["tp"] + ridge_sensitive["fn"])
        if (ridge_sensitive["tp"] + ridge_sensitive["fn"]) > 0
        else 0
    )

    gen_precision = (
        ridge_general["tp"] / (ridge_general["tp"] + ridge_general["fp"])
        if (ridge_general["tp"] + ridge_general["fp"]) > 0
        else 0
    )
    gen_fnr = (
        ridge_general["fn"] / (ridge_general["tp"] + ridge_general["fn"])
        if (ridge_general["tp"] + ridge_general["fn"]) > 0
        else 0
    )

    report_content.append(f"\nRIDGE HEALTH WARNING PERFORMANCE:")
    report_content.append(
        f"Sensitive Population: {sens_precision:.1%} precision, {sens_fnr:.1%} false negative rate"
    )
    report_content.append(
        f"General Population: {gen_precision:.1%} precision, {gen_fnr:.1%} false negative rate"
    )

    # Cities analysis summary
    total_cities = len(results)
    cities_improved = sum(
        1
        for df in pollutant_dfs.values()
        for _, row in df.iterrows()
        if row["Ridge_Improvement_%"] > 0
    ) / len(pollutant_dfs)

    report_content.append(f"\nCITY PERFORMANCE SUMMARY:")
    report_content.append(f"Total cities evaluated: {total_cities}")
    report_content.append(
        f"Average cities where Ridge outperformed benchmarks: {cities_improved:.0f} cities per pollutant"
    )
    report_content.append(
        f"Best performing city overall: {rankings_df.iloc[0]['City']} ({rankings_df.iloc[0]['Avg_Improvement_%']:.1f}% improvement)"
    )

    # Write report to file
    output_file = "../final_dataset/DETAILED_PERFORMANCE_COMPARISON_REPORT.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print(f"\nReport generated: {output_file}")
    print("Report summary:")
    print(f"- {total_avg:.1f}% average improvement across all pollutants")
    print(f"- {sens_precision:.1%} precision for sensitive population health warnings")
    print(f"- {total_cities} cities evaluated across 5 continents")

    return report_content


if __name__ == "__main__":
    main()
