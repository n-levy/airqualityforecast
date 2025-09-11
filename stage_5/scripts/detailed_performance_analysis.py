#!/usr/bin/env python3
"""
Detailed Performance Analysis Script

Compare Ridge regression vs CAMS vs NOAA benchmarks across all cities,
examining individual pollutants, AQI scores, and health warning performance.
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DetailedPerformanceAnalyzer:
    """Analyze detailed performance comparisons across all cities."""

    def __init__(
        self, results_file="../final_dataset/full_100_city_results_20250911_121246.json"
    ):
        """Initialize with results file."""
        self.results_file = results_file
        self.results = self.load_results()
        self.pollutants = ["PM25", "PM10", "NO2", "O3", "SO2", "CO", "AQI"]
        self.methods = ["ridge", "cams", "noaa"]

    def load_results(self):
        """Load evaluation results."""
        with open(self.results_file, "r") as f:
            return json.load(f)

    def analyze_pollutant_performance(self, pollutant):
        """Analyze performance for a specific pollutant across all cities."""
        print(f"\n{'='*80}")
        print(f"POLLUTANT ANALYSIS: {pollutant}")
        print(f"{'='*80}")

        city_performance = []

        for city_name, city_data in self.results.items():
            continent = city_data["continent"]

            if pollutant in city_data["results"]:
                pollutant_data = city_data["results"][pollutant]

                ridge_mae = pollutant_data["ridge"]["MAE"]
                cams_mae = pollutant_data["cams"]["MAE"]
                noaa_mae = pollutant_data["noaa"]["MAE"]

                ridge_r2 = pollutant_data["ridge"]["R2"]
                cams_r2 = pollutant_data["cams"]["R2"]
                noaa_r2 = pollutant_data["noaa"]["R2"]

                # Calculate improvements
                ridge_vs_cams = (
                    ((cams_mae - ridge_mae) / cams_mae) * 100 if cams_mae > 0 else 0
                )
                ridge_vs_noaa = (
                    ((noaa_mae - ridge_mae) / noaa_mae) * 100 if noaa_mae > 0 else 0
                )
                best_benchmark_mae = min(cams_mae, noaa_mae)
                ridge_vs_best = (
                    ((best_benchmark_mae - ridge_mae) / best_benchmark_mae) * 100
                    if best_benchmark_mae > 0
                    else 0
                )

                city_performance.append(
                    {
                        "City": city_name,
                        "Continent": continent,
                        "Ridge_MAE": ridge_mae,
                        "CAMS_MAE": cams_mae,
                        "NOAA_MAE": noaa_mae,
                        "Ridge_R2": ridge_r2,
                        "CAMS_R2": cams_r2,
                        "NOAA_R2": noaa_r2,
                        "Ridge_vs_CAMS_%": ridge_vs_cams,
                        "Ridge_vs_NOAA_%": ridge_vs_noaa,
                        "Ridge_vs_Best_%": ridge_vs_best,
                    }
                )

        df = pd.DataFrame(city_performance)

        # Overall statistics
        print(f"\nOVERALL PERFORMANCE STATISTICS:")
        print(
            f"Average MAE - Ridge: {df['Ridge_MAE'].mean():.3f}, CAMS: {df['CAMS_MAE'].mean():.3f}, NOAA: {df['NOAA_MAE'].mean():.3f}"
        )
        print(
            f"Average R² - Ridge: {df['Ridge_R2'].mean():.3f}, CAMS: {df['CAMS_R2'].mean():.3f}, NOAA: {df['NOAA_R2'].mean():.3f}"
        )
        print(f"Ridge vs CAMS improvement: {df['Ridge_vs_CAMS_%'].mean():.1f}% average")
        print(f"Ridge vs NOAA improvement: {df['Ridge_vs_NOAA_%'].mean():.1f}% average")
        print(f"Ridge vs Best Benchmark: {df['Ridge_vs_Best_%'].mean():.1f}% average")

        # Continental breakdown
        print(f"\nCONTINENTAL BREAKDOWN:")
        continental_stats = (
            df.groupby("Continent")
            .agg(
                {
                    "Ridge_MAE": "mean",
                    "CAMS_MAE": "mean",
                    "NOAA_MAE": "mean",
                    "Ridge_vs_Best_%": "mean",
                }
            )
            .round(3)
        )
        print(continental_stats.to_string())

        # Top performers
        print(f"\nTOP 10 RIDGE IMPROVEMENTS (vs Best Benchmark):")
        top_improvements = df.nlargest(10, "Ridge_vs_Best_%")[
            ["City", "Continent", "Ridge_vs_Best_%"]
        ]
        print(top_improvements.to_string(index=False))

        # Cities where Ridge underperformed
        underperformed = df[df["Ridge_vs_Best_%"] < 0]
        if len(underperformed) > 0:
            print(f"\nCITIES WHERE RIDGE UNDERPERFORMED:")
            print(
                underperformed[["City", "Continent", "Ridge_vs_Best_%"]].to_string(
                    index=False
                )
            )
        else:
            print(f"\n✅ Ridge outperformed benchmarks in ALL {len(df)} cities!")

        return df

    def analyze_aqi_performance(self):
        """Detailed AQI performance analysis."""
        print(f"\n{'='*80}")
        print(f"AQI PERFORMANCE ANALYSIS")
        print(f"{'='*80}")

        return self.analyze_pollutant_performance("AQI")

    def analyze_health_warnings(self):
        """Analyze health warning confusion matrices across all cities."""
        print(f"\n{'='*80}")
        print(f"HEALTH WARNING CONFUSION MATRIX ANALYSIS")
        print(f"{'='*80}")

        # Aggregate confusion matrices
        method_totals = {}
        for method in self.methods:
            method_totals[method] = {
                "sensitive": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "general": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            }

        continental_breakdown = {}

        for city_name, city_data in self.results.items():
            continent = city_data["continent"]

            if continent not in continental_breakdown:
                continental_breakdown[continent] = {}
                for method in self.methods:
                    continental_breakdown[continent][method] = {
                        "sensitive": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                        "general": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                    }

            if "health_warnings" in city_data["results"]:
                health_data = city_data["results"]["health_warnings"]

                for method in self.methods:
                    if method in health_data:
                        for alert_type in ["sensitive", "general"]:
                            if alert_type in health_data[method]:
                                alert_data = health_data[method][alert_type]

                                # Global totals
                                method_totals[method][alert_type][
                                    "tp"
                                ] += alert_data.get("true_positives", 0)
                                method_totals[method][alert_type][
                                    "fp"
                                ] += alert_data.get("false_positives", 0)
                                method_totals[method][alert_type][
                                    "tn"
                                ] += alert_data.get("true_negatives", 0)
                                method_totals[method][alert_type][
                                    "fn"
                                ] += alert_data.get("false_negatives", 0)

                                # Continental totals
                                continental_breakdown[continent][method][alert_type][
                                    "tp"
                                ] += alert_data.get("true_positives", 0)
                                continental_breakdown[continent][method][alert_type][
                                    "fp"
                                ] += alert_data.get("false_positives", 0)
                                continental_breakdown[continent][method][alert_type][
                                    "tn"
                                ] += alert_data.get("true_negatives", 0)
                                continental_breakdown[continent][method][alert_type][
                                    "fn"
                                ] += alert_data.get("false_negatives", 0)

        # Calculate and display metrics
        print(f"\nGLOBAL HEALTH WARNING PERFORMANCE:")
        print(
            f"{'Method':<8} {'Alert Type':<12} {'Precision':<10} {'Recall':<8} {'FPR':<8} {'FNR':<8} {'F1':<8}"
        )
        print(f"{'-'*70}")

        for method in self.methods:
            for alert_type in ["sensitive", "general"]:
                stats = method_totals[method][alert_type]
                total = sum(stats.values())

                if total > 0:
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

                    print(
                        f"{method.upper():<8} {alert_type:<12} {precision:<10.3f} {recall:<8.3f} {fpr:<8.3f} {fnr:<8.3f} {f1:<8.3f}"
                    )

        # Continental breakdown for Ridge (best method)
        print(f"\nRIDGE PERFORMANCE BY CONTINENT:")
        print(
            f"{'Continent':<15} {'Alert Type':<12} {'Precision':<10} {'Recall':<8} {'FPR':<8} {'FNR':<8}"
        )
        print(f"{'-'*70}")

        for continent in sorted(continental_breakdown.keys()):
            for alert_type in ["sensitive", "general"]:
                stats = continental_breakdown[continent]["ridge"][alert_type]
                total = sum(stats.values())

                if total > 0:
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

                    print(
                        f"{continent:<15} {alert_type:<12} {precision:<10.3f} {recall:<8.3f} {fpr:<8.3f} {fnr:<8.3f}"
                    )

        return method_totals, continental_breakdown

    def generate_city_rankings(self):
        """Generate city rankings by improvement over benchmarks."""
        print(f"\n{'='*80}")
        print(f"CITY RANKINGS BY OVERALL IMPROVEMENT")
        print(f"{'='*80}")

        city_improvements = []

        for city_name, city_data in self.results.items():
            continent = city_data["continent"]
            country = city_data["country"]

            total_improvement = 0
            valid_pollutants = 0

            for pollutant in self.pollutants:
                if pollutant in city_data["results"]:
                    pollutant_data = city_data["results"][pollutant]

                    ridge_mae = pollutant_data["ridge"]["MAE"]
                    cams_mae = pollutant_data["cams"]["MAE"]
                    noaa_mae = pollutant_data["noaa"]["MAE"]

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
                    "Pollutants_Analyzed": valid_pollutants,
                }
            )

        df = pd.DataFrame(city_improvements)
        df = df.sort_values("Avg_Improvement_%", ascending=False)

        print(f"\nTOP 20 CITIES BY AVERAGE IMPROVEMENT:")
        print(
            df.head(20)[
                ["City", "Country", "Continent", "Avg_Improvement_%"]
            ].to_string(index=False)
        )

        print(f"\nBOTTOM 10 CITIES BY AVERAGE IMPROVEMENT:")
        print(
            df.tail(10)[
                ["City", "Country", "Continent", "Avg_Improvement_%"]
            ].to_string(index=False)
        )

        print(f"\nIMPROVEMENT BY CONTINENT:")
        continental_improvement = (
            df.groupby("Continent")["Avg_Improvement_%"]
            .agg(["mean", "std", "count"])
            .round(2)
        )
        print(continental_improvement.to_string())

        return df

    def run_complete_analysis(self):
        """Run complete performance analysis."""
        print(f"DETAILED PERFORMANCE ANALYSIS")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dataset: {len(self.results)} cities evaluated")

        # Analyze each pollutant
        pollutant_results = {}
        for pollutant in self.pollutants:
            pollutant_results[pollutant] = self.analyze_pollutant_performance(pollutant)

        # Health warning analysis
        health_totals, continental_health = self.analyze_health_warnings()

        # City rankings
        city_rankings = self.generate_city_rankings()

        # Summary
        print(f"\n{'='*80}")
        print(f"ANALYSIS SUMMARY")
        print(f"{'='*80}")

        overall_improvements = []
        for pollutant in self.pollutants:
            if pollutant in pollutant_results:
                avg_improvement = pollutant_results[pollutant]["Ridge_vs_Best_%"].mean()
                overall_improvements.append(avg_improvement)
                print(f"{pollutant}: {avg_improvement:.1f}% average improvement")

        total_avg = np.mean(overall_improvements)
        print(f"\nOVERALL AVERAGE IMPROVEMENT: {total_avg:.1f}%")

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

        print(f"\nRIDGE HEALTH WARNING PERFORMANCE:")
        print(
            f"Sensitive Population: {sens_precision:.1%} precision, {sens_fnr:.1%} false negative rate"
        )
        print(
            f"General Population: {gen_precision:.1%} precision, {gen_fnr:.1%} false negative rate"
        )

        return {
            "pollutant_results": pollutant_results,
            "health_analysis": (health_totals, continental_health),
            "city_rankings": city_rankings,
        }


def main():
    """Main execution function."""
    analyzer = DetailedPerformanceAnalyzer()
    results = analyzer.run_complete_analysis()
    return results


if __name__ == "__main__":
    results = main()
