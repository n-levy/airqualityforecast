#!/usr/bin/env python3
"""
Benchmark Health Warning Analysis

Detailed analysis of CAMS and NOAA health warning performance
compared to Ridge regression across all cities and continents.
"""

import json

import pandas as pd


def load_results(
    results_file="../final_dataset/full_100_city_results_20250911_121246.json",
):
    """Load evaluation results."""
    with open(results_file, "r") as f:
        return json.load(f)


def analyze_benchmark_health_warnings():
    """Analyze health warning performance for all methods."""

    results = load_results()

    print("=" * 80)
    print("BENCHMARK HEALTH WARNING PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Aggregate by method and continent
    methods = ["ridge", "cams", "noaa"]
    continents = ["Asia", "Africa", "Europe", "North_America", "South_America"]

    # Global totals
    global_totals = {}
    for method in methods:
        global_totals[method] = {
            "sensitive": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "general": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        }

    # Continental totals
    continental_totals = {}
    for continent in continents:
        continental_totals[continent] = {}
        for method in methods:
            continental_totals[continent][method] = {
                "sensitive": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "general": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            }

    # Collect data
    cities_by_continent = {continent: 0 for continent in continents}

    for city_name, city_data in results.items():
        continent = city_data["continent"]
        cities_by_continent[continent] += 1

        if "health_warnings" in city_data["results"]:
            health_data = city_data["results"]["health_warnings"]

            for method in methods:
                if method in health_data:
                    for alert_type in ["sensitive", "general"]:
                        if alert_type in health_data[method]:
                            alert = health_data[method][alert_type]

                            # Global aggregation
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

                            # Continental aggregation
                            continental_totals[continent][method][alert_type][
                                "tp"
                            ] += alert.get("true_positives", 0)
                            continental_totals[continent][method][alert_type][
                                "fp"
                            ] += alert.get("false_positives", 0)
                            continental_totals[continent][method][alert_type][
                                "tn"
                            ] += alert.get("true_negatives", 0)
                            continental_totals[continent][method][alert_type][
                                "fn"
                            ] += alert.get("false_negatives", 0)

    # Display global performance
    print("\nGLOBAL HEALTH WARNING PERFORMANCE COMPARISON:")
    print(
        f"{'Method':<8} {'Alert Type':<12} {'Precision':<10} {'Recall':<8} {'FPR':<8} {'FNR':<8} {'F1':<8} {'Rating':<12}"
    )
    print("-" * 88)

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

            # Rate performance
            if alert_type == "sensitive":
                if fnr < 0.05:
                    rating = "EXCELLENT"
                elif fnr < 0.10:
                    rating = "GOOD"
                elif fnr < 0.15:
                    rating = "ACCEPTABLE"
                else:
                    rating = "POOR"
            else:  # general
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
                "fpr": fpr,
                "fnr": fnr,
                "f1": f1,
                "rating": rating,
            }

            print(
                f"{method.upper():<8} {alert_type:<12} {precision:<10.3f} {recall:<8.3f} {fpr:<8.3f} {fnr:<8.3f} {f1:<8.3f} {rating:<12}"
            )

    # Benchmark comparison
    print(f"\nBENCHMARK PERFORMANCE ASSESSMENT:")
    print(
        f"{'Metric':<25} {'Ridge':<12} {'CAMS':<12} {'NOAA':<12} {'Best Benchmark':<15}"
    )
    print("-" * 80)

    for alert_type in ["sensitive", "general"]:
        ridge_fnr = method_ratings["ridge"][alert_type]["fnr"]
        cams_fnr = method_ratings["cams"][alert_type]["fnr"]
        noaa_fnr = method_ratings["noaa"][alert_type]["fnr"]

        ridge_fpr = method_ratings["ridge"][alert_type]["fpr"]
        cams_fpr = method_ratings["cams"][alert_type]["fpr"]
        noaa_fpr = method_ratings["noaa"][alert_type]["fpr"]

        ridge_prec = method_ratings["ridge"][alert_type]["precision"]
        cams_prec = method_ratings["cams"][alert_type]["precision"]
        noaa_prec = method_ratings["noaa"][alert_type]["precision"]

        best_benchmark_fnr = min(cams_fnr, noaa_fnr)
        best_benchmark_fpr = min(cams_fpr, noaa_fpr)
        best_benchmark_prec = max(cams_prec, noaa_prec)

        print(
            f"{alert_type.title() + ' FNR (lower better)':<25} {ridge_fnr:<12.3f} {cams_fnr:<12.3f} {noaa_fnr:<12.3f} {best_benchmark_fnr:<15.3f}"
        )
        print(
            f"{alert_type.title() + ' FPR (lower better)':<25} {ridge_fpr:<12.3f} {cams_fpr:<12.3f} {noaa_fpr:<12.3f} {best_benchmark_fpr:<15.3f}"
        )
        print(
            f"{alert_type.title() + ' Precision (higher better)':<25} {ridge_prec:<12.3f} {cams_prec:<12.3f} {noaa_prec:<12.3f} {best_benchmark_prec:<15.3f}"
        )
        print()

    # Continental performance for each method
    print("BENCHMARK PERFORMANCE BY CONTINENT:")
    print(
        f"{'Continent':<15} {'Method':<8} {'Sensitive FNR':<14} {'General FNR':<12} {'Sensitive FPR':<14} {'General FPR':<12} {'Rating':<12}"
    )
    print("-" * 98)

    for continent in continents:
        print(f"\n{continent.upper()} ({cities_by_continent[continent]} cities):")

        for method in methods:
            sens_stats = continental_totals[continent][method]["sensitive"]
            gen_stats = continental_totals[continent][method]["general"]

            sens_fnr = (
                sens_stats["fn"] / (sens_stats["tp"] + sens_stats["fn"])
                if (sens_stats["tp"] + sens_stats["fn"]) > 0
                else 0
            )
            gen_fnr = (
                gen_stats["fn"] / (gen_stats["tp"] + gen_stats["fn"])
                if (gen_stats["tp"] + gen_stats["fn"]) > 0
                else 0
            )

            sens_fpr = (
                sens_stats["fp"] / (sens_stats["fp"] + sens_stats["tn"])
                if (sens_stats["fp"] + sens_stats["tn"]) > 0
                else 0
            )
            gen_fpr = (
                gen_stats["fp"] / (gen_stats["fp"] + gen_stats["tn"])
                if (gen_stats["fp"] + gen_stats["tn"]) > 0
                else 0
            )

            # Overall rating for continent
            avg_fnr = (sens_fnr + gen_fnr) / 2
            if avg_fnr < 0.05:
                rating = "EXCELLENT"
            elif avg_fnr < 0.10:
                rating = "GOOD"
            elif avg_fnr < 0.15:
                rating = "ACCEPTABLE"
            else:
                rating = "POOR"

            print(
                f"{'':15} {method.upper():<8} {sens_fnr:<14.3f} {gen_fnr:<12.3f} {sens_fpr:<14.3f} {gen_fpr:<12.3f} {rating:<12}"
            )

    # Health impact assessment
    print(f"\nHEALTH IMPACT ASSESSMENT:")
    print(
        f"{'Method':<8} {'Sensitive Pop.':<15} {'General Pop.':<15} {'Overall Assessment':<20}"
    )
    print("-" * 65)

    for method in methods:
        sens_fnr = method_ratings[method]["sensitive"]["fnr"]
        gen_fnr = method_ratings[method]["general"]["fnr"]

        if sens_fnr < 0.01 and gen_fnr < 0.01:
            assessment = "OUTSTANDING"
        elif sens_fnr < 0.05 and gen_fnr < 0.05:
            assessment = "EXCELLENT"
        elif sens_fnr < 0.10 and gen_fnr < 0.10:
            assessment = "GOOD"
        elif sens_fnr < 0.15 and gen_fnr < 0.15:
            assessment = "ACCEPTABLE"
        else:
            assessment = "NEEDS IMPROVEMENT"

        sens_rating = method_ratings[method]["sensitive"]["rating"]
        gen_rating = method_ratings[method]["general"]["rating"]

        print(
            f"{method.upper():<8} {sens_rating:<15} {gen_rating:<15} {assessment:<20}"
        )

    # Summary comparison
    print(f"\nSUMMARY: RIDGE vs BENCHMARKS")
    print("=" * 50)

    ridge_sens_fnr = method_ratings["ridge"]["sensitive"]["fnr"]
    cams_sens_fnr = method_ratings["cams"]["sensitive"]["fnr"]
    noaa_sens_fnr = method_ratings["noaa"]["sensitive"]["fnr"]

    ridge_gen_fnr = method_ratings["ridge"]["general"]["fnr"]
    cams_gen_fnr = method_ratings["cams"]["general"]["fnr"]
    noaa_gen_fnr = method_ratings["noaa"]["general"]["fnr"]

    print(f"Sensitive Population False Negative Rates:")
    print(
        f"  Ridge: {ridge_sens_fnr:.1%} | CAMS: {cams_sens_fnr:.1%} | NOAA: {noaa_sens_fnr:.1%}"
    )

    print(f"General Population False Negative Rates:")
    print(
        f"  Ridge: {ridge_gen_fnr:.1%} | CAMS: {cams_gen_fnr:.1%} | NOAA: {noaa_gen_fnr:.1%}"
    )

    # Determine winner
    ridge_total_fnr = ridge_sens_fnr + ridge_gen_fnr
    cams_total_fnr = cams_sens_fnr + cams_gen_fnr
    noaa_total_fnr = noaa_sens_fnr + noaa_gen_fnr

    if ridge_total_fnr <= cams_total_fnr and ridge_total_fnr <= noaa_total_fnr:
        winner = "Ridge Regression"
        advantage = (
            min(
                (cams_total_fnr - ridge_total_fnr) / ridge_total_fnr,
                (noaa_total_fnr - ridge_total_fnr) / ridge_total_fnr,
            )
            * 100
        )
    elif cams_total_fnr <= noaa_total_fnr:
        winner = "CAMS"
        advantage = (ridge_total_fnr - cams_total_fnr) / cams_total_fnr * 100
    else:
        winner = "NOAA"
        advantage = (ridge_total_fnr - noaa_total_fnr) / noaa_total_fnr * 100

    print(f"\nBEST HEALTH WARNING PERFORMANCE: {winner}")
    print(f"Advantage over other methods: {advantage:.0f}%")

    return method_ratings, continental_totals


if __name__ == "__main__":
    analyze_benchmark_health_warnings()
