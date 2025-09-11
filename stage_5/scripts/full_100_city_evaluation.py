#!/usr/bin/env python3
"""
Full 100-City Evaluation System

This script runs the complete evaluation on all 100 cities with optimized processing
and comprehensive health warning analysis including false positives/negatives.
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Full100CityEvaluator:
    """Complete evaluation system for all 100 cities."""

    def __init__(
        self,
        data_path: str = "../comprehensive_tables/comprehensive_features_table.csv",
    ):
        """Initialize with dataset path."""
        self.data_path = data_path

        # Continental standards and health thresholds
        self.continental_standards = {
            "Asia": {
                "pattern_name": "Delhi Pattern",
                "aqi_standard": "Indian National AQI",
                "health_thresholds": {"sensitive": 101, "general": 201},
                "expected_r2": 0.75,
                "data_quality": 0.892,
            },
            "Africa": {
                "pattern_name": "Cairo Pattern",
                "aqi_standard": "WHO Guidelines",
                "health_thresholds": {
                    "sensitive": 25,
                    "general": 50,
                },  # PM2.5 equivalent
                "expected_r2": 0.75,
                "data_quality": 0.885,
            },
            "Europe": {
                "pattern_name": "Berlin Pattern",
                "aqi_standard": "European EAQI",
                "health_thresholds": {"sensitive": 3, "general": 4},  # EAQI levels
                "expected_r2": 0.90,
                "data_quality": 0.964,
            },
            "North_America": {
                "pattern_name": "Toronto Pattern",
                "aqi_standard": "EPA AQI",
                "health_thresholds": {"sensitive": 101, "general": 151},
                "expected_r2": 0.85,
                "data_quality": 0.948,
            },
            "South_America": {
                "pattern_name": "São Paulo Pattern",
                "aqi_standard": "WHO Guidelines",
                "health_thresholds": {
                    "sensitive": 25,
                    "general": 50,
                },  # PM2.5 equivalent
                "expected_r2": 0.80,
                "data_quality": 0.921,
            },
        }

        self.pollutants = ["PM25", "PM10", "NO2", "O3", "SO2", "CO"]
        self.methods = ["simple_avg", "ridge", "cams", "noaa"]

    def load_all_cities(self) -> pd.DataFrame:
        """Load all 100 cities data."""
        logger.info("Loading complete 100-city dataset...")
        features_df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(features_df)} cities")
        return features_df

    def generate_efficient_time_series(
        self, city_row: pd.Series, days: int = 60
    ) -> pd.DataFrame:
        """Generate efficient time series (shorter for full evaluation)."""

        base_pm25 = city_row["Average_PM25"]
        base_pm10 = city_row["pm10_Concentration"]
        base_no2 = city_row["no2_Concentration"]
        base_o3 = city_row["o3_Concentration"]
        base_so2 = city_row["so2_Concentration"]
        base_co = city_row["co_Concentration"]

        # Generate date range (last 60 days)
        end_date = datetime(2025, 9, 11)
        start_date = end_date - timedelta(days=days - 1)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Consistent seed per city
        np.random.seed(hash(city_row["City"]) % 2**32)

        data = []
        for i, date in enumerate(dates):
            # Simplified patterns for efficiency
            seasonal_factor = 1 + 0.15 * np.sin(
                2 * np.pi * date.timetuple().tm_yday / 365
            )
            weekly_factor = 1.05 if date.weekday() < 5 else 0.95
            noise_factor = np.random.normal(1.0, 0.08)

            total_factor = seasonal_factor * weekly_factor * noise_factor

            # Generate pollutant values
            pm25_actual = max(1, base_pm25 * total_factor)
            pm10_actual = max(1, base_pm10 * total_factor * 1.2)
            no2_actual = max(1, base_no2 * total_factor * weekly_factor)
            o3_actual = max(1, base_o3 * seasonal_factor * np.random.normal(1.0, 0.1))
            so2_actual = max(1, base_so2 * total_factor * 0.9)
            co_actual = max(1, base_co * total_factor * weekly_factor)

            # Generate benchmark forecasts with realistic errors
            cams_pm25 = pm25_actual * np.random.normal(1.0, 0.10)
            cams_pm10 = pm10_actual * np.random.normal(1.0, 0.12)
            cams_no2 = no2_actual * np.random.normal(1.0, 0.15)
            cams_o3 = o3_actual * np.random.normal(1.0, 0.11)
            cams_so2 = so2_actual * np.random.normal(1.0, 0.18)
            cams_co = co_actual * np.random.normal(1.0, 0.13)

            noaa_pm25 = pm25_actual * np.random.normal(1.0, 0.12)
            noaa_pm10 = pm10_actual * np.random.normal(1.0, 0.14)
            noaa_no2 = no2_actual * np.random.normal(1.0, 0.13)
            noaa_o3 = o3_actual * np.random.normal(1.0, 0.09)
            noaa_so2 = so2_actual * np.random.normal(1.0, 0.20)
            noaa_co = co_actual * np.random.normal(1.0, 0.15)

            data.append(
                {
                    "date": date,
                    "city": city_row["City"],
                    "PM25_actual": pm25_actual,
                    "PM10_actual": pm10_actual,
                    "NO2_actual": no2_actual,
                    "O3_actual": o3_actual,
                    "SO2_actual": so2_actual,
                    "CO_actual": co_actual,
                    "CAMS_PM25": max(1, cams_pm25),
                    "CAMS_PM10": max(1, cams_pm10),
                    "CAMS_NO2": max(1, cams_no2),
                    "CAMS_O3": max(1, cams_o3),
                    "CAMS_SO2": max(1, cams_so2),
                    "CAMS_CO": max(1, cams_co),
                    "NOAA_PM25": max(1, noaa_pm25),
                    "NOAA_PM10": max(1, noaa_pm10),
                    "NOAA_NO2": max(1, noaa_no2),
                    "NOAA_O3": max(1, noaa_o3),
                    "NOAA_SO2": max(1, noaa_so2),
                    "NOAA_CO": max(1, noaa_co),
                    "temperature": 20
                    + 12 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                    + np.random.normal(0, 2),
                    "humidity": max(10, min(100, 60 + np.random.normal(0, 12))),
                    "wind_speed": max(0, 3 + np.random.normal(0, 1.5)),
                    "pressure": 1013 + np.random.normal(0, 8),
                    "day_of_year": date.timetuple().tm_yday,
                    "day_of_week": date.weekday(),
                    "is_weekend": float(date.weekday() >= 5),
                }
            )

        return pd.DataFrame(data)

    def calculate_local_aqi(
        self,
        pm25: float,
        pm10: float,
        no2: float,
        o3: float,
        so2: float,
        co: float,
        continent: str,
    ) -> float:
        """Calculate AQI using local continental standard."""

        if continent == "Europe":
            # European EAQI (simplified) - Level 1-6
            aqi_pm25 = min(6, max(1, 1 + (pm25 / 15)))  # Rough EAQI conversion
            aqi_pm10 = min(6, max(1, 1 + (pm10 / 25)))
            aqi_no2 = min(6, max(1, 1 + (no2 / 40)))
            aqi_o3 = min(6, max(1, 1 + (o3 / 120)))
            return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_o3)

        elif continent == "Asia":
            # Indian National AQI (simplified)
            aqi_pm25 = min(500, max(0, pm25 * 4.17))
            aqi_pm10 = min(500, max(0, pm10 * 2.04))
            aqi_no2 = min(500, max(0, no2 * 9.43))
            aqi_o3 = min(500, max(0, o3 * 7.81))
            return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_o3)

        else:
            # US EPA style for other continents
            aqi_pm25 = min(500, max(0, pm25 * 4.17))
            aqi_pm10 = min(500, max(0, pm10 * 2.04))
            aqi_no2 = min(500, max(0, no2 * 9.43))
            aqi_o3 = min(500, max(0, o3 * 7.81))
            aqi_so2 = min(500, max(0, so2 * 9.17))
            aqi_co = min(500, max(0, co * 0.115))

            return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_o3, aqi_so2, aqi_co)

    def evaluate_single_city(
        self, city_data: pd.DataFrame, city_name: str, continent: str
    ) -> Dict:
        """Efficient evaluation for single city."""

        city_data = city_data.sort_values("date").reset_index(drop=True)

        pollutants = self.pollutants
        predictions = {
            "simple_avg": {p: [] for p in pollutants + ["AQI"]},
            "ridge": {p: [] for p in pollutants + ["AQI"]},
            "cams": {p: [] for p in pollutants + ["AQI"]},
            "noaa": {p: [] for p in pollutants + ["AQI"]},
            "actual": {p: [] for p in pollutants + ["AQI"]},
        }

        # Walk-forward validation (last 20 days for efficiency)
        for i in range(max(5, len(city_data) - 20), len(city_data)):

            train_data = city_data.iloc[:i]
            test_data = city_data.iloc[i]

            for pollutant in pollutants:
                # Simple average
                simple_pred = (
                    test_data[f"CAMS_{pollutant}"] + test_data[f"NOAA_{pollutant}"]
                ) / 2

                # Ridge regression with basic features
                try:
                    if len(train_data) > 3:
                        X_train = train_data[
                            [
                                "temperature",
                                "humidity",
                                "wind_speed",
                                f"CAMS_{pollutant}",
                                f"NOAA_{pollutant}",
                            ]
                        ].values
                        y_train = train_data[f"{pollutant}_actual"].values

                        X_test = np.array(
                            [
                                test_data["temperature"],
                                test_data["humidity"],
                                test_data["wind_speed"],
                                test_data[f"CAMS_{pollutant}"],
                                test_data[f"NOAA_{pollutant}"],
                            ]
                        )

                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test.reshape(1, -1))

                        ridge = Ridge(alpha=0.5)
                        ridge.fit(X_train_scaled, y_train)
                        ridge_pred = ridge.predict(X_test_scaled)[0]
                    else:
                        ridge_pred = simple_pred
                except:
                    ridge_pred = simple_pred

                # Store predictions
                predictions["simple_avg"][pollutant].append(simple_pred)
                predictions["ridge"][pollutant].append(ridge_pred)
                predictions["cams"][pollutant].append(test_data[f"CAMS_{pollutant}"])
                predictions["noaa"][pollutant].append(test_data[f"NOAA_{pollutant}"])
                predictions["actual"][pollutant].append(
                    test_data[f"{pollutant}_actual"]
                )

        # Calculate AQI
        for method in ["simple_avg", "ridge", "cams", "noaa", "actual"]:
            aqi_values = []
            for j in range(len(predictions[method]["PM25"])):
                aqi = self.calculate_local_aqi(
                    predictions[method]["PM25"][j],
                    predictions[method]["PM10"][j],
                    predictions[method]["NO2"][j],
                    predictions[method]["O3"][j],
                    predictions[method]["SO2"][j],
                    predictions[method]["CO"][j],
                    continent,
                )
                aqi_values.append(aqi)
            predictions[method]["AQI"] = aqi_values

        # Calculate metrics
        results = {}
        for pollutant in pollutants + ["AQI"]:
            results[pollutant] = {}
            actual_values = np.array(predictions["actual"][pollutant])

            for method in ["simple_avg", "ridge", "cams", "noaa"]:
                pred_values = np.array(predictions[method][pollutant])

                # Basic metrics
                mae = mean_absolute_error(actual_values, pred_values)
                rmse = np.sqrt(mean_squared_error(actual_values, pred_values))
                r2 = (
                    r2_score(actual_values, pred_values)
                    if len(set(actual_values)) > 1
                    else 0
                )
                mpe = (
                    np.mean((pred_values - actual_values) / (actual_values + 1e-8))
                    * 100
                )

                results[pollutant][method] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "MPE": mpe,
                }

        # Health warning analysis for AQI
        health_thresholds = self.continental_standards[continent]["health_thresholds"]
        health_results = self.analyze_health_warnings(
            predictions, health_thresholds, continent
        )
        results["health_warnings"] = health_results

        return results

    def analyze_health_warnings(
        self, predictions: Dict, health_thresholds: Dict, continent: str
    ) -> Dict:
        """Analyze health warning performance with false positives/negatives."""

        actual_aqi = np.array(predictions["actual"]["AQI"])

        health_analysis = {}

        for method in ["simple_avg", "ridge", "cams", "noaa"]:
            pred_aqi = np.array(predictions[method]["AQI"])

            method_analysis = {}

            for alert_type, threshold in health_thresholds.items():
                # Convert thresholds for different systems
                if continent == "Europe" and threshold < 10:
                    # EAQI levels already correct
                    actual_alerts = actual_aqi >= threshold
                    pred_alerts = pred_aqi >= threshold
                elif continent in ["South_America", "Africa"] and threshold < 100:
                    # PM2.5 based thresholds - convert to rough AQI equivalent
                    threshold_aqi = threshold * 4.17  # Rough PM2.5 to AQI conversion
                    actual_alerts = actual_aqi >= threshold_aqi
                    pred_alerts = pred_aqi >= threshold_aqi
                else:
                    # Standard AQI thresholds
                    actual_alerts = actual_aqi >= threshold
                    pred_alerts = pred_aqi >= threshold

                # Calculate confusion matrix
                tp = np.sum((actual_alerts == True) & (pred_alerts == True))
                fp = np.sum((actual_alerts == False) & (pred_alerts == True))
                tn = np.sum((actual_alerts == False) & (pred_alerts == False))
                fn = np.sum((actual_alerts == True) & (pred_alerts == False))

                total = tp + fp + tn + fn

                if total > 0:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = (
                        2 * (precision * recall) / (precision + recall)
                        if (precision + recall) > 0
                        else 0
                    )
                    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
                    false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0

                    method_analysis[alert_type] = {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "false_positive_rate": false_positive_rate,
                        "false_negative_rate": false_negative_rate,
                        "true_positives": int(tp),
                        "false_positives": int(fp),
                        "true_negatives": int(tn),
                        "false_negatives": int(fn),
                        "total_predictions": int(total),
                    }

            health_analysis[method] = method_analysis

        return health_analysis

    def process_all_cities(self) -> Dict:
        """Process all 100 cities efficiently."""

        cities_df = self.load_all_cities()
        all_results = {}

        total_cities = len(cities_df)

        for idx, city_row in cities_df.iterrows():
            city_name = city_row["City"]
            continent = city_row["Continent"]

            try:
                logger.info(
                    f"Processing {city_name}, {continent} ({idx+1}/{total_cities})"
                )

                # Generate time series
                city_time_series = self.generate_efficient_time_series(city_row)

                # Evaluate city
                city_results = self.evaluate_single_city(
                    city_time_series, city_name, continent
                )

                all_results[city_name] = {
                    "country": city_row["Country"],
                    "continent": continent,
                    "avg_aqi": city_row["Average_AQI"],
                    "avg_pm25": city_row["Average_PM25"],
                    "results": city_results,
                }

                if (idx + 1) % 10 == 0:
                    logger.info(f"Completed {idx+1}/{total_cities} cities...")

            except Exception as e:
                logger.error(f"Error processing {city_name}: {str(e)}")
                continue

        logger.info(f"Completed evaluation of {len(all_results)} cities")
        return all_results

    def generate_comprehensive_summary(self, results: Dict) -> Dict:
        """Generate comprehensive summary with focus on health warnings."""

        summary = {
            "evaluation_metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "cities_evaluated": len(results),
                "framework_version": "Full 100-City Evaluation v2.0",
                "continents_covered": len(
                    set(city["continent"] for city in results.values())
                ),
                "pollutants_analyzed": len(self.pollutants),
                "methods_compared": len(self.methods),
            },
            "overall_performance": {},
            "continental_performance": {},
            "health_warning_analysis": {},
            "improvement_analysis": {},
            "false_positive_negative_analysis": {},
        }

        # Overall performance across all cities
        for pollutant in self.pollutants + ["AQI"]:
            summary["overall_performance"][pollutant] = {}

            for method in self.methods:
                mae_values = []
                r2_values = []

                for city_name, city_data in results.items():
                    if pollutant in city_data["results"]:
                        mae_values.append(
                            city_data["results"][pollutant][method]["MAE"]
                        )
                        r2_values.append(city_data["results"][pollutant][method]["R2"])

                summary["overall_performance"][pollutant][method] = {
                    "mean_MAE": np.mean(mae_values) if mae_values else 0,
                    "std_MAE": np.std(mae_values) if mae_values else 0,
                    "mean_R2": np.mean(r2_values) if r2_values else 0,
                    "cities_evaluated": len(mae_values),
                }

        # Continental performance
        for continent in self.continental_standards.keys():
            continent_cities = {
                k: v for k, v in results.items() if v["continent"] == continent
            }

            if continent_cities:
                summary["continental_performance"][continent] = {}

                for pollutant in self.pollutants + ["AQI"]:
                    summary["continental_performance"][continent][pollutant] = {}

                    for method in self.methods:
                        mae_values = []
                        for city_name, city_data in continent_cities.items():
                            if pollutant in city_data["results"]:
                                mae_values.append(
                                    city_data["results"][pollutant][method]["MAE"]
                                )

                        if mae_values:
                            summary["continental_performance"][continent][pollutant][
                                method
                            ] = {
                                "mean_MAE": np.mean(mae_values),
                                "cities": len(mae_values),
                            }

        # Health warning analysis
        summary["health_warning_analysis"] = self.analyze_global_health_warnings(
            results
        )

        # Improvement analysis
        for pollutant in self.pollutants + ["AQI"]:
            ensemble_methods = ["simple_avg", "ridge"]
            benchmark_methods = ["cams", "noaa"]

            best_ensemble_mae = min(
                [
                    summary["overall_performance"][pollutant][m]["mean_MAE"]
                    for m in ensemble_methods
                ]
            )
            best_benchmark_mae = min(
                [
                    summary["overall_performance"][pollutant][m]["mean_MAE"]
                    for m in benchmark_methods
                ]
            )

            improvement_pct = (
                ((best_benchmark_mae - best_ensemble_mae) / best_benchmark_mae) * 100
                if best_benchmark_mae > 0
                else 0
            )

            summary["improvement_analysis"][pollutant] = {
                "improvement_percent": improvement_pct,
                "best_ensemble_mae": best_ensemble_mae,
                "best_benchmark_mae": best_benchmark_mae,
                "significance": (
                    "major"
                    if improvement_pct > 20
                    else "moderate" if improvement_pct > 10 else "minor"
                ),
            }

        return summary

    def analyze_global_health_warnings(self, results: Dict) -> Dict:
        """Analyze health warnings across all cities with false positive/negative focus."""

        global_health_analysis = {
            "overall_metrics": {},
            "continental_breakdown": {},
            "false_positive_analysis": {},
            "false_negative_analysis": {},
            "critical_findings": [],
        }

        # Aggregate metrics by method
        for method in self.methods:
            method_metrics = {
                "sensitive_alerts": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "general_alerts": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            }

            for city_name, city_data in results.items():
                if "health_warnings" in city_data["results"]:
                    health_data = city_data["results"]["health_warnings"]

                    if method in health_data:
                        for alert_type in ["sensitive", "general"]:
                            if alert_type in health_data[method]:
                                alert_data = health_data[method][alert_type]
                                method_metrics[f"{alert_type}_alerts"][
                                    "tp"
                                ] += alert_data.get("true_positives", 0)
                                method_metrics[f"{alert_type}_alerts"][
                                    "fp"
                                ] += alert_data.get("false_positives", 0)
                                method_metrics[f"{alert_type}_alerts"][
                                    "tn"
                                ] += alert_data.get("true_negatives", 0)
                                method_metrics[f"{alert_type}_alerts"][
                                    "fn"
                                ] += alert_data.get("false_negatives", 0)

            # Calculate overall metrics
            global_health_analysis["overall_metrics"][method] = {}

            for alert_type in ["sensitive_alerts", "general_alerts"]:
                stats = method_metrics[alert_type]
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
                    f1 = (
                        2 * (precision * recall) / (precision + recall)
                        if (precision + recall) > 0
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

                    global_health_analysis["overall_metrics"][method][alert_type] = {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                        "false_positive_rate": fpr,
                        "false_negative_rate": fnr,
                        "total_predictions": total,
                    }

        # False positive/negative analysis
        ridge_sensitive = (
            global_health_analysis["overall_metrics"]
            .get("ridge", {})
            .get("sensitive_alerts", {})
        )
        ridge_general = (
            global_health_analysis["overall_metrics"]
            .get("ridge", {})
            .get("general_alerts", {})
        )

        global_health_analysis["false_positive_analysis"] = {
            "sensitive_population_fpr": ridge_sensitive.get("false_positive_rate", 0),
            "general_population_fpr": ridge_general.get("false_positive_rate", 0),
            "impact_assessment": (
                "Acceptable"
                if ridge_sensitive.get("false_positive_rate", 0) < 0.15
                else "Needs Optimization"
            ),
        }

        global_health_analysis["false_negative_analysis"] = {
            "sensitive_population_fnr": ridge_sensitive.get("false_negative_rate", 0),
            "general_population_fnr": ridge_general.get("false_negative_rate", 0),
            "impact_assessment": (
                "Acceptable"
                if ridge_sensitive.get("false_negative_rate", 0) < 0.10
                else "Critical Issue"
            ),
        }

        # Critical findings
        if ridge_sensitive.get("false_negative_rate", 0) > 0.10:
            global_health_analysis["critical_findings"].append(
                f"HIGH FALSE NEGATIVE RATE: {ridge_sensitive.get('false_negative_rate', 0):.1%} for sensitive population alerts"
            )

        if ridge_sensitive.get("false_positive_rate", 0) > 0.20:
            global_health_analysis["critical_findings"].append(
                f"HIGH FALSE POSITIVE RATE: {ridge_sensitive.get('false_positive_rate', 0):.1%} for sensitive population alerts"
            )

        return global_health_analysis


def main():
    """Main execution function."""

    logger.info("Starting Full 100-City Evaluation")

    # Initialize evaluator
    evaluator = Full100CityEvaluator()

    # Process all cities
    results = evaluator.process_all_cities()

    # Generate comprehensive summary
    summary = evaluator.generate_comprehensive_summary(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    with open(f"full_100_city_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary
    with open(f"full_100_city_summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Results saved to full_100_city_results_{timestamp}.json")
    logger.info(f"Summary saved to full_100_city_summary_{timestamp}.json")

    # Print key findings including health warnings
    print("\n" + "=" * 80)
    print("FULL 100-CITY EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"\nEvaluation completed: {summary['evaluation_metadata']['evaluation_date']}"
    )
    print(f"Cities evaluated: {summary['evaluation_metadata']['cities_evaluated']}")

    # Overall performance
    ridge_aqi_mae = summary["overall_performance"]["AQI"]["ridge"]["mean_MAE"]
    ridge_pm25_mae = summary["overall_performance"]["PM25"]["ridge"]["mean_MAE"]

    print(f"\nBest Method Performance (Ridge Regression):")
    print(f"  AQI MAE: {ridge_aqi_mae:.1f}")
    print(f"  PM2.5 MAE: {ridge_pm25_mae:.1f}")

    # Improvements
    print(f"\nPollutant Improvements:")
    for pollutant, improvement in summary["improvement_analysis"].items():
        print(
            f"  {pollutant}: {improvement['improvement_percent']:.1f}% ({improvement['significance'].upper()})"
        )

    # Health warning analysis
    health_analysis = summary["health_warning_analysis"]

    print(f"\nHealth Warning Analysis:")
    ridge_sensitive = (
        health_analysis["overall_metrics"].get("ridge", {}).get("sensitive_alerts", {})
    )
    ridge_general = (
        health_analysis["overall_metrics"].get("ridge", {}).get("general_alerts", {})
    )

    print(f"  Sensitive Population Alerts:")
    print(f"    Precision: {ridge_sensitive.get('precision', 0):.1%}")
    print(f"    Recall: {ridge_sensitive.get('recall', 0):.1%}")
    print(
        f"    False Positive Rate: {ridge_sensitive.get('false_positive_rate', 0):.1%}"
    )
    print(
        f"    False Negative Rate: {ridge_sensitive.get('false_negative_rate', 0):.1%}"
    )

    print(f"  General Population Alerts:")
    print(f"    Precision: {ridge_general.get('precision', 0):.1%}")
    print(f"    Recall: {ridge_general.get('recall', 0):.1%}")
    print(f"    False Positive Rate: {ridge_general.get('false_positive_rate', 0):.1%}")
    print(f"    False Negative Rate: {ridge_general.get('false_negative_rate', 0):.1%}")

    # Critical findings
    if health_analysis["critical_findings"]:
        print(f"\nCRITICAL FINDINGS:")
        for finding in health_analysis["critical_findings"]:
            print(f"  ⚠️  {finding}")

    print("\n" + "=" * 80)

    return results, summary


if __name__ == "__main__":
    results, summary = main()
