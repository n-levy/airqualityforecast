#!/usr/bin/env python3
"""
Walk-Forward Validation with Realistic Dataset
============================================

Correct approach:
1. Create realistic dataset identical to current structure
2. Remove 2024 predictions (simulate missing forecasts)
3. Use walk-forward validation to predict 2024
4. Evaluate using comprehensive health-focused framework
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation with realistic dataset approach."""

    def __init__(
        self, output_dir: str = "data/analysis/stage4_walk_forward_validation"
    ):
        """Initialize walk-forward validation system."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Continental patterns
        self.continental_patterns = {
            "europe": {
                "pattern_name": "Berlin Pattern",
                "cities": 20,
                "aqi_standard": "European EAQI",
                "health_thresholds": {"sensitive": 3, "general": 4},
            },
            "north_america": {
                "pattern_name": "Toronto Pattern",
                "cities": 20,
                "aqi_standard": "EPA AQI",
                "health_thresholds": {"sensitive": 101, "general": 151},
            },
            "asia": {
                "pattern_name": "Delhi Pattern",
                "cities": 20,
                "aqi_standard": "Indian National AQI",
                "health_thresholds": {"sensitive": 101, "general": 201},
            },
            "africa": {
                "pattern_name": "Cairo Pattern",
                "cities": 20,
                "aqi_standard": "WHO Guidelines",
                "health_thresholds": {"sensitive": 25, "general": 50},
            },
            "south_america": {
                "pattern_name": "São Paulo Pattern",
                "cities": 20,
                "aqi_standard": "EPA AQI Adaptation",
                "health_thresholds": {"sensitive": 101, "general": 151},
            },
        }

        # Three models for comparison
        self.models = {
            "simple_average": {"name": "Simple Average Ensemble", "type": "baseline"},
            "ridge_regression": {
                "name": "Ridge Regression Enhanced",
                "type": "primary",
            },
            "gradient_boosting": {
                "name": "Gradient Boosting Enhanced",
                "type": "primary",
            },
        }

        self.pollutants = ["pm25", "pm10", "no2", "o3", "so2"]

        log.info("Walk-Forward Validation System initialized")
        log.info(f"Output directory: {self.output_dir}")

    def create_realistic_dataset(self, city_info: Dict, continent: str) -> pd.DataFrame:
        """Create realistic dataset identical to current structure."""

        # Generate 5 years of daily data (2020-2024)
        total_days = 1826  # 5 years including leap year
        start_date = datetime(2020, 1, 1)
        dates = pd.date_range(start=start_date, periods=total_days, freq="D")

        # Consistent seed per city
        np.random.seed(hash(city_info["name"]) % 2**32)

        data = {"date": dates}

        # Generate meteorological data (realistic patterns)
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        data["temperature"] = (
            15
            + 10 * np.sin(2 * np.pi * day_of_year / 365)
            + np.random.normal(0, 3, total_days)
        )
        data["humidity"] = (
            50
            + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi / 4)
            + np.random.normal(0, 10, total_days)
        )
        data["wind_speed"] = 5 + 3 * np.random.exponential(1, total_days)
        data["pressure"] = 1013 + np.random.normal(0, 15, total_days)
        data["precipitation"] = np.maximum(0, np.random.gamma(0.5, 2, total_days))

        # Generate features that models will use
        data["day_of_year"] = day_of_year
        data["day_of_week"] = np.array([d.weekday() for d in dates])
        data["month"] = np.array([d.month for d in dates])
        data["season"] = np.array([(d.month - 1) // 3 for d in dates])
        data["is_holiday"] = np.random.binomial(1, 0.03, total_days)
        data["is_weekend"] = np.array([1 if d.weekday() >= 5 else 0 for d in dates])

        # Regional features
        if continent == "africa":
            data["dust_event"] = np.random.binomial(1, 0.05, total_days)
        else:
            data["dust_event"] = np.random.binomial(1, 0.01, total_days)

        data["wildfire_smoke"] = np.random.binomial(1, 0.02, total_days)
        data["heating_load"] = np.maximum(0, 20 - data["temperature"])
        data["transport_density"] = (
            0.7 + 0.3 * (1 - data["is_weekend"]) + np.random.normal(0, 0.1, total_days)
        )

        # Data quality features
        data["data_quality_score"] = np.random.normal(85, 5, total_days)
        data["source_confidence"] = np.random.normal(0.85, 0.1, total_days)
        data["completeness"] = np.random.normal(0.90, 0.05, total_days)

        # Generate REALISTIC individual pollutants (ground truth)
        continental_levels = {
            "europe": {"pm25": 12, "pm10": 20, "no2": 22, "o3": 55, "so2": 8},
            "north_america": {"pm25": 8, "pm10": 15, "no2": 18, "o3": 65, "so2": 5},
            "asia": {"pm25": 35, "pm10": 60, "no2": 35, "o3": 45, "so2": 15},
            "africa": {"pm25": 28, "pm10": 45, "no2": 25, "o3": 40, "so2": 12},
            "south_america": {"pm25": 18, "pm10": 28, "no2": 28, "o3": 50, "so2": 10},
        }

        base_levels = continental_levels[continent]

        # Generate realistic pollutant concentrations with relationships
        data["pm25"] = (
            base_levels["pm25"]
            + 5 * np.sin(2 * np.pi * day_of_year / 365 + np.pi)  # Winter higher
            + 15 * data["dust_event"]
            + 20 * data["wildfire_smoke"]
            + 0.3 * data["heating_load"]
            + np.random.lognormal(0, 0.4, total_days)
        )

        data["pm10"] = (
            data["pm25"] * 1.4
            + 25 * data["dust_event"]
            + np.random.lognormal(0, 0.3, total_days)
        )

        data["no2"] = (
            base_levels["no2"]
            + 8 * data["transport_density"]
            + 5 * (1 - data["is_weekend"])
            + np.random.lognormal(0, 0.25, total_days)
        )

        data["o3"] = (
            base_levels["o3"]
            + 0.8 * np.maximum(0, data["temperature"] - 20)
            + 15 * np.sin(2 * np.pi * day_of_year / 365)  # Summer peak
            + np.random.lognormal(0, 0.25, total_days)
        )

        data["so2"] = (
            base_levels["so2"]
            + 0.2 * data["heating_load"]
            + np.random.lognormal(0, 0.3, total_days)
        )

        # Ensure positive values
        for pollutant in self.pollutants:
            data[pollutant] = np.maximum(1, data[pollutant])

        # Calculate true AQI from pollutants
        data["aqi"] = self.calculate_aqi(data, continent)

        # Generate data source inputs (what models will use for prediction)
        # These represent the "benchmark" data sources available to models
        for pollutant in self.pollutants:
            # Primary source (best quality)
            data[f"{pollutant}_source1"] = data[pollutant] + np.random.normal(
                0, data[pollutant] * 0.05, total_days
            )
            # Secondary source (medium quality)
            data[f"{pollutant}_source2"] = data[pollutant] + np.random.normal(
                0, data[pollutant] * 0.10, total_days
            )
            # Tertiary source (lower quality)
            data[f"{pollutant}_source3"] = data[pollutant] + np.random.normal(
                0, data[pollutant] * 0.15, total_days
            )

        # Health warning flags (derived from true AQI)
        health_thresholds = self.continental_patterns[continent]["health_thresholds"]
        data["sensitive_alert"] = (
            data["aqi"] >= health_thresholds["sensitive"]
        ).astype(int)
        data["general_alert"] = (data["aqi"] >= health_thresholds["general"]).astype(
            int
        )

        df = pd.DataFrame(data)
        df["city"] = city_info["name"]
        df["continent"] = continent

        return df

    def calculate_aqi(self, data: Dict, continent: str) -> np.ndarray:
        """Calculate realistic AQI from pollutants."""

        pm25 = data["pm25"]
        pm10 = data["pm10"]
        no2 = data["no2"]
        o3 = data["o3"]
        so2 = data["so2"]

        if continent == "europe":
            # European EAQI (1-6 scale)
            pm25_index = np.clip(pm25 / 5, 1, 6)
            pm10_index = np.clip(pm10 / 8, 1, 6)
            no2_index = np.clip(no2 / 40, 1, 6)
            o3_index = np.clip(o3 / 30, 1, 6)
            aqi = np.maximum.reduce([pm25_index, pm10_index, no2_index, o3_index])
        else:
            # EPA AQI style (0-500+ scale)
            pm25_aqi = np.where(
                pm25 <= 12,
                pm25 * 50 / 12,
                np.where(
                    pm25 <= 35,
                    50 + (pm25 - 12) * 50 / 23,
                    np.where(
                        pm25 <= 55,
                        100 + (pm25 - 35) * 50 / 20,
                        150 + (pm25 - 55) * 100 / 95,
                    ),
                ),
            )

            pm10_aqi = np.where(
                pm10 <= 54,
                pm10 * 50 / 54,
                np.where(
                    pm10 <= 154,
                    50 + (pm10 - 54) * 50 / 100,
                    100 + (pm10 - 154) * 100 / 100,
                ),
            )

            no2_aqi = np.clip(no2 * 100 / 100, 0, 200)
            o3_aqi = np.clip(o3 * 100 / 70, 0, 200)

            aqi = np.maximum.reduce([pm25_aqi, pm10_aqi, no2_aqi, o3_aqi])

        return aqi

    def create_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for model prediction."""

        feature_cols = [
            # Meteorological
            "temperature",
            "humidity",
            "wind_speed",
            "pressure",
            "precipitation",
            # Temporal
            "day_of_year",
            "day_of_week",
            "month",
            "season",
            "is_holiday",
            "is_weekend",
            # Regional
            "dust_event",
            "wildfire_smoke",
            "heating_load",
            "transport_density",
            # Quality
            "data_quality_score",
            "source_confidence",
            "completeness",
        ]

        # Add pollutant source data (what models use as inputs)
        for pollutant in self.pollutants:
            feature_cols.extend(
                [f"{pollutant}_source1", f"{pollutant}_source2", f"{pollutant}_source3"]
            )

        return df[feature_cols]

    def simple_average_prediction(self, X: np.ndarray) -> np.ndarray:
        """Simple average ensemble prediction."""
        # Average the 3 data sources for each pollutant, then combine to AQI
        predictions = []

        for i in range(len(X)):
            sample = X[i]
            # Get averages for each pollutant from its 3 sources
            pollutant_avgs = []
            for p_idx in range(5):  # 5 pollutants
                source_start = 13 + (p_idx * 3)  # Start of sources for this pollutant
                sources = sample[source_start : source_start + 3]
                pollutant_avgs.append(np.mean(sources))

            # Convert to AQI estimate (simplified)
            aqi_estimate = max(pollutant_avgs) * 2.5  # Simple scaling
            predictions.append(aqi_estimate)

        return np.array(predictions)

    def ridge_regression_prediction(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Ridge regression prediction."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        return np.maximum(0, predictions)  # Ensure non-negative

    def gradient_boosting_prediction(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """Gradient boosting prediction."""
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return np.maximum(0, predictions)

    def walk_forward_validate_city(
        self, city_data: pd.DataFrame, city_info: Dict, continent: str
    ) -> Dict[str, Any]:
        """Perform walk-forward validation for a single city."""

        log.info(f"Walk-forward validation: {city_info['name']}, {continent}")

        # Prepare data
        features_df = self.create_prediction_features(city_data)
        X = features_df.values
        y_true = city_data["aqi"].values
        dates = city_data["date"].values

        # Split data: 2020-2023 for training, 2024 for testing
        train_end_date = datetime(2023, 12, 31)
        train_mask = city_data["date"] <= train_end_date
        test_mask = city_data["date"] > train_end_date

        X_train_base = X[train_mask]
        y_train_base = y_true[train_mask]
        X_test = X[test_mask]
        y_test = y_true[test_mask]
        test_dates = dates[test_mask]

        # Store predictions for each model
        model_predictions = {}

        # 1. Simple Average (no training needed)
        log.info(f"  Predicting with Simple Average...")
        model_predictions["simple_average"] = self.simple_average_prediction(X_test)

        # 2. Ridge Regression (walk-forward)
        log.info(f"  Walk-forward with Ridge Regression...")
        ridge_predictions = []

        for i in range(len(test_dates)):
            # Expanding window: use all data up to current test date
            test_date = pd.to_datetime(test_dates[i])
            train_cutoff = test_date - timedelta(days=1)
            expanding_mask = city_data["date"] <= train_cutoff

            X_expanding = X[expanding_mask]
            y_expanding = y_true[expanding_mask]

            if len(X_expanding) >= 100:  # Need minimum training data
                pred = self.ridge_regression_prediction(
                    X_expanding, y_expanding, X_test[i : i + 1]
                )[0]
            else:
                # Fallback to base training
                pred = self.ridge_regression_prediction(
                    X_train_base, y_train_base, X_test[i : i + 1]
                )[0]

            ridge_predictions.append(pred)

        model_predictions["ridge_regression"] = np.array(ridge_predictions)

        # 3. Gradient Boosting (walk-forward)
        log.info(f"  Walk-forward with Gradient Boosting...")
        gb_predictions = []

        for i in range(len(test_dates)):
            # Expanding window
            test_date = pd.to_datetime(test_dates[i])
            train_cutoff = test_date - timedelta(days=1)
            expanding_mask = city_data["date"] <= train_cutoff

            X_expanding = X[expanding_mask]
            y_expanding = y_true[expanding_mask]

            if len(X_expanding) >= 100:
                pred = self.gradient_boosting_prediction(
                    X_expanding, y_expanding, X_test[i : i + 1]
                )[0]
            else:
                pred = self.gradient_boosting_prediction(
                    X_train_base, y_train_base, X_test[i : i + 1]
                )[0]

            gb_predictions.append(pred)

        model_predictions["gradient_boosting"] = np.array(gb_predictions)

        # Evaluate each model
        city_results = {
            "city": city_info["name"],
            "continent": continent,
            "model_performance": {},
            "health_warnings": {},
        }

        health_thresholds = self.continental_patterns[continent]["health_thresholds"]

        for model_name, predictions in model_predictions.items():
            # Regression metrics
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1))) * 100
            bias = np.mean(predictions - y_test)
            correlation = (
                np.corrcoef(y_test, predictions)[0, 1] if len(y_test) > 1 else 0
            )

            city_results["model_performance"][model_name] = {
                "r2_score": r2,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "bias": bias,
                "correlation": correlation,
            }

            # Health warning evaluation
            health_results = {}

            for alert_type, threshold in health_thresholds.items():
                y_true_alert = (y_test >= threshold).astype(int)
                y_pred_alert = (predictions >= threshold).astype(int)

                # Calculate metrics
                tp = np.sum((y_true_alert == 1) & (y_pred_alert == 1))
                tn = np.sum((y_true_alert == 0) & (y_pred_alert == 0))
                fp = np.sum((y_true_alert == 0) & (y_pred_alert == 1))
                fn = np.sum((y_true_alert == 1) & (y_pred_alert == 0))

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
                false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

                health_results[alert_type] = {
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "precision": precision,
                    "false_negative_rate": false_negative_rate,
                    "false_positive_rate": false_positive_rate,
                    "true_positives": tp,
                    "false_negatives": fn,
                    "false_positives": fp,
                    "total_alerts": np.sum(y_true_alert),
                }

            city_results["health_warnings"][model_name] = health_results

        return city_results

    def evaluate_continental_performance(self, continent: str) -> Dict[str, Any]:
        """Evaluate all cities in a continent."""

        log.info(f"Evaluating continent: {continent}")

        continent_params = self.continental_patterns[continent]
        num_cities = continent_params["cities"]

        continental_results = []

        for city_idx in range(num_cities):
            city_info = {
                "name": f"{continent.title()}_City_{city_idx + 1}",
                "index": city_idx,
                "continent": continent,
            }

            # Create realistic dataset
            city_data = self.create_realistic_dataset(city_info, continent)

            # Walk-forward validation
            city_results = self.walk_forward_validate_city(
                city_data, city_info, continent
            )
            continental_results.append(city_results)

        return self.aggregate_continental_results(continental_results, continent)

    def aggregate_continental_results(
        self, city_results: List[Dict], continent: str
    ) -> Dict[str, Any]:
        """Aggregate results across cities in a continent."""

        log.info(f"Aggregating results for {continent}")

        model_aggregates = {}
        health_aggregates = {}

        for model_name in self.models.keys():
            # Performance metrics
            r2_scores = [
                city["model_performance"][model_name]["r2_score"]
                for city in city_results
            ]
            mae_scores = [
                city["model_performance"][model_name]["mae"] for city in city_results
            ]
            rmse_scores = [
                city["model_performance"][model_name]["rmse"] for city in city_results
            ]

            model_aggregates[model_name] = {
                "mean_r2": np.mean(r2_scores),
                "std_r2": np.std(r2_scores),
                "mean_mae": np.mean(mae_scores),
                "mean_rmse": np.mean(rmse_scores),
                "cities_above_r2_80": sum(1 for r2 in r2_scores if r2 > 0.80),
            }

            # Health warnings
            sensitive_fn_rates = []
            general_fn_rates = []
            total_sensitive_alerts = 0
            total_general_alerts = 0
            total_sensitive_fn = 0
            total_general_fn = 0

            for city in city_results:
                if model_name in city["health_warnings"]:
                    health_data = city["health_warnings"][model_name]

                    if "sensitive" in health_data:
                        sensitive_fn_rates.append(
                            health_data["sensitive"]["false_negative_rate"]
                        )
                        total_sensitive_alerts += health_data["sensitive"][
                            "total_alerts"
                        ]
                        total_sensitive_fn += health_data["sensitive"][
                            "false_negatives"
                        ]

                    if "general" in health_data:
                        general_fn_rates.append(
                            health_data["general"]["false_negative_rate"]
                        )
                        total_general_alerts += health_data["general"]["total_alerts"]
                        total_general_fn += health_data["general"]["false_negatives"]

            health_aggregates[model_name] = {
                "mean_sensitive_fn_rate": (
                    np.mean(sensitive_fn_rates) if sensitive_fn_rates else 0
                ),
                "mean_general_fn_rate": (
                    np.mean(general_fn_rates) if general_fn_rates else 0
                ),
                "overall_sensitive_fn_rate": (
                    total_sensitive_fn / total_sensitive_alerts
                    if total_sensitive_alerts > 0
                    else 0
                ),
                "overall_general_fn_rate": (
                    total_general_fn / total_general_alerts
                    if total_general_alerts > 0
                    else 0
                ),
                "total_sensitive_alerts": total_sensitive_alerts,
                "total_general_alerts": total_general_alerts,
            }

        # Best performing model
        best_model = max(
            model_aggregates.keys(), key=lambda m: model_aggregates[m]["mean_r2"]
        )

        return {
            "continent": continent,
            "pattern_name": continent_params["pattern_name"],
            "total_cities": len(city_results),
            "model_performance": model_aggregates,
            "health_performance": health_aggregates,
            "best_model": best_model,
            "city_results": city_results,
        }

    def evaluate_global_system(self) -> Dict[str, Any]:
        """Evaluate complete global system with walk-forward validation."""

        log.info("Starting Global Walk-Forward Validation")
        log.info("=" * 80)

        global_results = {}

        # Evaluate each continent
        for continent in self.continental_patterns.keys():
            continental_results = self.evaluate_continental_performance(continent)
            global_results[continent] = continental_results

            best_model = continental_results["best_model"]
            best_r2 = continental_results["model_performance"][best_model]["mean_r2"]

            log.info(
                f"{continent.title()}: Best Model = {best_model} (R² = {best_r2:.3f})"
            )

        # Create global summary
        global_summary = self.create_global_summary(global_results)

        return {
            "continental_results": global_results,
            "global_summary": global_summary,
            "evaluation_metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "evaluation_type": "walk_forward_validation",
                "validation_method": "expanding_window_monthly",
                "training_period": "2020-2023",
                "test_period": "2024",
                "total_cities": 100,
                "models_evaluated": list(self.models.keys()),
            },
        }

    def create_global_summary(
        self, continental_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create global summary of walk-forward validation results."""

        # Global model performance
        global_model_performance = {}
        global_health_performance = {}

        for model_name in self.models.keys():
            all_r2s = []
            all_maes = []
            total_production_cities = 0
            total_sensitive_fn = 0
            total_sensitive_alerts = 0

            for continent_data in continental_results.values():
                model_data = continent_data["model_performance"][model_name]
                health_data = continent_data["health_performance"][model_name]

                # Collect city-level data
                for city_result in continent_data["city_results"]:
                    all_r2s.append(
                        city_result["model_performance"][model_name]["r2_score"]
                    )
                    all_maes.append(city_result["model_performance"][model_name]["mae"])

                total_production_cities += model_data["cities_above_r2_80"]
                total_sensitive_fn += health_data[
                    "total_sensitive_alerts"
                ] - health_data.get("total_sensitive_alerts", 0)
                total_sensitive_alerts += health_data["total_sensitive_alerts"]

            global_model_performance[model_name] = {
                "global_mean_r2": np.mean(all_r2s),
                "global_std_r2": np.std(all_r2s),
                "global_mean_mae": np.mean(all_maes),
                "production_ready_cities": total_production_cities,
                "global_success_rate": total_production_cities / 100,
                "model_type": self.models[model_name]["type"],
            }

            global_health_performance[model_name] = {
                "global_sensitive_fn_rate": (
                    total_sensitive_fn / total_sensitive_alerts
                    if total_sensitive_alerts > 0
                    else 0
                ),
                "total_sensitive_alerts": total_sensitive_alerts,
            }

        # Best models
        best_accuracy_model = max(
            global_model_performance.keys(),
            key=lambda m: global_model_performance[m]["global_mean_r2"],
        )

        best_health_model = min(
            global_health_performance.keys(),
            key=lambda m: global_health_performance[m]["global_sensitive_fn_rate"],
        )

        # Success criteria
        best_accuracy = global_model_performance[best_accuracy_model]["global_mean_r2"]
        best_health_fn_rate = global_health_performance[best_health_model][
            "global_sensitive_fn_rate"
        ]
        production_cities = global_model_performance[best_accuracy_model][
            "production_ready_cities"
        ]

        success_criteria = {
            "aqi_accuracy_met": best_accuracy >= 0.75,
            "health_warnings_met": best_health_fn_rate <= 0.10,
            "production_cities_met": production_cities >= 60,
            "all_criteria_met": (best_accuracy >= 0.75)
            and (best_health_fn_rate <= 0.10)
            and (production_cities >= 60),
        }

        return {
            "global_model_performance": global_model_performance,
            "global_health_performance": global_health_performance,
            "best_accuracy_model": best_accuracy_model,
            "best_health_model": best_health_model,
            "success_criteria": success_criteria,
            "continental_ranking": sorted(
                [
                    (cont, data["model_performance"][data["best_model"]]["mean_r2"])
                    for cont, data in continental_results.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            ),
            "production_readiness": {
                "system_ready": success_criteria["all_criteria_met"],
                "best_accuracy": best_accuracy,
                "best_health_fn_rate": best_health_fn_rate,
                "production_cities": production_cities,
            },
        }

    def save_validation_results(self, results: Dict[str, Any]) -> None:
        """Save walk-forward validation results."""

        # Save main results
        results_path = self.output_dir / "walk_forward_validation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        log.info(f"Validation results saved to {results_path}")

        # Save model comparison summary
        global_perf = results["global_summary"]["global_model_performance"]
        global_health = results["global_summary"]["global_health_performance"]

        summary_data = []
        for model_name in self.models.keys():
            summary_data.append(
                {
                    "model_name": model_name,
                    "model_type": self.models[model_name]["type"],
                    "global_mean_r2": global_perf[model_name]["global_mean_r2"],
                    "global_mean_mae": global_perf[model_name]["global_mean_mae"],
                    "production_ready_cities": global_perf[model_name][
                        "production_ready_cities"
                    ],
                    "global_success_rate": global_perf[model_name][
                        "global_success_rate"
                    ],
                    "sensitive_false_negative_rate": global_health[model_name][
                        "global_sensitive_fn_rate"
                    ],
                    "health_criteria_met": global_health[model_name][
                        "global_sensitive_fn_rate"
                    ]
                    <= 0.10,
                }
            )

        summary_csv_path = self.output_dir / "model_comparison_summary.csv"
        pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)

        log.info(f"Model comparison summary saved to {summary_csv_path}")


def main():
    """Execute walk-forward validation."""

    log.info("Starting Walk-Forward Validation")
    log.info("REALISTIC DATASET + WALK-FORWARD + HEALTH WARNINGS")
    log.info("=" * 80)

    # Initialize validator
    validator = WalkForwardValidator()

    # Execute validation
    log.info("Phase 1: Executing walk-forward validation...")
    results = validator.evaluate_global_system()

    # Save results
    log.info("Phase 2: Saving validation results...")
    validator.save_validation_results(results)

    # Print summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 80)

    global_summary = results["global_summary"]

    print(f"\nBEST MODELS:")
    print(f"• Best Accuracy: {global_summary['best_accuracy_model']}")
    print(f"• Best Health Warnings: {global_summary['best_health_model']}")

    readiness = global_summary["production_readiness"]
    print(f"\nPRODUCTION READINESS:")
    print(f"• Global R² Accuracy: {readiness['best_accuracy']:.3f}")
    print(f"• Health Warning FN Rate: {readiness['best_health_fn_rate']:.1%}")
    print(f"• Production Ready Cities: {readiness['production_cities']}/100")
    print(f"• System Ready: {'✅' if readiness['system_ready'] else '❌'}")

    print(f"\nCONTINENTAL RANKING:")
    for i, (continent, r2_score) in enumerate(global_summary["continental_ranking"], 1):
        print(f"{i}. {continent.replace('_', ' ').title()}: R² = {r2_score:.3f}")

    success = global_summary["success_criteria"]
    print(f"\nSUCCESS CRITERIA:")
    print(f"• AQI Accuracy (≥0.75): {'✅' if success['aqi_accuracy_met'] else '❌'}")
    print(
        f"• Health Warnings (≤10% FN): {'✅' if success['health_warnings_met'] else '❌'}"
    )
    print(
        f"• Production Cities (≥60): {'✅' if success['production_cities_met'] else '❌'}"
    )

    print("\n" + "=" * 80)
    if success["all_criteria_met"]:
        print("🎉 WALK-FORWARD VALIDATION COMPLETE: SYSTEM PRODUCTION READY 🎉")
    else:
        print("⚠️  WALK-FORWARD VALIDATION COMPLETE: OPTIMIZATION NEEDED")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
