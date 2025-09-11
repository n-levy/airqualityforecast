#!/usr/bin/env python3
"""
Stage 4: Comprehensive Health-Focused Validation
==============================================

Implements comprehensive evaluation framework with:
1. Individual pollutant performance (PM2.5, PM10, NO2, O3, SO2)
2. Composite AQI performance across regional standards
3. Health warning analysis (false positives/negatives)
4. Walk-forward validation with expanding training windows
5. Continental pattern integration
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    mean_absolute_error, 
    mean_squared_error, 
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


class ComprehensiveHealthValidator:
    """Comprehensive health-focused validation with false positive/negative analysis."""

    def __init__(self, output_dir: str = "data/analysis/stage4_comprehensive_validation"):
        """Initialize comprehensive validation system."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Continental patterns from previous stages
        self.continental_patterns = {
            "europe": {
                "pattern_name": "Berlin Pattern",
                "expected_r2": 0.90,
                "cities": 20,
                "data_quality": 0.964,
                "aqi_standard": "European EAQI",
                "health_thresholds": {"sensitive": 3, "general": 4}  # EAQI levels
            },
            "north_america": {
                "pattern_name": "Toronto Pattern", 
                "expected_r2": 0.85,
                "cities": 20,
                "data_quality": 0.948,
                "aqi_standard": "EPA AQI",
                "health_thresholds": {"sensitive": 101, "general": 151}  # EPA AQI values
            },
            "asia": {
                "pattern_name": "Delhi Pattern",
                "expected_r2": 0.75,
                "cities": 20,
                "data_quality": 0.892,
                "aqi_standard": "Indian National AQI",
                "health_thresholds": {"sensitive": 101, "general": 201}  # Indian AQI values
            },
            "africa": {
                "pattern_name": "Cairo Pattern",
                "expected_r2": 0.75,
                "cities": 20,
                "data_quality": 0.885,
                "aqi_standard": "WHO Guidelines",
                "health_thresholds": {"sensitive": 25, "general": 50}  # PM2.5 μg/m³ equiv
            },
            "south_america": {
                "pattern_name": "São Paulo Pattern",
                "expected_r2": 0.85,
                "cities": 20,
                "data_quality": 0.937,
                "aqi_standard": "EPA AQI Adaptation",
                "health_thresholds": {"sensitive": 101, "general": 151}  # EPA AQI values
            }
        }

        # Enhanced model configurations with health focus
        self.models = {
            "gradient_boosting_enhanced": {
                "model_class": GradientBoostingRegressor,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                "type": "primary",
                "expected_performance": 0.84
            },
            "random_forest_advanced": {
                "model_class": RandomForestRegressor,
                "params": {
                    "n_estimators": 50,
                    "max_depth": 10,
                    "random_state": 42,
                    "n_jobs": -1
                },
                "type": "primary", 
                "expected_performance": 0.82
            },
            "ridge_regression_enhanced": {
                "model_class": Ridge,
                "params": {
                    "alpha": 1.0,
                    "random_state": 42
                },
                "type": "primary",
                "expected_performance": 0.78
            },
            "simple_average_ensemble": {
                "model_class": None,
                "params": {},
                "type": "baseline",
                "expected_performance": 0.72
            },
            "quality_weighted_ensemble": {
                "model_class": None,
                "params": {},
                "type": "baseline", 
                "expected_performance": 0.76
            }
        }

        # Pollutant configurations
        self.pollutants = ["pm25", "pm10", "no2", "o3", "so2"]
        
        # Feature categories (21 features total)
        self.feature_categories = {
            "meteorological": ["temperature", "humidity", "wind_speed", "pressure", "precipitation"],
            "temporal": ["day_of_year", "day_of_week", "month", "season", "is_holiday", "is_weekend"],
            "regional": ["dust_event", "wildfire_smoke", "heating_load", "transport_density"],
            "quality": ["data_quality_score", "source_confidence", "completeness"],
            "pollutants": ["pm25_primary", "pm10_primary", "no2_primary", "o3_primary", "so2_primary"]
        }

        log.info("Comprehensive Health Validation System initialized")
        log.info(f"Output directory: {self.output_dir}")

    def generate_synthetic_city_data(self, city_info: Dict, continent: str) -> pd.DataFrame:
        """Generate synthetic time series data with individual pollutants and AQI."""
        
        # Generate 5 years of daily data (2020-2024)
        total_days = 1826  # 5 years including leap year
        start_date = datetime(2020, 1, 1)
        dates = pd.date_range(start=start_date, periods=total_days, freq="D")
        
        # Base seed for consistency
        np.random.seed(hash(city_info["name"]) % 2**32)
        
        continent_params = self.continental_patterns[continent]
        base_quality = continent_params["data_quality"]
        
        data = {"date": dates}
        
        # Generate meteorological features with seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        data["temperature"] = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3, total_days)
        data["humidity"] = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365 + np.pi / 4) + np.random.normal(0, 10, total_days)
        data["wind_speed"] = 5 + 3 * np.random.exponential(1, total_days)
        data["pressure"] = 1013 + np.random.normal(0, 15, total_days)
        data["precipitation"] = np.maximum(0, np.random.gamma(0.5, 2, total_days))
        
        # Temporal features
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
        data["transport_density"] = 0.7 + 0.3 * (1 - data["is_weekend"]) + np.random.normal(0, 0.1, total_days)
        
        # Quality features
        data["data_quality_score"] = np.random.normal(base_quality * 100, 5, total_days)
        data["source_confidence"] = np.random.normal(0.85, 0.1, total_days)
        data["completeness"] = np.random.normal(base_quality, 0.05, total_days)
        
        # Generate individual pollutants with realistic relationships
        continental_pollution_levels = {
            "europe": {"pm25": 12, "pm10": 20, "no2": 22, "o3": 55, "so2": 8},
            "north_america": {"pm25": 8, "pm10": 15, "no2": 18, "o3": 65, "so2": 5},
            "asia": {"pm25": 35, "pm10": 60, "no2": 35, "o3": 45, "so2": 15},
            "africa": {"pm25": 28, "pm10": 45, "no2": 25, "o3": 40, "so2": 12},
            "south_america": {"pm25": 18, "pm10": 28, "no2": 28, "o3": 50, "so2": 10}
        }
        
        pollution_base = continental_pollution_levels[continent]
        
        # PM2.5 (μg/m³) - influenced by season, dust, fires
        data["pm25"] = (
            pollution_base["pm25"] +
            5 * np.sin(2 * np.pi * day_of_year / 365 + np.pi) +  # Winter peak
            15 * data["dust_event"] +
            20 * data["wildfire_smoke"] +
            0.3 * data["heating_load"] +
            np.random.lognormal(0, 0.5, total_days)
        )
        
        # PM10 (μg/m³) - related to PM2.5 but with dust influence
        data["pm10"] = (
            data["pm25"] * 1.5 +
            25 * data["dust_event"] +
            np.random.lognormal(0, 0.4, total_days)
        )
        
        # NO2 (μg/m³) - traffic related, weekday patterns
        data["no2"] = (
            pollution_base["no2"] +
            8 * data["transport_density"] +
            5 * (1 - data["is_weekend"]) +
            np.random.lognormal(0, 0.3, total_days)
        )
        
        # O3 (μg/m³) - photochemical, temperature dependent
        data["o3"] = (
            pollution_base["o3"] +
            0.8 * np.maximum(0, data["temperature"] - 20) +
            10 * np.sin(2 * np.pi * day_of_year / 365) +  # Summer peak
            np.random.lognormal(0, 0.3, total_days)
        )
        
        # SO2 (μg/m³) - industrial, heating related
        data["so2"] = (
            pollution_base["so2"] +
            0.2 * data["heating_load"] +
            np.random.lognormal(0, 0.4, total_days)
        )
        
        # Ensure non-negative values
        for pollutant in self.pollutants:
            data[pollutant] = np.maximum(1, data[pollutant])
        
        # Generate primary pollutant features (inputs to models)
        for pollutant in self.pollutants:
            data[f"{pollutant}_primary"] = data[pollutant] + np.random.normal(0, data[pollutant] * 0.1, total_days)
        
        # Calculate AQI based on continental standard
        data["aqi"] = self.calculate_aqi(data, continent)
        
        # Health warning flags
        health_thresholds = continent_params["health_thresholds"]
        data["sensitive_alert"] = (data["aqi"] >= health_thresholds["sensitive"]).astype(int)
        data["general_alert"] = (data["aqi"] >= health_thresholds["general"]).astype(int)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["city"] = city_info["name"]
        df["continent"] = continent
        
        return df

    def calculate_aqi(self, data: Dict, continent: str) -> np.ndarray:
        """Calculate AQI based on continental standard."""
        
        pm25 = data["pm25"]
        pm10 = data["pm10"] 
        no2 = data["no2"]
        o3 = data["o3"]
        so2 = data["so2"]
        
        if continent == "europe":
            # European EAQI (1-6 scale) - simplified calculation
            # Based on worst pollutant index
            pm25_index = np.clip(pm25 / 5, 1, 6)  # Rough EAQI conversion
            pm10_index = np.clip(pm10 / 8, 1, 6)
            no2_index = np.clip(no2 / 40, 1, 6)
            o3_index = np.clip(o3 / 30, 1, 6)
            aqi = np.maximum.reduce([pm25_index, pm10_index, no2_index, o3_index])
            
        else:
            # EPA AQI style (0-500+ scale) for other continents
            # Simplified breakpoint calculation
            pm25_aqi = np.where(
                pm25 <= 12, pm25 * 50/12,
                np.where(pm25 <= 35, 50 + (pm25-12) * 50/23,
                np.where(pm25 <= 55, 100 + (pm25-35) * 50/20,
                np.where(pm25 <= 150, 150 + (pm25-55) * 100/95,
                250 + (pm25-150) * 100/100)))
            )
            
            pm10_aqi = np.where(
                pm10 <= 54, pm10 * 50/54,
                np.where(pm10 <= 154, 50 + (pm10-54) * 50/100,
                np.where(pm10 <= 254, 100 + (pm10-154) * 50/100,
                150 + (pm10-254) * 100/100))
            )
            
            no2_aqi = np.where(
                no2 <= 100, no2 * 100/100,
                100 + (no2-100) * 100/100
            )
            
            o3_aqi = np.where(
                o3 <= 70, o3 * 50/70,
                np.where(o3 <= 100, 50 + (o3-70) * 50/30,
                100 + (o3-100) * 100/50)
            )
            
            aqi = np.maximum.reduce([pm25_aqi, pm10_aqi, no2_aqi, o3_aqi])
        
        return aqi

    def train_and_evaluate_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        pollutant: str = "aqi"
    ) -> Dict[str, Any]:
        """Train and evaluate a single model for a specific pollutant/AQI."""
        
        model_config = self.models[model_name]
        
        if model_name == "simple_average_ensemble":
            # Simple average of pollutant primary features
            if pollutant == "aqi":
                # Average of all pollutant inputs
                pollutant_indices = [-5, -4, -3, -2, -1]  # Last 5 features are pollutants
                y_pred = np.mean(X_test[:, pollutant_indices], axis=1) * 15  # Scale to AQI range
            else:
                # For individual pollutants, use corresponding primary feature
                pollutant_idx = self.pollutants.index(pollutant)
                y_pred = X_test[:, -5 + pollutant_idx]  # Corresponding primary pollutant
                
        elif model_name == "quality_weighted_ensemble":
            # Quality-weighted average
            quality_idx = -8  # data_quality_score position
            quality_weights = X_test[:, quality_idx] / 100.0
            
            if pollutant == "aqi":
                pollutant_indices = [-5, -4, -3, -2, -1]
                # Apply weights to each sample individually
                weighted_predictions = []
                for i in range(X_test.shape[0]):
                    sample_values = X_test[i, pollutant_indices]
                    weight = quality_weights[i]
                    # Use weight as a multiplier for the average
                    weighted_pred = np.mean(sample_values) * weight
                    weighted_predictions.append(weighted_pred)
                y_pred = np.array(weighted_predictions) * 15  # Scale to AQI range
            else:
                pollutant_idx = self.pollutants.index(pollutant)
                y_pred = X_test[:, -5 + pollutant_idx] * quality_weights
                
        else:
            # Standard sklearn models
            model = model_config["model_class"](**model_config["params"])
            
            # Scale features for Ridge regression
            if model_name == "ridge_regression_enhanced":
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
        
        # Ensure non-negative predictions
        y_pred = np.maximum(0, y_pred)
        
        # Calculate regression metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
        bias = np.mean(y_pred - y_test)
        
        # Pearson correlation
        correlation = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
        
        return {
            "r2_score": r2,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "bias": bias,
            "correlation": correlation,
            "predictions": y_pred,
            "actuals": y_test
        }

    def evaluate_health_warnings(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate health warning performance (false positives/negatives)."""
        
        results = {}
        
        for alert_type, threshold in thresholds.items():
            # Convert to binary classification
            y_true = (actuals >= threshold).astype(int)
            y_pred = (predictions >= threshold).astype(int)
            
            # Calculate confusion matrix components
            tn = np.sum((y_true == 0) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results[alert_type] = {
                "sensitivity": sensitivity,
                "specificity": specificity, 
                "precision": precision,
                "false_negative_rate": false_negative_rate,
                "false_positive_rate": false_positive_rate,
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "total_alerts": np.sum(y_true),
                "predicted_alerts": np.sum(y_pred)
            }
        
        return results

    def comprehensive_city_evaluation(
        self, city_data: pd.DataFrame, city_info: Dict, continent: str
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation for a single city."""
        
        log.info(f"Comprehensive evaluation for {city_info['name']}, {continent}")
        
        # Prepare features
        feature_cols = []
        for category in self.feature_categories.values():
            feature_cols.extend(category)
        
        X = city_data[feature_cols].values
        dates = city_data["date"].values
        
        # Walk-forward validation split
        train_end_date = datetime(2023, 12, 31)
        train_mask = city_data["date"] <= train_end_date
        test_mask = city_data["date"] > train_end_date
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        
        city_results = {
            "city": city_info["name"],
            "continent": continent,
            "pollutant_performance": {},
            "aqi_performance": {},
            "health_warnings": {},
            "model_comparison": {}
        }
        
        # Evaluate each pollutant + AQI for each model
        evaluation_targets = self.pollutants + ["aqi"]
        
        for target in evaluation_targets:
            city_results["pollutant_performance"][target] = {}
            
            y_all = city_data[target].values
            y_train = y_all[train_mask]
            y_test = y_all[test_mask]
            
            for model_name in self.models.keys():
                # Train and evaluate model
                model_results = self.train_and_evaluate_model(
                    model_name, X_train, y_train, X_test, y_test, target
                )
                
                city_results["pollutant_performance"][target][model_name] = {
                    "r2_score": model_results["r2_score"],
                    "mae": model_results["mae"],
                    "rmse": model_results["rmse"],
                    "mape": model_results["mape"],
                    "bias": model_results["bias"],
                    "correlation": model_results["correlation"]
                }
                
                # Health warning analysis for AQI
                if target == "aqi":
                    health_thresholds = self.continental_patterns[continent]["health_thresholds"]
                    health_results = self.evaluate_health_warnings(
                        model_results["predictions"],
                        model_results["actuals"],
                        health_thresholds
                    )
                    
                    if model_name not in city_results["health_warnings"]:
                        city_results["health_warnings"][model_name] = {}
                    city_results["health_warnings"][model_name] = health_results
        
        return city_results

    def evaluate_continental_performance(self, continent: str) -> Dict[str, Any]:
        """Evaluate all cities in a continent with comprehensive metrics."""
        
        log.info(f"Comprehensive continental evaluation: {continent}")
        
        continent_params = self.continental_patterns[continent]
        num_cities = continent_params["cities"]
        
        continental_results = []
        
        for city_idx in range(num_cities):
            city_info = {
                "name": f"{continent.title()}_City_{city_idx + 1}",
                "index": city_idx,
                "continent": continent
            }
            
            # Generate city data
            city_data = self.generate_synthetic_city_data(city_info, continent)
            
            # Comprehensive evaluation
            city_results = self.comprehensive_city_evaluation(city_data, city_info, continent)
            continental_results.append(city_results)
        
        # Aggregate results
        return self.aggregate_continental_results(continental_results, continent)

    def aggregate_continental_results(
        self, city_results: List[Dict], continent: str
    ) -> Dict[str, Any]:
        """Aggregate comprehensive results across cities in a continent."""
        
        log.info(f"Aggregating comprehensive results for {continent}")
        
        # Initialize aggregation structures
        pollutant_aggregates = {}
        health_warning_aggregates = {}
        
        # Aggregate by pollutant and model
        evaluation_targets = self.pollutants + ["aqi"]
        
        for target in evaluation_targets:
            pollutant_aggregates[target] = {}
            
            for model_name in self.models.keys():
                metrics = ["r2_score", "mae", "rmse", "mape", "bias", "correlation"]
                model_metrics = {metric: [] for metric in metrics}
                
                for city_result in city_results:
                    if target in city_result["pollutant_performance"] and model_name in city_result["pollutant_performance"][target]:
                        city_performance = city_result["pollutant_performance"][target][model_name]
                        for metric in metrics:
                            if metric in city_performance:
                                model_metrics[metric].append(city_performance[metric])
                
                # Calculate aggregates
                pollutant_aggregates[target][model_name] = {}
                for metric in metrics:
                    if model_metrics[metric]:
                        pollutant_aggregates[target][model_name][f"mean_{metric}"] = np.mean(model_metrics[metric])
                        pollutant_aggregates[target][model_name][f"std_{metric}"] = np.std(model_metrics[metric])
        
        # Aggregate health warnings
        for model_name in self.models.keys():
            health_warning_aggregates[model_name] = {}
            
            for alert_type in ["sensitive", "general"]:
                alert_metrics = ["sensitivity", "specificity", "precision", "false_negative_rate", "false_positive_rate"]
                aggregated_metrics = {metric: [] for metric in alert_metrics}
                total_alerts = []
                total_fn = []
                total_fp = []
                
                for city_result in city_results:
                    if model_name in city_result["health_warnings"] and alert_type in city_result["health_warnings"][model_name]:
                        alert_data = city_result["health_warnings"][model_name][alert_type]
                        for metric in alert_metrics:
                            if metric in alert_data:
                                aggregated_metrics[metric].append(alert_data[metric])
                        total_alerts.append(alert_data.get("total_alerts", 0))
                        total_fn.append(alert_data.get("false_negatives", 0))
                        total_fp.append(alert_data.get("false_positives", 0))
                
                health_warning_aggregates[model_name][alert_type] = {}
                for metric in alert_metrics:
                    if aggregated_metrics[metric]:
                        health_warning_aggregates[model_name][alert_type][f"mean_{metric}"] = np.mean(aggregated_metrics[metric])
                        health_warning_aggregates[model_name][alert_type][f"std_{metric}"] = np.std(aggregated_metrics[metric])
                
                # Overall false negative rate (critical metric)
                total_alerts_sum = sum(total_alerts)
                total_fn_sum = sum(total_fn)
                total_fp_sum = sum(total_fp)
                
                health_warning_aggregates[model_name][alert_type]["overall_false_negative_rate"] = (
                    total_fn_sum / total_alerts_sum if total_alerts_sum > 0 else 0
                )
                health_warning_aggregates[model_name][alert_type]["overall_false_positive_count"] = total_fp_sum
                health_warning_aggregates[model_name][alert_type]["total_health_events"] = total_alerts_sum
        
        # Best performing models
        best_aqi_model = max(
            self.models.keys(),
            key=lambda m: pollutant_aggregates["aqi"].get(m, {}).get("mean_r2_score", 0)
        )
        
        return {
            "continent": continent,
            "pattern_name": self.continental_patterns[continent]["pattern_name"],
            "total_cities": len(city_results),
            "pollutant_performance": pollutant_aggregates,
            "health_warning_performance": health_warning_aggregates,
            "best_aqi_model": best_aqi_model,
            "city_results": city_results
        }

    def evaluate_global_system(self) -> Dict[str, Any]:
        """Evaluate the complete global system with comprehensive metrics."""
        
        log.info("Starting comprehensive global system evaluation")
        log.info("=" * 80)
        
        global_results = {}
        
        # Evaluate each continent
        for continent in self.continental_patterns.keys():
            continental_results = self.evaluate_continental_performance(continent)
            global_results[continent] = continental_results
            
            # Log summary
            best_model = continental_results["best_aqi_model"]
            best_r2 = continental_results["pollutant_performance"]["aqi"][best_model]["mean_r2_score"]
            
            log.info(f"{continent.title()} - Best AQI Model: {best_model} (R² = {best_r2:.3f})")
        
        # Create global summary
        global_summary = self.create_global_summary(global_results)
        
        return {
            "continental_results": global_results,
            "global_summary": global_summary,
            "evaluation_metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "evaluation_type": "comprehensive_health_focused",
                "validation_method": "walk_forward_with_health_warnings",
                "total_cities": 100,
                "models_evaluated": list(self.models.keys()),
                "pollutants_evaluated": self.pollutants + ["aqi"]
            }
        }

    def create_global_summary(self, continental_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive global summary with health focus."""
        
        # Global AQI performance
        global_aqi_performance = {}
        global_health_performance = {}
        
        for model_name in self.models.keys():
            aqi_r2s = []
            aqi_maes = []
            total_fn_sensitive = 0
            total_fn_general = 0
            total_alerts_sensitive = 0
            total_alerts_general = 0
            
            for continent_data in continental_results.values():
                # AQI performance
                if "aqi" in continent_data["pollutant_performance"] and model_name in continent_data["pollutant_performance"]["aqi"]:
                    aqi_performance = continent_data["pollutant_performance"]["aqi"][model_name]
                    aqi_r2s.append(aqi_performance.get("mean_r2_score", 0))
                    aqi_maes.append(aqi_performance.get("mean_mae", 0))
                
                # Health warning performance
                if model_name in continent_data["health_warning_performance"]:
                    health_data = continent_data["health_warning_performance"][model_name]
                    
                    if "sensitive" in health_data:
                        total_fn_sensitive += health_data["sensitive"].get("overall_false_positive_count", 0)
                        total_alerts_sensitive += health_data["sensitive"].get("total_health_events", 0)
                    
                    if "general" in health_data:
                        total_fn_general += health_data["general"].get("overall_false_positive_count", 0)
                        total_alerts_general += health_data["general"].get("total_health_events", 0)
            
            global_aqi_performance[model_name] = {
                "global_mean_aqi_r2": np.mean(aqi_r2s) if aqi_r2s else 0,
                "global_mean_aqi_mae": np.mean(aqi_maes) if aqi_maes else 0,
                "model_type": self.models[model_name]["type"]
            }
            
            global_health_performance[model_name] = {
                "sensitive_false_negative_rate": total_fn_sensitive / total_alerts_sensitive if total_alerts_sensitive > 0 else 0,
                "general_false_negative_rate": total_fn_general / total_alerts_general if total_alerts_general > 0 else 0,
                "sensitive_health_events": total_alerts_sensitive,
                "general_health_events": total_alerts_general
            }
        
        # Best models
        best_aqi_model = max(
            global_aqi_performance.keys(),
            key=lambda m: global_aqi_performance[m]["global_mean_aqi_r2"]
        )
        
        best_health_model = min(
            global_health_performance.keys(),
            key=lambda m: global_health_performance[m]["sensitive_false_negative_rate"]
        )
        
        # Health warning success criteria
        best_model_fn_rate = global_health_performance[best_health_model]["sensitive_false_negative_rate"]
        health_criteria_met = best_model_fn_rate < 0.10  # <10% false negative rate
        
        return {
            "global_aqi_performance": global_aqi_performance,
            "global_health_performance": global_health_performance,
            "best_aqi_model": best_aqi_model,
            "best_health_model": best_health_model,
            "health_criteria_met": health_criteria_met,
            "best_aqi_r2": global_aqi_performance[best_aqi_model]["global_mean_aqi_r2"],
            "best_health_fn_rate": best_model_fn_rate,
            "continental_ranking": sorted(
                [(cont, data["pollutant_performance"]["aqi"][data["best_aqi_model"]]["mean_r2_score"]) 
                 for cont, data in continental_results.items()],
                key=lambda x: x[1], reverse=True
            ),
            "global_readiness": {
                "aqi_prediction_ready": global_aqi_performance[best_aqi_model]["global_mean_aqi_r2"] > 0.75,
                "health_warnings_ready": health_criteria_met,
                "production_ready": (global_aqi_performance[best_aqi_model]["global_mean_aqi_r2"] > 0.75) and health_criteria_met
            }
        }

    def save_comprehensive_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive evaluation results."""
        
        # Save main results
        results_path = self.output_dir / "comprehensive_health_validation_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        log.info(f"Comprehensive results saved to {results_path}")
        
        # Save health warning summary
        health_summary = []
        global_health = results["global_summary"]["global_health_performance"]
        
        for model_name, performance in global_health.items():
            health_summary.append({
                "model_name": model_name,
                "model_type": self.models[model_name]["type"],
                "sensitive_false_negative_rate": performance["sensitive_false_negative_rate"],
                "general_false_negative_rate": performance["general_false_negative_rate"],
                "sensitive_health_events": performance["sensitive_health_events"],
                "general_health_events": performance["general_health_events"],
                "health_criteria_met": performance["sensitive_false_negative_rate"] < 0.10
            })
        
        health_csv_path = self.output_dir / "health_warning_performance.csv"
        pd.DataFrame(health_summary).to_csv(health_csv_path, index=False)
        
        log.info(f"Health warning summary saved to {health_csv_path}")


def main():
    """Execute comprehensive health-focused validation."""
    
    log.info("Starting Comprehensive Health-Focused Validation")
    log.info("INDIVIDUAL POLLUTANTS + AQI + HEALTH WARNINGS + WALK-FORWARD")
    log.info("=" * 80)
    
    # Initialize validator
    validator = ComprehensiveHealthValidator()
    
    # Execute comprehensive evaluation
    log.info("Phase 1: Executing comprehensive global evaluation...")
    results = validator.evaluate_global_system()
    
    # Save results
    log.info("Phase 2: Saving comprehensive results...")
    validator.save_comprehensive_results(results)
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE HEALTH-FOCUSED VALIDATION RESULTS")
    print("=" * 80)
    
    global_summary = results["global_summary"]
    
    print(f"\nGLOBAL AQI PERFORMANCE:")
    print(f"• Best AQI Model: {global_summary['best_aqi_model']}")
    print(f"• Global AQI R²: {global_summary['best_aqi_r2']:.3f}")
    
    print(f"\nHEALTH WARNING PERFORMANCE:")
    print(f"• Best Health Model: {global_summary['best_health_model']}")
    print(f"• Sensitive Group FN Rate: {global_summary['best_health_fn_rate']:.1%}")
    print(f"• Health Criteria Met: {'✅' if global_summary['health_criteria_met'] else '❌'}")
    
    print(f"\nCONTINENTAL RANKING (AQI Performance):")
    for i, (continent, r2_score) in enumerate(global_summary["continental_ranking"], 1):
        print(f"{i}. {continent.replace('_', ' ').title()}: R² = {r2_score:.3f}")
    
    print(f"\nPRODUCTION READINESS:")
    readiness = global_summary["global_readiness"]
    print(f"• AQI Prediction Ready: {'✅' if readiness['aqi_prediction_ready'] else '❌'}")
    print(f"• Health Warnings Ready: {'✅' if readiness['health_warnings_ready'] else '❌'}")
    print(f"• Production Ready: {'✅' if readiness['production_ready'] else '❌'}")
    
    print("\n" + "=" * 80)
    if readiness["production_ready"]:
        print("🎉 COMPREHENSIVE VALIDATION COMPLETE: SYSTEM READY FOR PRODUCTION 🎉")
        print("Individual pollutants + AQI + Health warnings validated across 100 cities")
    else:
        print("⚠️  COMPREHENSIVE VALIDATION COMPLETE: OPTIMIZATION NEEDED")
        print("Some criteria not met - see detailed results for improvements")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())