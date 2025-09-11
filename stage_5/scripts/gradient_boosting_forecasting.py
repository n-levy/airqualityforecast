#!/usr/bin/env python3
"""
Gradient Boosting Forecasting with AQI Health Warning Analysis

Adds Gradient Boosting as a third forecasting model and performs comprehensive
evaluation including confusion matrices with false positives/negatives.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

def safe_print(text):
    """Print text safely handling Unicode characters."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for console encoding issues
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


class GradientBoostingForecaster:
    """Enhanced forecasting with Gradient Boosting model."""

    def __init__(self, data_path="."):
        self.data_path = Path(data_path)
        self.cities_df = None
        self.aqi_standards = None
        self.results = {
            "generation_time": datetime.now().isoformat(),
            "model_performance": {},
            "city_level_results": {},
            "enhanced_aqi_predictions": {},
            "dataset_info": {}
        }

    def load_data(self):
        """Load all necessary data files."""
        safe_print("Loading data files...")

        # Load cities features
        features_file = self.data_path / "comprehensive_tables" / "comprehensive_features_table.csv"
        if not features_file.exists():
            safe_print(f"Error: Features file not found at {features_file}")
            return False
        
        self.cities_df = pd.read_csv(features_file)

        # Load AQI standards
        aqi_file = self.data_path / "comprehensive_tables" / "comprehensive_aqi_standards_table.csv"
        if aqi_file.exists():
            self.aqi_standards = pd.read_csv(aqi_file)

        safe_print(f"Loaded {len(self.cities_df)} cities")
        return True

    def generate_time_series_data(self, city_name, days=365):
        """Generate realistic time series data for a city."""
        city_info = self.cities_df[self.cities_df["City"] == city_name]
        if city_info.empty:
            return None

        base_aqi = city_info.iloc[0]["Average_AQI"]
        base_pm25 = city_info.iloc[0]["Average_PM25"]
        
        # Generate dates
        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        data = []
        for i, date in enumerate(dates):
            # Add seasonal and trend components
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Annual cycle
            weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)     # Weekly cycle
            trend_factor = 1 + 0.001 * i  # Slight upward trend
            noise = np.random.normal(0, 0.2)
            
            factor = seasonal_factor * weekly_factor * trend_factor * (1 + noise)
            
            # Generate correlated pollutant values
            pm25 = max(1, base_pm25 * factor)
            aqi = self.pm25_to_aqi_estimate(pm25)
            
            # Add meteorological features
            temp = 20 + 15 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 3)
            humidity = 50 + 20 * np.sin(2 * np.pi * (i + 90) / 365) + np.random.normal(0, 5)
            wind_speed = max(0.1, 5 + np.random.normal(0, 2))
            pressure = 1013 + np.random.normal(0, 10)
            
            # Holiday effects
            is_weekend = date.weekday() >= 5
            holiday_factor = 0.8 if is_weekend else 1.0
            
            data.append({
                'date': date,
                'day_of_year': i + 1,
                'pm25': pm25 * holiday_factor,
                'aqi': aqi * holiday_factor,
                'temperature': temp,
                'humidity': max(0, min(100, humidity)),
                'wind_speed': wind_speed,
                'pressure': pressure,
                'is_weekend': is_weekend,
                'month': date.month,
                'season': (date.month - 1) // 3 + 1
            })
        
        return pd.DataFrame(data)

    def pm25_to_aqi_estimate(self, pm25):
        """Convert PM2.5 to AQI using EPA formula."""
        if pm25 <= 12:
            return pm25 * 50 / 12
        elif pm25 <= 35.4:
            return 50 + (pm25 - 12) * 50 / (35.4 - 12)
        elif pm25 <= 55.4:
            return 100 + (pm25 - 35.4) * 50 / (55.4 - 35.4)
        elif pm25 <= 150.4:
            return 150 + (pm25 - 55.4) * 50 / (150.4 - 55.4)
        else:
            return min(500, 200 + (pm25 - 150.4) * 100 / (250.4 - 150.4))

    def train_gradient_boosting_model(self, train_data):
        """Train Gradient Boosting model."""
        features = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                   'day_of_year', 'month', 'season', 'is_weekend']
        
        X = train_data[features].fillna(0)
        y = train_data['aqi'].fillna(0)
        
        # Use optimized parameters for better performance
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )
        
        # Fit scaler and model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        return model, scaler

    def train_ridge_model(self, train_data):
        """Train Ridge regression model."""
        features = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                   'day_of_year', 'month', 'season', 'is_weekend']
        
        X = train_data[features].fillna(0)
        y = train_data['aqi'].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler

    def generate_benchmark_forecasts(self, actual_values, model_type="cams"):
        """Generate benchmark forecast values."""
        if model_type == "cams":
            # CAMS-style: higher error, systematic bias
            noise_std = 15
            bias = 5
            error = np.random.normal(bias, noise_std, len(actual_values))
        else:  # NOAA-style
            # NOAA-style: moderate error, less bias
            noise_std = 8
            bias = -2
            error = np.random.normal(bias, noise_std, len(actual_values))
        
        forecasts = np.maximum(1, actual_values + error)
        return forecasts

    def simple_average_forecast(self, cams_forecast, noaa_forecast):
        """Calculate simple average of benchmarks."""
        return (cams_forecast + noaa_forecast) / 2

    def walk_forward_validation(self, city_name):
        """Perform walk-forward validation with all four models."""
        safe_print(f"Processing {city_name}...")
        
        # Generate time series data
        city_data = self.generate_time_series_data(city_name, days=365)
        if city_data is None:
            return None
        
        # Walk-forward validation setup
        train_start = 30  # Minimum training window
        predictions = []
        model_performance = {
            'cams_benchmark': {'predictions': [], 'actual': []},
            'noaa_benchmark': {'predictions': [], 'actual': []},
            'simple_average': {'predictions': [], 'actual': []},
            'ridge_regression': {'predictions': [], 'actual': []},
            'gradient_boosting': {'predictions': [], 'actual': []}
        }
        
        for day in range(train_start, len(city_data)):
            # Training data: all previous days
            train_data = city_data.iloc[:day]
            actual_today = city_data.iloc[day]['aqi']
            
            try:
                # Train models
                ridge_model, ridge_scaler = self.train_ridge_model(train_data)
                gb_model, gb_scaler = self.train_gradient_boosting_model(train_data)
                
                # Prepare today's features
                features = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                           'day_of_year', 'month', 'season', 'is_weekend']
                today_features = city_data.iloc[day][features].values.reshape(1, -1)
                
                # Make predictions
                ridge_pred = ridge_model.predict(ridge_scaler.transform(today_features))[0]
                gb_pred = gb_model.predict(gb_scaler.transform(today_features))[0]
                
                # Generate benchmark forecasts
                recent_actual = train_data['aqi'].tail(7).mean()  # Use recent average as base
                cams_pred = self.generate_benchmark_forecasts([recent_actual], "cams")[0]
                noaa_pred = self.generate_benchmark_forecasts([recent_actual], "noaa")[0]
                simple_avg_pred = self.simple_average_forecast(cams_pred, noaa_pred)
                
                # Store predictions
                model_performance['cams_benchmark']['predictions'].append(cams_pred)
                model_performance['noaa_benchmark']['predictions'].append(noaa_pred)
                model_performance['simple_average']['predictions'].append(simple_avg_pred)
                model_performance['ridge_regression']['predictions'].append(ridge_pred)
                model_performance['gradient_boosting']['predictions'].append(gb_pred)
                
                for model in model_performance:
                    model_performance[model]['actual'].append(actual_today)
                
                predictions.append({
                    'day': day,
                    'date': city_data.iloc[day]['date'].strftime('%Y-%m-%d'),
                    'actual_aqi': actual_today,
                    'cams_benchmark': cams_pred,
                    'noaa_benchmark': noaa_pred,
                    'simple_average': simple_avg_pred,
                    'ridge_regression': ridge_pred,
                    'gradient_boosting': gb_pred
                })
                
            except Exception as e:
                safe_print(f"Error on day {day} for {city_name}: {e}")
                continue
        
        # Calculate performance metrics
        city_metrics = {}
        for model_name, data in model_performance.items():
            if data['predictions']:
                actual = np.array(data['actual'])
                pred = np.array(data['predictions'])
                
                mae = mean_absolute_error(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))
                r2 = r2_score(actual, pred)
                mape = np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100
                
                city_metrics[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'predictions_count': len(pred)
                }
        
        return {
            'city_name': city_name,
            'predictions': predictions,
            'performance': city_metrics,
            'total_predictions': len(predictions)
        }

    def get_city_aqi_standard(self, city_name):
        """Get the appropriate AQI standard for a city."""
        city_info = self.cities_df[self.cities_df["City"] == city_name]
        if city_info.empty:
            return "EPA_AQI"

        continent = city_info.iloc[0]["Continent"]
        continent_to_aqi = {
            "North_America": "EPA_AQI",
            "Europe": "European_EAQI", 
            "Asia": "Indian_AQI",
            "Africa": "WHO_Guidelines",
            "South_America": "EPA_AQI"
        }
        return continent_to_aqi.get(continent, "EPA_AQI")

    def calculate_aqi_from_pollutants(self, pm25, aqi_standard="EPA_AQI"):
        """Calculate AQI from PM2.5 concentration using specified standard."""
        if pd.isna(pm25) or pm25 < 0:
            return np.nan

        if aqi_standard == "EPA_AQI":
            return self.pm25_to_aqi_estimate(pm25)
        elif aqi_standard == "European_EAQI":
            if pm25 <= 10: return 1
            elif pm25 <= 20: return 2
            elif pm25 <= 25: return 3
            elif pm25 <= 50: return 4
            elif pm25 <= 75: return 5
            else: return 6
        elif aqi_standard == "Indian_AQI":
            breakpoints = [
                (0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
                (91, 120, 201, 300), (121, 250, 301, 400), (251, 380, 401, 500)
            ]
            for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
                if bp_lo <= pm25 <= bp_hi:
                    aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
                    return round(aqi)
            return 500
        else:  # WHO Guidelines
            if pm25 <= 15: return 50
            elif pm25 <= 35: return 100
            elif pm25 <= 65: return 150
            elif pm25 <= 150: return 200
            else: return 300

    def get_aqi_category_and_warning(self, aqi_value, aqi_standard="EPA_AQI"):
        """Get AQI category and health warning level."""
        if pd.isna(aqi_value):
            return "Unknown", "none"

        if aqi_standard == "EPA_AQI":
            if aqi_value <= 50: return "Good", "none"
            elif aqi_value <= 100: return "Moderate", "none"
            elif aqi_value <= 150: return "Unhealthy for Sensitive Groups", "sensitive"
            elif aqi_value <= 200: return "Unhealthy", "general"
            elif aqi_value <= 300: return "Very Unhealthy", "general"
            else: return "Hazardous", "emergency"
        elif aqi_standard == "European_EAQI":
            categories = ["Very Good", "Good", "Medium", "Poor", "Very Poor", "Extremely Poor"]
            warnings = ["none", "none", "none", "sensitive", "general", "emergency"]
            idx = min(int(aqi_value) - 1, 5) if aqi_value >= 1 else 0
            return categories[idx], warnings[idx]
        else:
            if aqi_value <= 50: return "Good", "none"
            elif aqi_value <= 100: return "Moderate", "none"
            elif aqi_value <= 150: return "Unhealthy for Sensitive", "sensitive"
            else: return "Unhealthy", "general"

    def process_all_cities(self):
        """Process all cities with walk-forward validation."""
        safe_print("Starting comprehensive walk-forward validation with Gradient Boosting...")
        
        all_results = {}
        model_metrics = {
            'cams_benchmark': {'mae': [], 'rmse': [], 'r2': [], 'mape': []},
            'noaa_benchmark': {'mae': [], 'rmse': [], 'r2': [], 'mape': []},
            'simple_average': {'mae': [], 'rmse': [], 'r2': [], 'mape': []},
            'ridge_regression': {'mae': [], 'rmse': [], 'r2': [], 'mape': []},
            'gradient_boosting': {'mae': [], 'rmse': [], 'r2': [], 'mape': []}
        }
        
        total_predictions = 0
        successful_cities = 0
        
        for idx, city in enumerate(self.cities_df['City'].head(100)):  # Process all 100 cities
            try:
                result = self.walk_forward_validation(city)
                if result:
                    all_results[city] = result
                    total_predictions += result['total_predictions']
                    successful_cities += 1
                    
                    # Aggregate metrics
                    for model_name, metrics in result['performance'].items():
                        if model_name in model_metrics:
                            for metric in ['mae', 'rmse', 'r2', 'mape']:
                                if metric in metrics:
                                    model_metrics[model_name][metric].append(metrics[metric])
                    
                    if (idx + 1) % 10 == 0:
                        safe_print(f"Processed {idx + 1} cities, {successful_cities} successful")
                        
            except Exception as e:
                safe_print(f"Error processing {city}: {e}")
                continue
        
        # Calculate aggregate performance statistics
        performance_summary = {}
        for model_name, metrics in model_metrics.items():
            performance_summary[model_name] = {}
            for metric_name, values in metrics.items():
                if values:
                    performance_summary[model_name][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        self.results = {
            "generation_time": datetime.now().isoformat(),
            "model_performance": performance_summary,
            "city_level_results": all_results,
            "dataset_info": {
                "total_cities": successful_cities,
                "total_predictions": total_predictions,
                "models_evaluated": list(model_metrics.keys())
            }
        }
        
        safe_print(f"\nValidation completed successfully!")
        safe_print(f"Cities processed: {successful_cities}")
        safe_print(f"Total predictions: {total_predictions}")
        
        return self.results

    def generate_aqi_health_analysis(self):
        """Generate comprehensive AQI health warning analysis."""
        safe_print("Generating AQI health warning analysis with Gradient Boosting...")
        
        enhanced_results = {}
        
        for city_name, city_data in self.results["city_level_results"].items():
            aqi_standard = self.get_city_aqi_standard(city_name)
            
            enhanced_city_results = {
                "city_info": {
                    "aqi_standard": aqi_standard,
                    "continent": self.cities_df[self.cities_df["City"] == city_name].iloc[0]["Continent"] if not self.cities_df[self.cities_df["City"] == city_name].empty else "Unknown"
                },
                "aqi_predictions": [],
                "health_warnings": {
                    "ground_truth": [],
                    "cams_benchmark": [],
                    "noaa_benchmark": [], 
                    "simple_average": [],
                    "ridge_regression": [],
                    "gradient_boosting": []
                }
            }
            
            # Process each prediction
            for pred in city_data["predictions"]:
                actual_aqi = pred["actual_aqi"]
                
                # Calculate AQI using location-specific standard
                models_aqi = {
                    "ground_truth": self.calculate_aqi_from_pollutants(
                        self.aqi_to_pm25_estimate(actual_aqi), aqi_standard),
                    "cams_benchmark": self.calculate_aqi_from_pollutants(
                        self.aqi_to_pm25_estimate(pred["cams_benchmark"]), aqi_standard),
                    "noaa_benchmark": self.calculate_aqi_from_pollutants(
                        self.aqi_to_pm25_estimate(pred["noaa_benchmark"]), aqi_standard),
                    "simple_average": self.calculate_aqi_from_pollutants(
                        self.aqi_to_pm25_estimate(pred["simple_average"]), aqi_standard),
                    "ridge_regression": self.calculate_aqi_from_pollutants(
                        self.aqi_to_pm25_estimate(pred["ridge_regression"]), aqi_standard),
                    "gradient_boosting": self.calculate_aqi_from_pollutants(
                        self.aqi_to_pm25_estimate(pred["gradient_boosting"]), aqi_standard)
                }
                
                day_predictions = {"day": pred["day"], "date": pred["date"]}
                
                for model_name, aqi_val in models_aqi.items():
                    category, warning_level = self.get_aqi_category_and_warning(aqi_val, aqi_standard)
                    
                    day_predictions[f"{model_name}_aqi"] = aqi_val
                    day_predictions[f"{model_name}_category"] = category
                    day_predictions[f"{model_name}_warning"] = warning_level
                    
                    enhanced_city_results["health_warnings"][model_name].append(warning_level)
                
                enhanced_city_results["aqi_predictions"].append(day_predictions)
            
            enhanced_results[city_name] = enhanced_city_results
        
        return enhanced_results

    def aqi_to_pm25_estimate(self, aqi_value):
        """Rough conversion from AQI to PM2.5 for simulation purposes."""
        if aqi_value <= 50:
            return np.random.uniform(0, 12)
        elif aqi_value <= 100:
            return np.random.uniform(12, 35)
        elif aqi_value <= 150:
            return np.random.uniform(35, 55)
        elif aqi_value <= 200:
            return np.random.uniform(55, 150)
        else:
            return np.random.uniform(150, 250)

    def create_confusion_matrices(self, enhanced_results):
        """Create confusion matrices for all models including Gradient Boosting."""
        safe_print("Creating confusion matrices for health warning analysis...")
        
        confusion_results = {}
        
        for city_name, city_data in enhanced_results.items():
            city_confusion = {}
            ground_truth_warnings = city_data["health_warnings"]["ground_truth"]
            
            models = ["cams_benchmark", "noaa_benchmark", "simple_average", "ridge_regression", "gradient_boosting"]
            
            for model_name in models:
                model_warnings = city_data["health_warnings"][model_name]
                
                # Create confusion matrix
                tp = fp = tn = fn = 0
                
                for true_warning, pred_warning in zip(ground_truth_warnings, model_warnings):
                    # Binary classification: Warning (sensitive+) vs No Warning (none)
                    true_is_warning = true_warning in ["sensitive", "general", "emergency"]
                    pred_is_warning = pred_warning in ["sensitive", "general", "emergency"]
                    
                    if true_is_warning and pred_is_warning:
                        tp += 1
                    elif not true_is_warning and pred_is_warning:
                        fp += 1
                    elif not true_is_warning and not pred_is_warning:
                        tn += 1
                    else:
                        fn += 1
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                city_confusion[model_name] = {
                    "confusion_matrix": {
                        "true_positives": tp,
                        "false_positives": fp,
                        "true_negatives": tn,
                        "false_negatives": fn
                    },
                    "metrics": {
                        "precision": round(precision, 3),
                        "recall": round(recall, 3),
                        "specificity": round(specificity, 3),
                        "f1_score": round(f1_score, 3),
                        "false_negative_rate": round(fn / (tp + fn) if (tp + fn) > 0 else 0, 3),
                        "false_positive_rate": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 3)
                    }
                }
            
            confusion_results[city_name] = city_confusion
        
        return confusion_results

    def save_results(self):
        """Save comprehensive results including Gradient Boosting analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate enhanced AQI analysis
        enhanced_results = self.generate_aqi_health_analysis()
        confusion_results = self.create_confusion_matrices(enhanced_results)
        
        # Aggregate health warning metrics
        models = ["cams_benchmark", "noaa_benchmark", "simple_average", "ridge_regression", "gradient_boosting"]
        aggregated_metrics = {}
        
        for model in models:
            metrics_list = {
                "precision": [], "recall": [], "specificity": [], "f1_score": [],
                "false_negative_rate": [], "false_positive_rate": []
            }
            
            for city_data in confusion_results.values():
                if model in city_data:
                    for metric, value in city_data[model]["metrics"].items():
                        if metric in metrics_list:
                            metrics_list[metric].append(value)
            
            # Calculate aggregate statistics
            aggregated_metrics[model] = {}
            for metric, values in metrics_list.items():
                if values:
                    aggregated_metrics[model][metric] = {
                        "mean": round(np.mean(values), 3),
                        "std": round(np.std(values), 3),
                        "min": round(np.min(values), 3),
                        "max": round(np.max(values), 3)
                    }
        
        # Create comprehensive report
        comprehensive_report = {
            "generation_time": datetime.now().isoformat(),
            "analysis_summary": {
                "total_cities_analyzed": len(enhanced_results),
                "models_evaluated": models,
                "total_predictions_analyzed": sum(len(city_data["aqi_predictions"]) for city_data in enhanced_results.values())
            },
            "forecasting_performance": self.results["model_performance"],
            "health_warning_performance": aggregated_metrics,
            "city_level_confusion_matrices": confusion_results,
            "enhanced_aqi_predictions": enhanced_results,
            "key_findings": self._generate_key_findings(aggregated_metrics)
        }
        
        # Save detailed results
        results_file = self.data_path / "final_dataset" / f"gradient_boosting_comprehensive_analysis_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        # Create summary markdown
        self._create_summary_markdown(comprehensive_report, timestamp)
        
        safe_print(f"Gradient Boosting analysis results saved to: {results_file}")
        return results_file, comprehensive_report

    def _generate_key_findings(self, aggregated_metrics):
        """Generate key findings from the analysis."""
        findings = []
        
        # Find best model by false negative rate
        fn_rates = {}
        for model, metrics in aggregated_metrics.items():
            if "false_negative_rate" in metrics:
                fn_rates[model] = metrics["false_negative_rate"]["mean"]
        
        if fn_rates:
            best_model = min(fn_rates.keys(), key=lambda x: fn_rates[x])
            findings.append(f"Best health protection model: {best_model} (FN rate: {fn_rates[best_model]:.1%})")
            
            # Compare Gradient Boosting performance
            if "gradient_boosting" in fn_rates:
                gb_fn = fn_rates["gradient_boosting"]
                if "ridge_regression" in fn_rates:
                    ridge_fn = fn_rates["ridge_regression"]
                    if gb_fn < ridge_fn:
                        improvement = ((ridge_fn - gb_fn) / ridge_fn) * 100
                        findings.append(f"Gradient Boosting improves on Ridge Regression by {improvement:.1f}%")
                    elif ridge_fn < gb_fn:
                        difference = ((gb_fn - ridge_fn) / ridge_fn) * 100
                        findings.append(f"Ridge Regression outperforms Gradient Boosting by {difference:.1f}%")
                    else:
                        findings.append("Gradient Boosting and Ridge Regression show similar performance")
        
        return findings

    def _create_summary_markdown(self, report, timestamp):
        """Create comprehensive markdown summary."""
        md_content = f"""# Gradient Boosting Forecasting Analysis Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Analysis Overview
- **Cities Analyzed**: {report['analysis_summary']['total_cities_analyzed']}
- **Models Evaluated**: {len(report['analysis_summary']['models_evaluated'])} (including new Gradient Boosting)
- **Total Predictions**: {report['analysis_summary']['total_predictions_analyzed']:,}

## Model Performance Comparison (Health Warning Focus)

| Model | False Negative Rate | False Positive Rate | Precision | Recall | F1 Score |
|-------|-------------------|-------------------|-----------|--------|----------|
"""
        
        for model, metrics in report["health_warning_performance"].items():
            fn_rate = metrics.get("false_negative_rate", {}).get("mean", 0) * 100
            fp_rate = metrics.get("false_positive_rate", {}).get("mean", 0) * 100
            precision = metrics.get("precision", {}).get("mean", 0)
            recall = metrics.get("recall", {}).get("mean", 0)
            f1 = metrics.get("f1_score", {}).get("mean", 0)
            
            md_content += f"| {model} | {fn_rate:.1f}% | {fp_rate:.1f}% | {precision:.3f} | {recall:.3f} | {f1:.3f} |\\n"
        
        md_content += f"""

## Key Findings
"""
        for finding in report["key_findings"]:
            md_content += f"- {finding}\\n"
        
        md_content += f"""

## Forecasting Performance (MAE Comparison)

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|----- |
"""
        
        for model, metrics in report["forecasting_performance"].items():
            mae = metrics.get("mae", {}).get("mean", "N/A")
            rmse = metrics.get("rmse", {}).get("mean", "N/A")
            r2 = metrics.get("r2", {}).get("mean", "N/A")
            mape = metrics.get("mape", {}).get("mean", "N/A")
            
            if isinstance(mae, float):
                mae = f"{mae:.2f}"
            if isinstance(rmse, float):
                rmse = f"{rmse:.2f}"
            if isinstance(r2, float):
                r2 = f"{r2:.3f}"
            if isinstance(mape, float):
                mape = f"{mape:.1f}%"
                
            md_content += f"| {model} | {mae} | {rmse} | {r2} | {mape} |\\n"
        
        md_content += """

## Health Warning Categories
- **None**: No health warnings needed (AQI ≤ 100)
- **Sensitive**: Warnings for sensitive groups (AQI 101-150)
- **General**: Warnings for general population (AQI 151+)
- **Emergency**: Emergency health warnings (AQI 301+)

---
*Generated by Enhanced Forecasting System with Gradient Boosting*
"""
        
        md_file = self.data_path / "final_dataset" / f"GRADIENT_BOOSTING_ANALYSIS_SUMMARY_{timestamp}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        safe_print(f"Summary report saved to: {md_file}")


def main():
    """Main execution function."""
    safe_print("GRADIENT BOOSTING FORECASTING WITH AQI ANALYSIS")
    safe_print("=" * 60)
    
    forecaster = GradientBoostingForecaster()
    
    try:
        # Load data
        if not forecaster.load_data():
            safe_print("Failed to load data. Exiting.")
            return
        
        # Process all cities with walk-forward validation
        forecaster.process_all_cities()
        
        # Save comprehensive results
        result_file, report = forecaster.save_results()
        
        safe_print(f"\\nGradient Boosting analysis completed successfully!")
        safe_print(f"Results saved to: {result_file}")
        
        # Display key performance metrics
        if "health_warning_performance" in report:
            safe_print("\\n🏆 HEALTH WARNING PERFORMANCE SUMMARY:")
            models = ["gradient_boosting", "ridge_regression", "simple_average", "noaa_benchmark", "cams_benchmark"]
            
            for model in models:
                if model in report["health_warning_performance"]:
                    metrics = report["health_warning_performance"][model]
                    fn_rate = metrics.get("false_negative_rate", {}).get("mean", 0) * 100
                    fp_rate = metrics.get("false_positive_rate", {}).get("mean", 0) * 100
                    f1_score = metrics.get("f1_score", {}).get("mean", 0)
                    
                    safe_print(f"{model}: FN={fn_rate:.1f}%, FP={fp_rate:.1f}%, F1={f1_score:.3f}")
        
    except Exception as e:
        safe_print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()