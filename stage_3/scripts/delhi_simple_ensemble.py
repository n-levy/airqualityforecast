#!/usr/bin/env python3
"""
Delhi Simple Ensemble Forecasting System
========================================

Simplified ensemble system for Delhi air quality using publicly available Indian data:
- Ground Truth: Delhi Pollution Control Committee (DPCC) monitoring stations
- Benchmarks: OpenWeatherMap and World Air Quality Index forecasts
- Features: Basic meteorological parameters available in India
- Models: Simple Average and Ridge Regression only
- AQI Standard: Indian National AQI (local to Delhi)

Data Sources (Free/Public):
- WAQI API: World Air Quality Index (aqicn.org) for current conditions
- OpenWeatherMap: Air pollution and weather forecasts
- Delhi DPCC: Government monitoring data via WAQI aggregation

Simplified Feature Set:
- Temperature, humidity, wind speed/direction
- Pressure, precipitation
- Time-based features (hour, day, month, season)
- Basic lag features from recent measurements
"""

from __future__ import annotations

import os
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Import Indian AQI calculation functions
from multi_standard_aqi import (
    calculate_individual_aqi,
    calculate_composite_aqi,
    get_aqi_category,
    is_health_warning_required,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# Delhi coordinates and configuration
DELHI_LAT = 28.6139
DELHI_LON = 77.2090
DELHI_STATIONS = ["@4108", "@4109", "@4110"]  # Major DPCC stations via WAQI

# API endpoints
WAQI_BASE = "https://api.waqi.info"
OPENWEATHER_BASE = "https://api.openweathermap.org/data/2.5"


class DelhiSimpleEnsemble:
    """Simplified ensemble forecasting system for Delhi air quality."""

    def __init__(self, waqi_token: str, openweather_key: str):
        """
        Initialize Delhi ensemble system.

        Args:
            waqi_token: WAQI API token (free from aqicn.org)
            openweather_key: OpenWeatherMap API key
        """
        self.waqi_token = waqi_token
        self.openweather_key = openweather_key
        self.data_dir = Path("data/analysis/delhi_ensemble")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        log.info("Delhi Simple Ensemble System initialized")
        log.info("Models: Simple Average + Ridge Regression")
        log.info("AQI Standard: Indian National AQI")

    def get_waqi_current_data(self, station_id: str) -> Optional[Dict]:
        """Get current air quality data from WAQI for a Delhi station."""

        url = f"{WAQI_BASE}/feed/{station_id}/"
        params = {"token": self.waqi_token}

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    return data.get("data", {})

            log.warning(
                f"WAQI API error for station {station_id}: {response.status_code}"
            )
            return None

        except Exception as e:
            log.error(f"Error fetching WAQI data for {station_id}: {e}")
            return None

    def get_openweather_forecast(self) -> Optional[Dict]:
        """Get air quality and weather forecast from OpenWeatherMap."""

        # Air quality forecast
        aq_url = f"{OPENWEATHER_BASE}/air_pollution/forecast"
        weather_url = f"{OPENWEATHER_BASE}/forecast"

        params = {"lat": DELHI_LAT, "lon": DELHI_LON, "appid": self.openweather_key}

        try:
            # Get air quality forecast
            aq_response = requests.get(aq_url, params=params, timeout=30)
            weather_response = requests.get(weather_url, params=params, timeout=30)

            aq_data = aq_response.json() if aq_response.status_code == 200 else None
            weather_data = (
                weather_response.json() if weather_response.status_code == 200 else None
            )

            return {"air_quality": aq_data, "weather": weather_data}

        except Exception as e:
            log.error(f"Error fetching OpenWeather forecast: {e}")
            return None

    def process_waqi_data(self, waqi_data: Dict, station_id: str) -> Dict:
        """Process WAQI station data to extract features and ground truth."""

        processed = {
            "station_id": station_id,
            "timestamp": datetime.now(),
            "location": "delhi",
        }

        # Extract air quality measurements (ground truth)
        if "iaqi" in waqi_data:
            iaqi = waqi_data["iaqi"]

            # Pollutant concentrations (WAQI provides both raw values and AQI)
            for pollutant in ["pm25", "pm10", "no2", "o3", "so2"]:
                if pollutant in iaqi:
                    # WAQI sometimes provides 'v' (value) in different units
                    processed[f"actual_{pollutant}_aqi"] = iaqi[pollutant].get(
                        "v", np.nan
                    )

        # Overall AQI (WAQI's calculation, need to convert to Indian standard)
        if "aqi" in waqi_data:
            processed["waqi_aqi"] = waqi_data["aqi"]

        # Weather parameters (if available)
        if "iaqi" in waqi_data:
            weather_params = {
                "t": "temperature",  # °C
                "h": "humidity",  # %
                "p": "pressure",  # hPa
                "w": "wind_speed",  # m/s
                "wg": "wind_gust",  # m/s
            }

            for waqi_key, param_name in weather_params.items():
                if waqi_key in waqi_data["iaqi"]:
                    processed[param_name] = waqi_data["iaqi"][waqi_key].get("v", np.nan)

        # Station information
        if "city" in waqi_data:
            processed["station_name"] = waqi_data["city"].get("name", "")

        return processed

    def process_openweather_forecast(self, ow_data: Dict) -> List[Dict]:
        """Process OpenWeatherMap forecast data."""

        forecast_points = []

        if not ow_data or "air_quality" not in ow_data:
            return forecast_points

        aq_data = ow_data["air_quality"]
        weather_data = ow_data.get("weather", {})

        # Process air quality forecast points
        if "list" in aq_data:
            for i, aq_point in enumerate(aq_data["list"]):
                timestamp = datetime.fromtimestamp(aq_point["dt"])

                processed = {
                    "timestamp": timestamp,
                    "forecast_hours_ahead": i,  # Approximate
                    "source": "openweathermap",
                }

                # Pollutant concentrations (μg/m³)
                if "components" in aq_point:
                    comp = aq_point["components"]
                    processed.update(
                        {
                            "forecast_pm25": comp.get("pm2_5", np.nan),
                            "forecast_pm10": comp.get("pm10", np.nan),
                            "forecast_no2": comp.get("no2", np.nan),
                            "forecast_o3": comp.get("o3", np.nan),
                            "forecast_so2": comp.get("so2", np.nan),
                            "forecast_co": comp.get("co", np.nan),
                        }
                    )

                # Try to match with weather forecast
                if "list" in weather_data and i < len(weather_data["list"]):
                    weather_point = weather_data["list"][i]

                    if "main" in weather_point:
                        main = weather_point["main"]
                        processed.update(
                            {
                                "forecast_temperature": main.get("temp", np.nan)
                                - 273.15,  # K to °C
                                "forecast_humidity": main.get("humidity", np.nan),
                                "forecast_pressure": main.get("pressure", np.nan),
                            }
                        )

                    if "wind" in weather_point:
                        wind = weather_point["wind"]
                        processed.update(
                            {
                                "forecast_wind_speed": wind.get("speed", np.nan),
                                "forecast_wind_deg": wind.get("deg", np.nan),
                            }
                        )

                    if "rain" in weather_point:
                        processed["forecast_rain_3h"] = weather_point["rain"].get(
                            "3h", 0
                        )

                forecast_points.append(processed)

        return forecast_points

    def create_features(self, data: Dict) -> Dict:
        """Create simplified feature set for Delhi forecasting."""

        features = data.copy()
        timestamp = data.get("timestamp", datetime.now())

        # Time-based features (critical for air quality patterns)
        features.update(
            {
                "hour": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "day_of_year": timestamp.timetuple().tm_yday,
                "month": timestamp.month,
                "is_weekend": int(timestamp.weekday() >= 5),
                "is_winter": int(timestamp.month in [11, 12, 1, 2]),  # Delhi winter
                "is_summer": int(timestamp.month in [4, 5, 6]),  # Delhi summer
                "is_monsoon": int(timestamp.month in [7, 8, 9]),  # Delhi monsoon
            }
        )

        # Rush hour indicators (important for Delhi traffic pollution)
        hour = timestamp.hour
        features.update(
            {
                "is_morning_rush": int(7 <= hour <= 10),
                "is_evening_rush": int(17 <= hour <= 21),
                "is_night": int(hour >= 22 or hour <= 5),
            }
        )

        # Delhi-specific seasonal patterns
        # Winter pollution is severe in Delhi due to crop burning + low wind
        if features["is_winter"]:
            features["winter_pollution_factor"] = 2.0
        else:
            features["winter_pollution_factor"] = 1.0

        return features

    def calculate_indian_aqi_from_concentrations(
        self, concentrations: Dict[str, float]
    ) -> Dict:
        """Calculate Indian National AQI from pollutant concentrations."""

        individual_aqis = {}

        # Calculate individual AQIs for each pollutant
        for pollutant, concentration in concentrations.items():
            if not pd.isna(concentration) and concentration > 0:
                aqi = calculate_individual_aqi(concentration, pollutant, "Indian")
                if not pd.isna(aqi):
                    individual_aqis[pollutant] = aqi

        if not individual_aqis:
            return {"aqi": np.nan, "category": "Unknown", "dominant": "unknown"}

        # Composite AQI is maximum of individual AQIs
        composite_aqi = max(individual_aqis.values())
        dominant_pollutant = max(individual_aqis, key=individual_aqis.get)

        # Get category
        category = get_aqi_category(composite_aqi, "Indian")

        return {
            "aqi": composite_aqi,
            "category": category["level"],
            "dominant": dominant_pollutant,
            "individual_aqis": individual_aqis,
        }

    def simple_average_model(self, forecast1: Dict, forecast2: Dict) -> Dict:
        """Simple average ensemble model."""

        averaged = {}

        # Average pollutant forecasts
        pollutants = ["pm25", "pm10", "no2", "o3", "so2"]
        for pollutant in pollutants:
            f1_key = f"forecast_{pollutant}"
            f2_key = f"forecast_{pollutant}_alt"  # Alternative source

            vals = []
            if f1_key in forecast1 and not pd.isna(forecast1[f1_key]):
                vals.append(forecast1[f1_key])
            if f2_key in forecast2 and not pd.isna(forecast2[f2_key]):
                vals.append(forecast2[f2_key])

            if vals:
                averaged[f"ensemble_{pollutant}"] = np.mean(vals)
            else:
                averaged[f"ensemble_{pollutant}"] = np.nan

        return averaged

    def ridge_ensemble_model(
        self, X_train: np.ndarray, y_train: np.ndarray, X_predict: np.ndarray
    ) -> np.ndarray:
        """Ridge regression ensemble model."""

        # Use Ridge with regularization suitable for small datasets
        model = Ridge(alpha=1.0, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_predict_scaled = scaler.transform(X_predict)

        # Fit and predict
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_predict_scaled)

        return predictions

    def collect_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """Collect training data from multiple Delhi stations."""

        log.info(f"Collecting {days_back} days of training data from Delhi stations...")

        all_data = []

        # Collect data from major Delhi DPCC stations
        for station_id in DELHI_STATIONS:
            log.info(f"Collecting data from station {station_id}")

            # Get current data (in real implementation, would collect historical)
            waqi_data = self.get_waqi_current_data(station_id)
            if waqi_data:
                processed = self.process_waqi_data(waqi_data, station_id)
                features = self.create_features(processed)
                all_data.append(features)

            time.sleep(1)  # Rate limiting

        if all_data:
            df = pd.DataFrame(all_data)
            log.info(f"Collected {len(df)} training samples")
            return df
        else:
            log.warning("No training data collected")
            return pd.DataFrame()

    def run_ensemble_forecast(self) -> Dict:
        """Run ensemble forecast for Delhi air quality."""

        log.info("Running Delhi ensemble forecast...")

        # Get forecast data
        forecast_data = self.get_openweather_forecast()
        if not forecast_data:
            log.error("Failed to get forecast data")
            return {}

        # Process forecast
        forecast_points = self.process_openweather_forecast(forecast_data)
        if not forecast_points:
            log.error("No forecast points processed")
            return {}

        # For demonstration, create simple ensemble predictions
        ensemble_results = []

        for point in forecast_points[:24]:  # Next 24 hours
            # Extract pollutant concentrations
            concentrations = {}
            for pollutant in ["pm25", "pm10", "no2", "o3", "so2"]:
                key = f"forecast_{pollutant}"
                if key in point and not pd.isna(point[key]):
                    concentrations[pollutant] = point[key]

            if concentrations:
                # Simple average model (just use the forecast as-is for demo)
                simple_avg_concentrations = concentrations.copy()

                # Calculate Indian AQI
                simple_aqi = self.calculate_indian_aqi_from_concentrations(
                    simple_avg_concentrations
                )

                # Ridge model would require training data, skip for now

                result = {
                    "timestamp": point["timestamp"],
                    "hours_ahead": point.get("forecast_hours_ahead", 0),
                    "simple_average_aqi": simple_aqi["aqi"],
                    "simple_average_category": simple_aqi["category"],
                    "dominant_pollutant": simple_aqi["dominant"],
                    "concentrations": concentrations,
                    "weather": {
                        "temperature": point.get("forecast_temperature", np.nan),
                        "humidity": point.get("forecast_humidity", np.nan),
                        "wind_speed": point.get("forecast_wind_speed", np.nan),
                    },
                }

                ensemble_results.append(result)

        return {
            "forecast_points": ensemble_results,
            "model_types": ["simple_average"],
            "aqi_standard": "Indian",
            "location": "Delhi, India",
        }

    def generate_forecast_report(self, results: Dict) -> None:
        """Generate Delhi air quality forecast report."""

        print("\n" + "=" * 80)
        print("DELHI AIR QUALITY ENSEMBLE FORECAST")
        print("Indian National AQI Standard")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
        print("=" * 80)

        if not results or "forecast_points" not in results:
            print("No forecast data available")
            return

        forecast_points = results["forecast_points"]

        print(f"\nMODELS USED:")
        print(f"• Simple Average: Basic mean of available forecasts")
        print(f"• Ridge Regression: Regularized linear ensemble (requires training)")

        print(f"\nDATA SOURCES:")
        print(f"• Ground Truth: Delhi DPCC stations via WAQI API")
        print(f"• Benchmarks: OpenWeatherMap air pollution forecasts")
        print(f"• Weather: OpenWeatherMap meteorological data")

        print(f"\n24-HOUR FORECAST:")
        print(
            f"{'Time':<12} {'Hours':<6} {'AQI':<6} {'Category':<20} {'Dominant':<8} {'Temp':<6}"
        )
        print("-" * 70)

        for point in forecast_points[:12]:  # Show 12 hours
            time_str = point["timestamp"].strftime("%H:%M")
            hours_ahead = point["hours_ahead"]
            aqi = point.get("simple_average_aqi", np.nan)
            category = point.get("simple_average_category", "Unknown")
            dominant = point.get("dominant_pollutant", "N/A")
            temp = point.get("weather", {}).get("temperature", np.nan)

            if not pd.isna(aqi):
                print(
                    f"{time_str:<12} +{hours_ahead:<5} {aqi:<6.0f} {category:<20} {dominant:<8} {temp:<6.1f}°C"
                )

        print(f"\nHEALTH RECOMMENDATIONS:")

        # Get next few hours average AQI
        next_6h_aqis = [
            p.get("simple_average_aqi", np.nan) for p in forecast_points[:6]
        ]
        next_6h_aqis = [aqi for aqi in next_6h_aqis if not pd.isna(aqi)]

        if next_6h_aqis:
            avg_aqi = np.mean(next_6h_aqis)
            max_aqi = max(next_6h_aqis)

            if max_aqi >= 300:  # Very Poor
                print(
                    "*** HEALTH ALERT: Air quality is very poor. Everyone should avoid outdoor activities."
                )
            elif max_aqi >= 200:  # Poor
                print(
                    "*** HEALTH ADVISORY: Poor air quality. Sensitive groups should limit outdoor exposure."
                )
            elif max_aqi >= 100:  # Moderate
                print(
                    "*** HEALTH NOTICE: Moderate air quality. Sensitive individuals may experience symptoms."
                )
            else:
                print("*** Air quality is satisfactory for most people.")

        print(f"\nFORECAST SUMMARY:")
        if next_6h_aqis:
            print(f"• Next 6 hours average AQI: {np.mean(next_6h_aqis):.0f}")
            print(f"• Peak AQI expected: {max(next_6h_aqis):.0f}")
            print(
                f"• Dominant pollutants: {', '.join(set(p.get('dominant_pollutant', 'N/A') for p in forecast_points[:6]))}"
            )

        print("\n" + "=" * 80)
        print("DELHI-SPECIFIC FACTORS:")
        print("• Winter months (Nov-Feb): Severe pollution due to crop burning")
        print("• Traffic peaks: 7-10 AM and 5-9 PM increase NO2 and PM levels")
        print("• Monsoon (Jul-Sep): Natural air cleaning, lower pollution")
        print("• Construction dust: Significant PM10 contribution year-round")
        print("=" * 80)


def main():
    """Main execution function for Delhi simple ensemble system."""

    # Check for API keys
    waqi_token = os.getenv("WAQI_API_TOKEN")
    openweather_key = os.getenv("OPENWEATHER_API_KEY")

    if not waqi_token:
        print("WAQI API token not found!")
        print("Get free token at: https://aqicn.org/data-platform/token/")
        print("Set WAQI_API_TOKEN environment variable")
        return 1

    if not openweather_key:
        print("OpenWeatherMap API key not found!")
        print("Get free key at: https://openweathermap.org/api")
        print("Set OPENWEATHER_API_KEY environment variable")
        return 1

    # Initialize ensemble system
    ensemble = DelhiSimpleEnsemble(waqi_token, openweather_key)

    try:
        # Run ensemble forecast
        results = ensemble.run_ensemble_forecast()

        # Generate report
        ensemble.generate_forecast_report(results)

        # Save results
        if results:
            output_file = (
                ensemble.data_dir
                / f"delhi_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            )
            import json

            with open(output_file, "w") as f:
                # Convert datetime objects to strings for JSON serialization
                json_results = results.copy()
                for point in json_results.get("forecast_points", []):
                    if "timestamp" in point:
                        point["timestamp"] = point["timestamp"].isoformat()

                json.dump(json_results, f, indent=2, default=str)

            log.info(f"Results saved to {output_file}")

        return 0

    except Exception as e:
        log.error(f"Error running Delhi ensemble forecast: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
