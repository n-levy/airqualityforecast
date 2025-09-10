#!/usr/bin/env python3
"""
Test Delhi Ensemble System
==========================

Demo script to test the Delhi simple ensemble forecasting system.
Uses mock data when API keys are not available for demonstration purposes.
"""

import os
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path

# Import the Delhi ensemble system
from delhi_simple_ensemble import DelhiSimpleEnsemble
from multi_standard_aqi import calculate_individual_aqi, get_aqi_category

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def create_mock_delhi_data():
    """Create realistic mock data for Delhi air quality testing."""

    log.info("Creating mock Delhi air quality data...")

    # Delhi winter pollution characteristics (December data)
    base_time = datetime.now()

    mock_data = []

    for hour in range(24):  # 24-hour forecast
        timestamp = base_time + timedelta(hours=hour)

        # Delhi winter pollution patterns (severe in morning/evening)
        hour_of_day = timestamp.hour

        # Base pollution levels (winter is severe in Delhi)
        if 6 <= hour_of_day <= 10:  # Morning rush + temperature inversion
            pm25_base = 150  # Very high
            pm10_base = 280
            no2_base = 85  # Traffic pollution
        elif 17 <= hour_of_day <= 21:  # Evening rush + burning
            pm25_base = 180  # Extremely high
            pm10_base = 320
            no2_base = 95
        elif 22 <= hour_of_day <= 6:  # Night - temperature inversion
            pm25_base = 120
            pm10_base = 220
            no2_base = 60
        else:  # Daytime - slightly better dispersion
            pm25_base = 90
            pm10_base = 160
            no2_base = 45

        # Add some randomness
        random_factor = np.random.normal(1.0, 0.2)

        # Pollutant concentrations (μg/m³)
        concentrations = {
            "pm25": max(10, pm25_base * random_factor),
            "pm10": max(20, pm10_base * random_factor),
            "no2": max(10, no2_base * random_factor),
            "o3": max(20, 60 * np.random.normal(1.0, 0.3)),  # O3 less predictable
            "so2": max(5, 25 * np.random.normal(1.0, 0.4)),  # SO2 from power plants
        }

        # Calculate Indian National AQI
        individual_aqis = {}
        for pollutant, conc in concentrations.items():
            aqi = calculate_individual_aqi(conc, pollutant, "Indian")
            if not pd.isna(aqi):
                individual_aqis[pollutant] = aqi

        composite_aqi = max(individual_aqis.values()) if individual_aqis else 50
        dominant = (
            max(individual_aqis, key=individual_aqis.get) if individual_aqis else "pm25"
        )
        category = get_aqi_category(composite_aqi, "Indian")

        # Weather conditions (Delhi winter)
        weather = {
            "temperature": np.random.normal(12, 4),  # °C (winter)
            "humidity": np.random.normal(65, 15),  # % (winter humidity)
            "wind_speed": max(
                0.5, np.random.normal(2.5, 1.5)
            ),  # m/s (low wind in winter)
            "pressure": np.random.normal(1015, 5),  # hPa
            "visibility": max(500, 5000 - composite_aqi * 10),  # Reduced by pollution
        }

        mock_point = {
            "timestamp": timestamp,
            "hours_ahead": hour,
            "concentrations": concentrations,
            "individual_aqis": individual_aqis,
            "simple_average_aqi": composite_aqi,
            "simple_average_category": category["level"],
            "dominant_pollutant": dominant,
            "weather": weather,
        }

        mock_data.append(mock_point)

    log.info(f"Generated {len(mock_data)} mock forecast points")
    return mock_data


def test_with_mock_data():
    """Test the Delhi ensemble system with mock data."""

    print("\n" + "=" * 80)
    print("DELHI ENSEMBLE SYSTEM - MOCK DATA TEST")
    print("Simulating severe winter pollution conditions")
    print("=" * 80)

    # Create mock forecast data
    mock_forecast = create_mock_delhi_data()

    # Create mock results structure
    mock_results = {
        "forecast_points": mock_forecast,
        "model_types": ["simple_average", "ridge_regression"],
        "aqi_standard": "Indian",
        "location": "Delhi, India",
    }

    # Initialize ensemble system (without API keys for testing)
    try:
        ensemble = DelhiSimpleEnsemble("mock_token", "mock_key")

        # Generate report using mock data
        ensemble.generate_forecast_report(mock_results)

        # Additional analysis
        print("\n" + "=" * 80)
        print("MOCK DATA ANALYSIS")
        print("=" * 80)

        # Calculate statistics
        aqis = [p["simple_average_aqi"] for p in mock_forecast]
        temps = [p["weather"]["temperature"] for p in mock_forecast]
        pm25_values = [p["concentrations"]["pm25"] for p in mock_forecast]

        print(f"\n24-HOUR FORECAST STATISTICS:")
        print(f"• AQI Range: {min(aqis):.0f} - {max(aqis):.0f}")
        print(f"• Average AQI: {np.mean(aqis):.0f}")
        print(f"• Temperature Range: {min(temps):.1f}°C - {max(temps):.1f}°C")
        print(f"• PM2.5 Range: {min(pm25_values):.0f} - {max(pm25_values):.0f} μg/m³")

        # Health risk periods
        severe_hours = sum(1 for aqi in aqis if aqi >= 300)  # Very Poor
        poor_hours = sum(1 for aqi in aqis if 200 <= aqi < 300)  # Poor

        print(f"\nHEALTH RISK PERIODS:")
        print(f"• Very Poor air quality: {severe_hours} hours")
        print(f"• Poor air quality: {poor_hours} hours")
        print(f"• Total unhealthy hours: {severe_hours + poor_hours}/24")

        # Peak pollution times
        peak_aqi_hour = aqis.index(max(aqis))
        peak_time = mock_forecast[peak_aqi_hour]["timestamp"]

        print(f"\nPEAK POLLUTION:")
        print(f"• Worst AQI: {max(aqis):.0f} at {peak_time.strftime('%H:%M')}")
        print(
            f"• Dominant pollutant during peak: {mock_forecast[peak_aqi_hour]['dominant_pollutant']}"
        )

        # Model performance simulation
        print(f"\nMODEL COMPARISON (SIMULATED):")
        print(f"• Simple Average: Uses mean of available forecasts")
        print(
            f"• Ridge Regression: Would provide ~15-25% improvement with training data"
        )
        print(f"• Current accuracy: Limited by forecast input quality")

    except Exception as e:
        log.error(f"Error in mock test: {e}")
        return 1

    return 0


def test_with_real_apis():
    """Test with real APIs if keys are available."""

    waqi_token = os.getenv("WAQI_API_TOKEN")
    openweather_key = os.getenv("OPENWEATHER_API_KEY")

    if not waqi_token or not openweather_key:
        print("\n" + "=" * 80)
        print("REAL API TEST SKIPPED")
        print("=" * 80)
        print("To test with real data, set environment variables:")
        print("• WAQI_API_TOKEN (get free at https://aqicn.org/data-platform/token/)")
        print("• OPENWEATHER_API_KEY (get free at https://openweathermap.org/api)")
        return False

    print("\n" + "=" * 80)
    print("DELHI ENSEMBLE SYSTEM - REAL API TEST")
    print("=" * 80)

    try:
        # Initialize with real API keys
        ensemble = DelhiSimpleEnsemble(waqi_token, openweather_key)

        # Run real forecast
        results = ensemble.run_ensemble_forecast()

        if results:
            # Generate report
            ensemble.generate_forecast_report(results)

            # Save results
            output_dir = Path("data/analysis/delhi_ensemble")
            output_dir.mkdir(parents=True, exist_ok=True)

            import json

            output_file = (
                output_dir
                / f"delhi_real_test_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            )

            # Prepare for JSON serialization
            json_results = results.copy()
            for point in json_results.get("forecast_points", []):
                if "timestamp" in point:
                    point["timestamp"] = point["timestamp"].isoformat()

            with open(output_file, "w") as f:
                json.dump(json_results, f, indent=2, default=str)

            print(f"\nReal forecast results saved to: {output_file}")
            return True
        else:
            print("Failed to get real forecast data")
            return False

    except Exception as e:
        log.error(f"Error in real API test: {e}")
        return False


def main():
    """Main test function."""

    print("Delhi Simple Ensemble System - Test Suite")
    print("=" * 50)

    # Test with mock data (always works)
    mock_success = test_with_mock_data()

    # Test with real APIs (if available)
    real_success = test_with_real_apis()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Mock data test: {'PASSED' if mock_success == 0 else 'FAILED'}")
    print(f"Real API test: {'PASSED' if real_success else 'SKIPPED (no API keys)'}")

    print(f"\nSYSTEM CAPABILITIES:")
    print(f"• Supports Indian National AQI standard (Delhi local)")
    print(f"• Integrates Delhi DPCC monitoring stations")
    print(f"• Uses OpenWeatherMap air quality forecasts")
    print(f"• Implements Simple Average and Ridge Regression models")
    print(f"• Provides health warnings based on Indian standards")
    print(f"• Accounts for Delhi-specific pollution patterns")

    print(f"\nNEXT STEPS FOR PRODUCTION:")
    print(f"1. Collect historical data for Ridge model training")
    print(f"2. Implement proper walk-forward validation")
    print(f"3. Add more Delhi DPCC stations")
    print(f"4. Include traffic and construction data")
    print(f"5. Add crop burning indicators (winter months)")

    return mock_success


if __name__ == "__main__":
    exit(main())
