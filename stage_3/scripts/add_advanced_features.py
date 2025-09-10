#!/usr/bin/env python3
"""
Advanced Feature Engineering for Air Quality Forecasting

Implements comprehensive feature engineering including:
1. Meteorological features (synthetic weather data)
2. Advanced temporal patterns
3. Cross-pollutant relationships
4. Spatial features
5. Forecast uncertainty features
6. External data integration (synthetic)

This builds upon the existing feature engineering to create a comprehensive
feature set for improved air quality forecasting performance.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple
import math

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# City coordinates (lat, lon) for spatial features
CITY_COORDS = {
    "Berlin": (52.5200, 13.4050),
    "Hamburg": (53.5511, 9.9937),
    "Munich": (48.1351, 11.5820),
}

# German school holidays and special events (simplified)
SCHOOL_HOLIDAYS = {
    "2025-08-01": "Summer Holiday",
    "2025-08-02": "Summer Holiday",
    "2025-08-03": "Summer Holiday",
    "2025-08-04": "Summer Holiday",
    "2025-08-05": "Summer Holiday",
    "2025-08-06": "Summer Holiday",
    "2025-08-07": "Summer Holiday",
    "2025-08-08": "Summer Holiday",
    "2025-08-09": "Summer Holiday",
    "2025-08-10": "Summer Holiday",
    "2025-08-25": "Back to School Prep",
    "2025-08-26": "Back to School Prep",
    "2025-08-27": "Back to School Prep",
    "2025-08-28": "Back to School Prep",
    "2025-08-29": "Back to School Prep",
    "2025-08-30": "Back to School Prep",
    "2025-08-31": "Back to School Prep",
}


def add_meteorological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic meteorological features that affect air quality.
    In production, these would come from weather forecast APIs.
    """
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])

    log.info("Adding meteorological features...")

    # Initialize met features
    met_features = [
        "wind_speed",
        "wind_direction",
        "temperature",
        "humidity",
        "precipitation",
        "boundary_layer_height",
        "solar_radiation",
        "pressure",
    ]

    for feature in met_features:
        df[feature] = np.nan

    # Generate realistic synthetic meteorological data
    np.random.seed(42)

    for i, row in df.iterrows():
        city = row["city"]
        date_dt = row["date_dt"]
        day_of_year = date_dt.timetuple().tm_yday

        # City-specific meteorological baselines
        city_met_factors = {
            "Berlin": {"temp_base": 22.0, "humid_base": 65, "wind_base": 3.2},
            "Hamburg": {
                "temp_base": 20.0,
                "humid_base": 75,
                "wind_base": 4.1,
            },  # Coastal
            "Munich": {
                "temp_base": 24.0,
                "humid_base": 60,
                "wind_base": 2.8,
            },  # Continental
        }

        factors = city_met_factors.get(city, city_met_factors["Berlin"])

        # Temperature (°C) - August summer pattern
        temp_seasonal = factors["temp_base"] + 3 * np.sin(
            (day_of_year - 213) * 2 * np.pi / 365
        )  # Peak around Aug 1
        temp_daily_var = np.random.normal(0, 2.5)
        df.loc[i, "temperature"] = max(15, temp_seasonal + temp_daily_var)

        # Humidity (%) - inversely related to temperature
        humid_base = factors["humid_base"] - (df.loc[i, "temperature"] - 20) * 1.5
        df.loc[i, "humidity"] = max(30, min(95, humid_base + np.random.normal(0, 8)))

        # Wind speed (m/s) - affects dispersion
        wind_seasonal = factors["wind_base"] * (
            1 + 0.1 * np.cos((day_of_year - 180) * 2 * np.pi / 365)
        )
        df.loc[i, "wind_speed"] = max(0.5, wind_seasonal + np.random.normal(0, 1.2))

        # Wind direction (degrees) - prevailing westerly with variation
        prevailing = 270  # Westerly for Germany
        df.loc[i, "wind_direction"] = (prevailing + np.random.normal(0, 45)) % 360

        # Precipitation (mm) - summer thunderstorm pattern
        precip_prob = 0.15 + 0.1 * np.sin((day_of_year - 200) * 2 * np.pi / 365)
        if np.random.random() < precip_prob:
            df.loc[i, "precipitation"] = np.random.exponential(
                5.0
            )  # Exponential distribution
        else:
            df.loc[i, "precipitation"] = 0.0

        # Boundary layer height (m) - crucial for dispersion
        temp_effect = (df.loc[i, "temperature"] - 15) * 20  # Higher temp = higher BLH
        wind_effect = df.loc[i, "wind_speed"] * 50  # Higher wind = more mixing
        df.loc[i, "boundary_layer_height"] = max(
            200, 800 + temp_effect + wind_effect + np.random.normal(0, 150)
        )

        # Solar radiation (W/m²) - for photochemical reactions
        # Simplified model: depends on day of year and weather
        max_solar = (
            800 * np.sin((day_of_year - 80) * np.pi / 365)
            if day_of_year > 80 and day_of_year < 300
            else 0
        )
        cloud_factor = 1 - (df.loc[i, "humidity"] - 40) / 100  # High humidity = clouds
        df.loc[i, "solar_radiation"] = max(
            0, max_solar * max(0.1, cloud_factor) + np.random.normal(0, 50)
        )

        # Pressure (hPa) - affects stagnation
        df.loc[i, "pressure"] = 1013 + np.random.normal(
            0, 8
        )  # Standard atmosphere with variation

    # Derived meteorological features
    df["temp_squared"] = df["temperature"] ** 2  # Non-linear temperature effects
    df["wind_u"] = df["wind_speed"] * np.cos(
        np.radians(df["wind_direction"])
    )  # U component
    df["wind_v"] = df["wind_speed"] * np.sin(
        np.radians(df["wind_direction"])
    )  # V component
    df["ventilation_index"] = (
        df["wind_speed"] * df["boundary_layer_height"]
    )  # Dispersion capacity
    df["stability_indicator"] = (
        df["temperature"] / df["boundary_layer_height"]
    )  # Atmospheric stability

    log.info(f"Added {len(met_features) + 5} meteorological features")
    return df


def add_advanced_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced temporal pattern features beyond basic calendar features.
    """
    df = df.copy()
    if "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"])

    log.info("Adding advanced temporal features...")

    # Hour of day effects (simulate rush hours)
    # Since we have daily data, simulate typical daily patterns
    df["simulated_morning_rush"] = 0.8 + 0.2 * np.random.random(
        len(df)
    )  # Normalize 0-1
    df["simulated_evening_rush"] = 0.7 + 0.3 * np.random.random(len(df))
    df["simulated_midday_low"] = 0.3 + 0.2 * np.random.random(len(df))

    # Multi-day persistence features
    df["day_of_month"] = df["date_dt"].dt.day
    df["week_of_year"] = df["date_dt"].dt.isocalendar().week

    # Seasonal progression within August
    df["august_progression"] = (
        df["date_dt"].dt.day - 1
    ) / 30.0  # 0 to 1 through August
    df["late_summer_indicator"] = (df["august_progression"] > 0.6).astype(int)

    # Holiday proximity features
    df["school_holiday"] = df["date"].isin(SCHOOL_HOLIDAYS.keys()).astype(int)
    df["holiday_type"] = df["date"].map(SCHOOL_HOLIDAYS)

    # Distance to nearest weekend
    df["days_to_weekend"] = df["date_dt"].apply(
        lambda x: min((5 - x.weekday()) % 7, (6 - x.weekday()) % 7)
    )
    df["days_from_weekend"] = df["date_dt"].apply(
        lambda x: min(x.weekday(), (x.weekday() - 6) % 7)
    )

    # Week position (beginning, middle, end)
    df["week_position"] = df["date_dt"].dt.dayofweek.map(
        {
            0: "start",
            1: "start",
            2: "middle",
            3: "middle",
            4: "end",
            5: "weekend",
            6: "weekend",
        }
    )

    # Cyclical encoding for better ML handling
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["date_dt"].dt.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["date_dt"].dt.dayofweek / 7)
    df["day_of_month_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_of_month_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    log.info("Added 15 advanced temporal features")
    return df


def add_cross_pollutant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features based on cross-pollutant relationships and chemical interactions.
    """
    df = df.copy()

    log.info("Adding cross-pollutant relationship features...")

    providers = ["cams", "noaa_gefs_aerosol"]
    pollutants = ["pm25", "pm10", "no2", "o3"]

    for provider in providers:
        # Pollutant ratios (source indicators)
        pm25_col = f"forecast_{provider}_pm25"
        pm10_col = f"forecast_{provider}_pm10"
        no2_col = f"forecast_{provider}_no2"
        o3_col = f"forecast_{provider}_o3"

        if all(col in df.columns for col in [pm25_col, pm10_col, no2_col, o3_col]):
            # PM ratios
            df[f"{provider}_pm25_pm10_ratio"] = df[pm25_col] / (df[pm10_col] + 0.1)
            df[f"{provider}_coarse_pm"] = (
                df[pm10_col] - df[pm25_col]
            )  # Coarse particle fraction

            # NOx-O3 relationships (photochemical indicators)
            df[f"{provider}_no2_o3_ratio"] = df[no2_col] / (df[o3_col] + 1.0)
            df[f"{provider}_photochem_potential"] = (
                df[no2_col] * df["solar_radiation"] / 1000
                if "solar_radiation" in df.columns
                else df[no2_col]
            )

            # Secondary aerosol indicators
            df[f"{provider}_secondary_aerosol_proxy"] = (
                df[no2_col] + df[o3_col]
            ) / 2  # Precursor average

            # Total pollutant load
            df[f"{provider}_total_load"] = (
                df[pm25_col] + df[pm10_col] + df[no2_col] + df[o3_col] / 10
            )  # Scale O3

            # Air mass characteristics
            df[f"{provider}_fresh_emissions"] = df[no2_col] / (
                df[o3_col] + 5
            )  # High NO2, low O3 = fresh
            df[f"{provider}_aged_air"] = df[o3_col] / (
                df[no2_col] + 5
            )  # High O3, low NO2 = aged

    # Cross-provider pollutant relationships
    for pollutant in pollutants:
        cams_col = f"forecast_cams_{pollutant}"
        noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

        if cams_col in df.columns and noaa_col in df.columns:
            # Model agreement on chemical relationships
            df[f"cross_provider_{pollutant}_agreement"] = 1 / (
                1 + abs(df[cams_col] - df[noaa_col])
            )

    # Weather-chemistry interactions
    if "temperature" in df.columns and "solar_radiation" in df.columns:
        # Temperature-dependent chemistry
        df["temp_o3_interaction"] = (
            df["temperature"] * df.get("forecast_cams_o3", 0) / 100
        )
        df["solar_no2_interaction"] = (
            df["solar_radiation"] * df.get("forecast_cams_no2", 0) / 1000
        )

        # Biogenic emission proxy (temperature + solar)
        df["biogenic_emission_proxy"] = (
            (df["temperature"] - 15) * df["solar_radiation"] / 1000
        )

    log.info("Added cross-pollutant relationship features")
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spatial features based on inter-city relationships and transport.
    """
    df = df.copy()
    df_sorted = df.sort_values(["date", "city"]).copy()

    log.info("Adding spatial features...")

    # Calculate inter-city distances
    cities = list(CITY_COORDS.keys())
    city_distances = {}

    for i, city1 in enumerate(cities):
        for j, city2 in enumerate(cities):
            if i != j:
                lat1, lon1 = CITY_COORDS[city1]
                lat2, lon2 = CITY_COORDS[city2]

                # Haversine distance
                R = 6371  # Earth radius in km
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = (
                    math.sin(dlat / 2) ** 2
                    + math.cos(math.radians(lat1))
                    * math.cos(math.radians(lat2))
                    * math.sin(dlon / 2) ** 2
                )
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                distance = R * c

                city_distances[(city1, city2)] = distance

    # Add spatial features
    pollutants = ["pm25", "pm10", "no2", "o3"]
    providers = ["cams", "noaa_gefs_aerosol"]

    for date in df["date"].unique():
        date_mask = df_sorted["date"] == date
        date_data = df_sorted[date_mask].copy()

        if len(date_data) < 2:
            continue

        for i, row in date_data.iterrows():
            city = row["city"]
            other_cities = [c for c in cities if c != city]

            for provider in providers:
                for pollutant in pollutants:
                    col = f"forecast_{provider}_{pollutant}"
                    if col not in date_data.columns:
                        continue

                    # Distance-weighted average of other cities
                    weighted_sum = 0
                    weight_sum = 0

                    for other_city in other_cities:
                        other_city_data = date_data[date_data["city"] == other_city]
                        if not other_city_data.empty:
                            other_value = other_city_data.iloc[0][col]
                            distance = city_distances.get(
                                (city, other_city), 500
                            )  # Default 500km
                            weight = 1 / (distance + 10)  # Inverse distance weighting

                            weighted_sum += other_value * weight
                            weight_sum += weight

                    if weight_sum > 0:
                        df_sorted.loc[i, f"{provider}_{pollutant}_spatial_avg"] = (
                            weighted_sum / weight_sum
                        )

                        # Spatial gradient (difference from regional average)
                        current_value = row[col]
                        regional_avg = weighted_sum / weight_sum
                        df_sorted.loc[i, f"{provider}_{pollutant}_spatial_gradient"] = (
                            current_value - regional_avg
                        )

    # Regional transport indicators
    if "wind_direction" in df_sorted.columns and "wind_speed" in df_sorted.columns:
        for i, row in df_sorted.iterrows():
            city = row["city"]
            wind_dir = row["wind_direction"]
            wind_speed = row["wind_speed"]

            # Determine upwind city (simplified)
            upwind_city = None
            if city == "Berlin":
                if 225 <= wind_dir < 315:  # SW to NW winds
                    upwind_city = "Hamburg" if 270 <= wind_dir < 315 else "Munich"
            elif city == "Hamburg":
                if 45 <= wind_dir < 135:  # NE to SE winds
                    upwind_city = "Berlin"
            elif city == "Munich":
                if 315 <= wind_dir or wind_dir < 45:  # N to NE winds
                    upwind_city = "Berlin"

            # Transport potential (wind speed * transport efficiency)
            transport_efficiency = wind_speed / 10.0  # Normalize wind speed
            df_sorted.loc[i, "upwind_transport_potential"] = transport_efficiency
            df_sorted.loc[i, "has_upwind_city"] = 1 if upwind_city else 0

    # City characteristics
    city_characteristics = {
        "Berlin": {
            "urban_intensity": 0.9,
            "coastal_influence": 0.1,
            "industrial_score": 0.7,
        },
        "Hamburg": {
            "urban_intensity": 0.8,
            "coastal_influence": 0.9,
            "industrial_score": 0.8,
        },
        "Munich": {
            "urban_intensity": 0.8,
            "coastal_influence": 0.0,
            "industrial_score": 0.6,
        },
    }

    for char in ["urban_intensity", "coastal_influence", "industrial_score"]:
        df_sorted[char] = df_sorted["city"].map(
            {city: chars[char] for city, chars in city_characteristics.items()}
        )

    # Merge back to original dataframe
    spatial_cols = [col for col in df_sorted.columns if col not in df.columns]
    for col in spatial_cols:
        df[col] = df_sorted[col]

    log.info(f"Added {len(spatial_cols)} spatial features")
    return df


def add_forecast_uncertainty_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features related to forecast uncertainty and model confidence.
    """
    df = df.copy()

    log.info("Adding forecast uncertainty features...")

    pollutants = ["pm25", "pm10", "no2", "o3"]

    for pollutant in pollutants:
        cams_col = f"forecast_cams_{pollutant}"
        noaa_col = f"forecast_noaa_gefs_aerosol_{pollutant}"

        if cams_col in df.columns and noaa_col in df.columns:
            # Model spread (disagreement)
            df[f"model_spread_{pollutant}"] = abs(df[cams_col] - df[noaa_col])
            df[f"model_spread_relative_{pollutant}"] = df[
                f"model_spread_{pollutant}"
            ] / ((df[cams_col] + df[noaa_col]) / 2 + 0.1)

            # Model agreement confidence (inverse of spread)
            df[f"model_confidence_{pollutant}"] = 1 / (
                1 + df[f"model_spread_{pollutant}"]
            )

            # Extreme value flags
            mean_val = (df[cams_col] + df[noaa_col]) / 2
            df[f"extreme_high_{pollutant}"] = (
                mean_val > mean_val.quantile(0.9)
            ).astype(int)
            df[f"extreme_low_{pollutant}"] = (mean_val < mean_val.quantile(0.1)).astype(
                int
            )

    # Overall model agreement
    spread_cols = [
        col
        for col in df.columns
        if col.startswith("model_spread_") and not "relative" in col
    ]
    if spread_cols:
        df["overall_model_agreement"] = df[spread_cols].mean(axis=1)
        df["overall_model_confidence"] = 1 / (1 + df["overall_model_agreement"])

    # Ensemble variance (if ensemble forecasts exist)
    ensemble_methods = [
        "simple_avg",
        "weighted_avg",
        "ridge",
        "xgboost",
        "bias_corrected",
    ]

    for pollutant in pollutants:
        ensemble_cols = []
        for method in ensemble_methods:
            col = f"forecast_{method}_{pollutant}"
            if col in df.columns:
                ensemble_cols.append(col)

        if len(ensemble_cols) > 1:
            # Calculate variance across ensemble methods
            ensemble_data = df[ensemble_cols]
            df[f"ensemble_variance_{pollutant}"] = ensemble_data.var(axis=1)
            df[f"ensemble_range_{pollutant}"] = ensemble_data.max(
                axis=1
            ) - ensemble_data.min(axis=1)

            # Ensemble consensus strength
            df[f"ensemble_consensus_{pollutant}"] = 1 / (
                1 + df[f"ensemble_variance_{pollutant}"]
            )

    log.info("Added forecast uncertainty features")
    return df


def add_external_data_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add synthetic external data features that would come from external APIs in production.
    """
    df = df.copy()
    if "date_dt" not in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"])

    log.info("Adding external data integration features...")

    np.random.seed(43)  # Different seed for external data

    # Traffic data (synthetic)
    for i, row in df.iterrows():
        city = row["city"]
        day_of_week = row["date_dt"].dayofweek

        # City-specific traffic baselines
        traffic_baselines = {
            "Berlin": 0.8,  # High traffic
            "Hamburg": 0.7,  # Medium-high traffic
            "Munich": 0.75,  # Medium-high traffic
        }

        base_traffic = traffic_baselines.get(city, 0.7)

        # Weekend reduction
        if day_of_week >= 5:  # Weekend
            base_traffic *= 0.6

        # Holiday reduction
        if row.get("school_holiday", 0):
            base_traffic *= 0.8

        # Add noise
        df.loc[i, "traffic_intensity"] = max(
            0.1, base_traffic + np.random.normal(0, 0.1)
        )

        # Rush hour simulation (morning and evening peaks)
        morning_peak = 0.3 + 0.2 * np.random.random()
        evening_peak = 0.4 + 0.3 * np.random.random()
        df.loc[i, "morning_traffic_peak"] = (
            morning_peak if day_of_week < 5 else morning_peak * 0.5
        )
        df.loc[i, "evening_traffic_peak"] = (
            evening_peak if day_of_week < 5 else evening_peak * 0.5
        )

    # Industrial activity (synthetic)
    for i, row in df.iterrows():
        city = row["city"]
        day_of_week = row["date_dt"].dayofweek

        # City-specific industrial activity
        industrial_baselines = {
            "Berlin": 0.7,  # Mixed economy
            "Hamburg": 0.85,  # Major port and industry
            "Munich": 0.6,  # More service-oriented
        }

        base_industrial = industrial_baselines.get(city, 0.7)

        # Weekend reduction
        if day_of_week >= 5:
            base_industrial *= 0.3

        # Holiday reduction
        if row.get("school_holiday", 0):
            base_industrial *= 0.5

        df.loc[i, "industrial_activity"] = max(
            0.1, base_industrial + np.random.normal(0, 0.1)
        )

    # Construction activity (synthetic)
    # Simulate seasonal construction patterns
    for i, row in df.iterrows():
        day_of_year = row["date_dt"].timetuple().tm_yday

        # Construction peaks in summer
        seasonal_construction = 0.5 + 0.3 * np.sin((day_of_year - 60) * 2 * np.pi / 365)

        # City-specific construction levels
        city_construction_factors = {
            "Berlin": 1.2,  # High development
            "Hamburg": 1.0,  # Moderate
            "Munich": 1.1,  # High development
        }

        city_factor = city_construction_factors.get(row["city"], 1.0)
        construction_level = seasonal_construction * city_factor

        # Weekend and holiday reductions
        if row["date_dt"].dayofweek >= 5:
            construction_level *= 0.2
        if row.get("school_holiday", 0):
            construction_level *= 0.8

        df.loc[i, "construction_activity"] = max(
            0.05, construction_level + np.random.normal(0, 0.1)
        )

    # Fire/wildfire indicators (synthetic)
    # Simulate occasional fire events
    for i, row in df.iterrows():
        # Random fire events (low probability)
        fire_prob = 0.05  # 5% chance per day
        if np.random.random() < fire_prob:
            fire_intensity = np.random.exponential(0.3)  # Exponential distribution
            df.loc[i, "fire_activity"] = min(1.0, fire_intensity)
        else:
            df.loc[i, "fire_activity"] = 0.0

        # Drought conditions affect fire risk
        if "precipitation" in df.columns:
            drought_days = max(
                0, 7 - row["precipitation"]
            )  # Days since rain (simplified)
            df.loc[i, "drought_fire_risk"] = min(1.0, drought_days / 14.0)
        else:
            df.loc[i, "drought_fire_risk"] = 0.2

    # Economic activity indicators (synthetic)
    # Simulate economic cycles affecting emissions
    for i, row in df.iterrows():
        day_of_week = row["date_dt"].dayofweek

        # Base economic activity
        base_economic = 0.75

        # Weekly pattern
        if day_of_week < 5:  # Weekday
            base_economic = 0.8
        elif day_of_week == 5:  # Saturday
            base_economic = 0.6
        else:  # Sunday
            base_economic = 0.4

        # Holiday effects
        if row.get("school_holiday", 0):
            base_economic *= 0.9  # Slight reduction during holidays

        # Seasonal tourism (August is peak tourist season)
        tourism_boost = 1.1  # 10% boost in August
        base_economic *= tourism_boost

        df.loc[i, "economic_activity"] = max(
            0.2, base_economic + np.random.normal(0, 0.05)
        )

    # Composite external indicators
    df["anthropogenic_activity"] = (
        df["traffic_intensity"] * 0.4
        + df["industrial_activity"] * 0.3
        + df["construction_activity"] * 0.2
        + df["economic_activity"] * 0.1
    )

    df["episodic_events"] = df["fire_activity"] + df["drought_fire_risk"] * 0.5

    log.info("Added external data integration features")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between different feature categories.
    """
    df = df.copy()

    log.info("Adding interaction features...")

    # Weather-pollution interactions
    if all(col in df.columns for col in ["temperature", "wind_speed", "humidity"]):
        # Temperature-wind interaction (affects dispersion)
        df["temp_wind_interaction"] = df["temperature"] * df["wind_speed"] / 100

        # Humidity-temperature interaction (affects secondary aerosol formation)
        df["humid_temp_interaction"] = df["humidity"] * df["temperature"] / 1000

        # Stagnation index (low wind + high pressure)
        if "pressure" in df.columns:
            df["stagnation_index"] = df["pressure"] / (df["wind_speed"] + 0.5)

    # Traffic-meteorology interactions
    if all(
        col in df.columns
        for col in ["traffic_intensity", "wind_speed", "boundary_layer_height"]
    ):
        # Traffic under poor dispersion conditions
        df["traffic_stagnation"] = df["traffic_intensity"] / (
            df["wind_speed"] * df["boundary_layer_height"] / 1000 + 0.1
        )

    # Temporal-activity interactions
    if "is_weekend" in df.columns and "anthropogenic_activity" in df.columns:
        df["weekend_activity_change"] = df["anthropogenic_activity"] * (
            1 - df["is_weekend"]
        )

    # City-specific weather interactions
    city_dummies = pd.get_dummies(df["city"], prefix="city")
    for city_col in city_dummies.columns:
        if "temperature" in df.columns:
            df[f"{city_col}_temp_interaction"] = (
                city_dummies[city_col] * df["temperature"]
            )

    log.info("Added interaction features")
    return df


def create_comprehensive_enhanced_dataset(
    input_path: Path, output_path: Path = None
) -> pd.DataFrame:
    """
    Main function to create comprehensively enhanced dataset with all advanced features.
    """
    log.info(f"Loading dataset from {input_path}")

    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    log.info(f"Original dataset shape: {df.shape}")
    original_cols = df.shape[1]

    # Apply all feature engineering steps
    df = add_meteorological_features(df)
    df = add_advanced_temporal_features(df)
    df = add_cross_pollutant_features(df)
    df = add_spatial_features(df)
    df = add_forecast_uncertainty_features(df)
    df = add_external_data_features(df)
    df = add_interaction_features(df)

    # Clean up temporary columns
    df = df.drop(columns=["date_dt"], errors="ignore")

    log.info(f"Enhanced dataset shape: {df.shape}")
    log.info(f"Added {df.shape[1] - original_cols} new features")

    # Save enhanced dataset
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        # Also save as CSV for inspection
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

        log.info(f"Saved comprehensively enhanced dataset to {output_path}")

    return df


def print_comprehensive_feature_summary(df: pd.DataFrame):
    """Print a comprehensive summary of all features in the enhanced dataset."""

    print("\n" + "=" * 100)
    print("COMPREHENSIVE ENHANCED DATASET FEATURE SUMMARY")
    print("=" * 100)

    feature_groups = {
        "Identity": [
            col
            for col in df.columns
            if col in ["city", "date", "forecast_made_date", "forecast_lead_hours"]
        ],
        "Actuals": [col for col in df.columns if col.startswith("actual_")],
        "CAMS Forecasts": [
            col for col in df.columns if col.startswith("forecast_cams_")
        ],
        "NOAA Forecasts": [
            col for col in df.columns if col.startswith("forecast_noaa_")
        ],
        "Ensemble Forecasts": [
            col
            for col in df.columns
            if any(
                method in col
                for method in [
                    "simple_avg",
                    "weighted_avg",
                    "ridge",
                    "xgboost",
                    "bias_corrected",
                ]
            )
        ],
        "Meteorological": [
            col
            for col in df.columns
            if any(
                met in col
                for met in [
                    "wind",
                    "temp",
                    "humid",
                    "precip",
                    "pressure",
                    "solar",
                    "boundary",
                ]
            )
        ],
        "Temporal Advanced": [
            col
            for col in df.columns
            if any(
                temp in col
                for temp in [
                    "rush",
                    "holiday",
                    "weekend",
                    "week_",
                    "august_",
                    "sin",
                    "cos",
                ]
            )
        ],
        "Cross-Pollutant": [
            col
            for col in df.columns
            if any(
                cross in col
                for cross in [
                    "ratio",
                    "load",
                    "secondary",
                    "fresh",
                    "aged",
                    "photochem",
                ]
            )
        ],
        "Spatial": [
            col
            for col in df.columns
            if any(
                spatial in col
                for spatial in [
                    "spatial",
                    "gradient",
                    "upwind",
                    "urban",
                    "coastal",
                    "industrial",
                ]
            )
        ],
        "Uncertainty": [
            col
            for col in df.columns
            if any(
                unc in col
                for unc in ["spread", "confidence", "variance", "consensus", "extreme"]
            )
        ],
        "External Data": [
            col
            for col in df.columns
            if any(
                ext in col
                for ext in [
                    "traffic",
                    "construction",
                    "fire",
                    "economic",
                    "anthropogenic",
                ]
            )
        ],
        "Interactions": [
            col for col in df.columns if "interaction" in col or "stagnation" in col
        ],
    }

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Cities: {df['city'].unique().tolist()}")
    print()

    total_features = 0
    for group_name, cols in feature_groups.items():
        if cols:
            print(f"{group_name} ({len(cols)} features):")
            total_features += len(cols)
            # Show first few features and non-null counts
            for col in cols[:5]:  # Show first 5 features per group
                if col in df.columns:
                    non_null = df[col].notna().sum()
                    print(f"  - {col} ({non_null}/{len(df)} non-null)")
            if len(cols) > 5:
                print(f"  ... and {len(cols) - 5} more features")
            print()

    print(f"TOTAL FEATURES: {total_features}")
    print(f"Feature expansion: {df.shape[1]} columns (from original ~36)")

    # Sample data showcase
    print("\nSAMPLE ENHANCED ROW (Berlin, 2025-08-01):")
    sample_row = df[(df["city"] == "Berlin") & (df["date"] == "2025-08-01")]
    if not sample_row.empty:
        row = sample_row.iloc[0]
        print(
            f"  Weather: {row.get('temperature', 'N/A'):.1f}°C, Wind: {row.get('wind_speed', 'N/A'):.1f}m/s, Humidity: {row.get('humidity', 'N/A'):.0f}%"
        )
        print(
            f"  Air Quality: PM2.5={row.get('actual_pm25', 'N/A'):.1f}, O3={row.get('actual_o3', 'N/A'):.1f}"
        )
        print(
            f"  External: Traffic={row.get('traffic_intensity', 'N/A'):.2f}, Industrial={row.get('industrial_activity', 'N/A'):.2f}"
        )
        print(
            f"  Spatial: Urban={row.get('urban_intensity', 'N/A'):.1f}, Transport={row.get('upwind_transport_potential', 'N/A'):.2f}"
        )
        print(
            f"  Uncertainty: Model Agreement={row.get('overall_model_confidence', 'N/A'):.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Add comprehensive advanced features to forecast dataset"
    )
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument(
        "--output", help="Output path for comprehensively enhanced dataset"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print detailed feature summary"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        log.error(f"Input file does not exist: {input_path}")
        return 1

    # Default output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / input_path.name.replace(
            ".", "_comprehensive_enhanced."
        )

    # Create comprehensively enhanced dataset
    enhanced_df = create_comprehensive_enhanced_dataset(input_path, output_path)

    if args.summary:
        print_comprehensive_feature_summary(enhanced_df)
    else:
        print(f"\nComprehensively enhanced dataset created: {output_path}")
        print(
            f"Shape: {enhanced_df.shape} (added {enhanced_df.shape[1] - 36} features)"
        )

    return 0


if __name__ == "__main__":
    exit(main())
