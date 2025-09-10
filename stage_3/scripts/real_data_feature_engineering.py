#!/usr/bin/env python3
"""
Real Data Feature Engineering Pipeline

This script takes real external data collected from free APIs and transforms it
into features that can enhance air quality forecasting models.

It integrates with the existing synthetic features and creates a hybrid dataset
using both real external data and forecast data.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


class RealDataFeatureEngineer:
    """Transform real external data into ML features."""

    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}

    def process_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process real weather data into advanced features."""
        df = df.copy()

        log.info("Processing real weather features...")

        # Basic weather features (rename for consistency)
        weather_mapping = {
            "weather_temperature": "real_temperature",
            "weather_humidity": "real_humidity",
            "weather_pressure": "real_pressure",
            "weather_wind_speed": "real_wind_speed",
            "weather_wind_direction": "real_wind_direction",
            "weather_cloud_cover": "real_cloud_cover",
            "weather_visibility": "real_visibility",
        }

        for old_col, new_col in weather_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]

        # Derived meteorological features
        if "real_temperature" in df.columns:
            df["real_temp_squared"] = df["real_temperature"] ** 2

            # Temperature categories
            df["real_temp_category"] = pd.cut(
                df["real_temperature"],
                bins=[-np.inf, 10, 20, 25, np.inf],
                labels=["cold", "cool", "warm", "hot"],
            )

        if "real_wind_speed" in df.columns and "real_wind_direction" in df.columns:
            # Wind components
            df["real_wind_u"] = df["real_wind_speed"] * np.cos(
                np.radians(df["real_wind_direction"])
            )
            df["real_wind_v"] = df["real_wind_speed"] * np.sin(
                np.radians(df["real_wind_direction"])
            )

            # Wind categories
            df["real_wind_category"] = pd.cut(
                df["real_wind_speed"],
                bins=[0, 2, 5, 10, np.inf],
                labels=["calm", "light", "moderate", "strong"],
            )

        # Atmospheric stability indicators
        if all(
            col in df.columns
            for col in ["real_temperature", "real_wind_speed", "real_cloud_cover"]
        ):
            # Simple stability index (high temp + low wind + clear skies = unstable)
            df["real_stability_index"] = (
                (df["real_temperature"] - 15) / 10  # Temperature effect
                + (5 - df["real_wind_speed"]) / 5  # Wind effect (inverted)
                + (100 - df["real_cloud_cover"]) / 100  # Cloud effect (inverted)
            ) / 3

        # Weather condition encoding
        if "weather_weather_description" in df.columns:
            # Create weather type categories
            df["weather_description_clean"] = df["weather_weather_description"].fillna(
                "clear"
            )

            # Group similar weather conditions
            weather_groups = {
                "clear": ["clear", "sunny"],
                "cloudy": ["cloud", "overcast", "fog", "mist"],
                "rainy": ["rain", "drizzle", "shower"],
                "stormy": ["storm", "thunder"],
                "snowy": ["snow", "sleet", "hail"],
            }

            df["weather_group"] = "other"
            for group, keywords in weather_groups.items():
                mask = df["weather_description_clean"].str.contains(
                    "|".join(keywords), case=False, na=False
                )
                df.loc[mask, "weather_group"] = group

        log.info("Added real weather features")
        return df

    def process_fire_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process real fire data into features."""
        df = df.copy()

        log.info("Processing real fire features...")

        # Fire activity features
        fire_cols = [col for col in df.columns if col.startswith("fire_")]

        if fire_cols:
            # Normalize fire count and intensity
            if "fire_fire_count" in df.columns:
                df["real_fire_activity"] = df["fire_fire_count"]
                df["real_fire_active"] = (df["fire_fire_count"] > 0).astype(int)

            if "fire_total_frp" in df.columns:
                df["real_fire_intensity"] = df["fire_total_frp"]

                # Fire intensity categories
                if df["fire_total_frp"].max() > 0:
                    df["real_fire_intensity_category"] = pd.cut(
                        df["fire_total_frp"],
                        bins=[0, 10, 50, 200, np.inf],
                        labels=["none", "low", "moderate", "high"],
                    )
                else:
                    df["real_fire_intensity_category"] = "none"

            if "fire_avg_distance" in df.columns:
                df["real_fire_proximity"] = 1 / (
                    df["fire_avg_distance"] + 1
                )  # Closer fires = higher value

            if "fire_high_confidence_fires" in df.columns:
                df["real_fire_confidence_ratio"] = df["fire_high_confidence_fires"] / (
                    df["fire_fire_count"] + 1
                )
        else:
            # No fire data available
            df["real_fire_activity"] = 0
            df["real_fire_active"] = 0
            df["real_fire_intensity"] = 0
            df["real_fire_intensity_category"] = "none"
            df["real_fire_proximity"] = 0
            df["real_fire_confidence_ratio"] = 0

        log.info("Added real fire features")
        return df

    def process_infrastructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process real infrastructure data into features."""
        df = df.copy()

        log.info("Processing real infrastructure features...")

        # Construction activity
        if "construction_site_count" in df.columns:
            df["real_construction_activity"] = df["construction_site_count"]
            df["real_construction_active"] = (df["construction_site_count"] > 0).astype(
                int
            )

            # Construction intensity categories
            df["real_construction_intensity"] = pd.cut(
                df["construction_site_count"],
                bins=[0, 1, 5, 20, np.inf],
                labels=["none", "low", "moderate", "high"],
            )
        else:
            df["real_construction_activity"] = 0
            df["real_construction_active"] = 0
            df["real_construction_intensity"] = "none"

        # Traffic infrastructure density
        infrastructure_cols = [
            col for col in df.columns if col.startswith("infrastructure_")
        ]

        if infrastructure_cols:
            # Create infrastructure density score
            infra_weights = {
                "infrastructure_major_roads": 3,  # High traffic impact
                "infrastructure_railways": 1,  # Moderate impact
                "infrastructure_fuel_stations": 2,  # Moderate-high impact
                "infrastructure_industrial_areas": 4,  # High emission impact
            }

            df["real_infrastructure_density"] = 0
            for col, weight in infra_weights.items():
                if col in df.columns:
                    df["real_infrastructure_density"] += df[col] * weight

            # Infrastructure categories
            if df["real_infrastructure_density"].max() > 0:
                df["real_infrastructure_category"] = pd.cut(
                    df["real_infrastructure_density"],
                    bins=[0, 5, 15, 30, np.inf],
                    labels=["low", "medium", "high", "very_high"],
                )
            else:
                df["real_infrastructure_category"] = "low"

            # Individual infrastructure features
            for col in infra_weights.keys():
                if col in df.columns:
                    new_col = col.replace("infrastructure_", "real_")
                    df[new_col] = df[col]
        else:
            df["real_infrastructure_density"] = 0
            df["real_infrastructure_category"] = "low"
            df["real_major_roads"] = 0
            df["real_railways"] = 0
            df["real_fuel_stations"] = 0
            df["real_industrial_areas"] = 0

        log.info("Added real infrastructure features")
        return df

    def process_seismic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process earthquake data into features."""
        df = df.copy()

        log.info("Processing real seismic features...")

        if "earthquake_count" in df.columns:
            df["real_seismic_activity"] = df["earthquake_count"]
            df["real_seismic_active"] = (df["earthquake_count"] > 0).astype(int)
        else:
            df["real_seismic_activity"] = 0
            df["real_seismic_active"] = 0

        if "max_earthquake_magnitude" in df.columns:
            df["real_max_earthquake_magnitude"] = df["max_earthquake_magnitude"]

            # Earthquake risk categories
            df["real_earthquake_risk"] = pd.cut(
                df["max_earthquake_magnitude"],
                bins=[0, 3, 5, 7, np.inf],
                labels=["low", "moderate", "high", "extreme"],
            )
        else:
            df["real_max_earthquake_magnitude"] = 0
            df["real_earthquake_risk"] = "low"

        if "avg_earthquake_depth" in df.columns:
            df["real_avg_earthquake_depth"] = df["avg_earthquake_depth"]
        else:
            df["real_avg_earthquake_depth"] = 0

        log.info("Added real seismic features")
        return df

    def process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process real temporal data (holidays, etc.)."""
        df = df.copy()

        log.info("Processing real temporal features...")

        # Holiday features
        if "is_public_holiday" in df.columns:
            df["real_is_public_holiday"] = df["is_public_holiday"].astype(int)
        else:
            df["real_is_public_holiday"] = 0

        if "is_school_holiday" in df.columns:
            df["real_is_school_holiday"] = df["is_school_holiday"].astype(int)
        else:
            df["real_is_school_holiday"] = 0

        # Combined holiday effect
        df["real_holiday_effect"] = (
            df.get("real_is_public_holiday", 0)
            * 2  # Public holidays have stronger effect
            + df.get("real_is_school_holiday", 0)
            * 1  # School holidays have moderate effect
        )

        # Encode holiday names if available
        if "school_holiday_name" in df.columns:
            df["school_holiday_name_clean"] = df["school_holiday_name"].fillna("none")
            if "school_holiday_name_clean" not in self.label_encoders:
                self.label_encoders["school_holiday_name_clean"] = LabelEncoder()
                df["real_school_holiday_type"] = self.label_encoders[
                    "school_holiday_name_clean"
                ].fit_transform(df["school_holiday_name_clean"])
            else:
                df["real_school_holiday_type"] = self.label_encoders[
                    "school_holiday_name_clean"
                ].transform(df["school_holiday_name_clean"])
        else:
            df["real_school_holiday_type"] = 0

        log.info("Added real temporal features")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between real data sources."""
        df = df.copy()

        log.info("Creating real data interaction features...")

        # Weather-fire interactions
        if all(
            col in df.columns
            for col in ["real_temperature", "real_wind_speed", "real_fire_activity"]
        ):
            # Fire risk increases with high temp, low wind
            df["real_fire_weather_risk"] = (
                (df["real_temperature"] - 15)
                / 10  # Temperature factor
                * (5 - df["real_wind_speed"])
                / 5  # Wind factor (inverted)
                * (df["real_fire_activity"] + 1)  # Current fire activity
            )

        # Construction-weather interactions
        if all(
            col in df.columns
            for col in ["real_construction_activity", "real_wind_speed"]
        ):
            # Construction dust dispersal
            df["real_construction_dispersion"] = df["real_construction_activity"] / (
                df["real_wind_speed"] + 1
            )

        # Infrastructure-weather interactions
        if all(
            col in df.columns
            for col in ["real_infrastructure_density", "real_stability_index"]
        ):
            # Urban heat island and stagnation effect
            df["real_urban_stagnation"] = df["real_infrastructure_density"] * (
                df["real_stability_index"] + 1
            )

        # Holiday-infrastructure interactions
        if all(
            col in df.columns for col in ["real_holiday_effect", "real_major_roads"]
        ):
            # Traffic reduction during holidays
            df["real_holiday_traffic_reduction"] = (
                df["real_holiday_effect"] * df["real_major_roads"] / 10
            )

        log.info("Added real data interaction features")
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for ML models."""
        df = df.copy()

        log.info("Encoding categorical features...")

        categorical_cols = [
            "real_temp_category",
            "real_wind_category",
            "weather_group",
            "real_fire_intensity_category",
            "real_construction_intensity",
            "real_infrastructure_category",
            "real_earthquake_risk",
        ]

        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                        df[col].astype(str).fillna("unknown")
                    )
                else:
                    df[f"{col}_encoded"] = self.label_encoders[col].transform(
                        df[col].astype(str).fillna("unknown")
                    )

        log.info("Encoded categorical features")
        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        df = df.copy()

        log.info("Normalizing numerical features...")

        # Features to normalize
        normalize_cols = [
            "real_temperature",
            "real_humidity",
            "real_pressure",
            "real_wind_speed",
            "real_infrastructure_density",
            "real_construction_activity",
            "real_fire_intensity",
        ]

        for col in normalize_cols:
            if col in df.columns and df[col].std() > 0:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[f"{col}_normalized"] = (
                        self.scalers[col]
                        .fit_transform(df[[col]].fillna(df[col].mean()))
                        .flatten()
                    )
                else:
                    df[f"{col}_normalized"] = (
                        self.scalers[col]
                        .transform(df[[col]].fillna(df[col].mean()))
                        .flatten()
                    )

        log.info("Normalized numerical features")
        return df

    def process_all_real_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all real data into ML features."""
        log.info(f"Processing real data features. Input shape: {df.shape}")

        # Apply all feature engineering steps
        df = self.process_weather_features(df)
        df = self.process_fire_features(df)
        df = self.process_infrastructure_features(df)
        df = self.process_seismic_features(df)
        df = self.process_temporal_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df)
        df = self.normalize_features(df)

        log.info(f"Real feature processing complete. Output shape: {df.shape}")

        return df


def integrate_real_and_forecast_data(
    forecast_df: pd.DataFrame,
    real_data_df: pd.DataFrame,
    join_keys: List[str] = ["city", "date"],
) -> pd.DataFrame:
    """
    Integrate real external data with forecast comparison data.
    """
    log.info("Integrating real data with forecast data...")
    log.info(f"Forecast data shape: {forecast_df.shape}")
    log.info(f"Real data shape: {real_data_df.shape}")

    # Process real data features
    feature_engineer = RealDataFeatureEngineer()
    real_features_df = feature_engineer.process_all_real_features(real_data_df)

    # Merge datasets
    integrated_df = forecast_df.merge(
        real_features_df, on=join_keys, how="left", suffixes=("", "_real_duplicate")
    )

    # Remove duplicate columns
    duplicate_cols = [
        col for col in integrated_df.columns if col.endswith("_real_duplicate")
    ]
    integrated_df = integrated_df.drop(columns=duplicate_cols)

    log.info(f"Integrated data shape: {integrated_df.shape}")
    log.info(
        f"Added {integrated_df.shape[1] - forecast_df.shape[1]} real data features"
    )

    return integrated_df


def main():
    """Test real data feature engineering."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process real external data into ML features"
    )
    parser.add_argument(
        "--real-data", required=True, help="Real external data CSV file"
    )
    parser.add_argument(
        "--forecast-data", help="Forecast comparison data to integrate with"
    )
    parser.add_argument(
        "--output",
        default="data/analysis/real_features_processed.csv",
        help="Output file",
    )

    args = parser.parse_args()

    # Load real data
    real_data_path = Path(args.real_data)
    if not real_data_path.exists():
        log.error(f"Real data file not found: {real_data_path}")
        return 1

    log.info(f"Loading real data from {real_data_path}")
    real_df = pd.read_csv(real_data_path)

    if args.forecast_data:
        # Integration mode
        forecast_path = Path(args.forecast_data)
        if not forecast_path.exists():
            log.error(f"Forecast data file not found: {forecast_path}")
            return 1

        log.info(f"Loading forecast data from {forecast_path}")
        if forecast_path.suffix == ".csv":
            forecast_df = pd.read_csv(forecast_path)
        else:
            forecast_df = pd.read_parquet(forecast_path)

        # Integrate datasets
        result_df = integrate_real_and_forecast_data(forecast_df, real_df)
    else:
        # Feature engineering only mode
        feature_engineer = RealDataFeatureEngineer()
        result_df = feature_engineer.process_all_real_features(real_df)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    log.info(f"Processed real data features saved to {output_path}")
    log.info(f"Output shape: {result_df.shape}")

    # Show sample features
    real_feature_cols = [col for col in result_df.columns if col.startswith("real_")]
    log.info(f"Created {len(real_feature_cols)} real data features")

    print(f"\nReal data features created ({len(real_feature_cols)}):")
    for col in real_feature_cols[:15]:  # Show first 15
        print(f"  - {col}")
    if len(real_feature_cols) > 15:
        print(f"  ... and {len(real_feature_cols) - 15} more")

    return 0


if __name__ == "__main__":
    exit(main())
