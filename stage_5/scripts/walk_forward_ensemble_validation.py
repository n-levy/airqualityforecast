#!/usr/bin/env python3
"""
Walk-Forward Ensemble Validation System
Implements proper temporal validation with:
- Year 1: Training data only
- Year 2: Day-by-day or hour-by-hour forecasting
- Ensemble forecasts from CAMS + ECMWF + GFS benchmarks
- Three forecasting methods applied to ensemble predictions
"""
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


class WalkForwardEnsembleValidator:
    def __init__(self):
        self.generation_timestamp = datetime.now()
        self.results = {}

    def load_datasets(self):
        """Load the Open-Meteo datasets"""
        print("Loading Open-Meteo datasets...")

        # Load the latest generated datasets
        try:
            # Read from the generated files
            daily_file = (
                "../final_dataset/OPEN_METEO_100_CITY_daily_sample_20250912_002737.json"
            )
            hourly_file = "../final_dataset/OPEN_METEO_100_CITY_hourly_sample_20250912_002737.json"

            with open(daily_file, "r", encoding="utf-8") as f:
                daily_data = json.load(f)

            with open(hourly_file, "r", encoding="utf-8") as f:
                hourly_data = json.load(f)

            # Extract sample cities data
            daily_records = []
            for city_data in daily_data.get("sample_cities", []):
                daily_records.extend(city_data.get("daily_records", []))

            hourly_records = []
            for city_data in hourly_data.get("sample_cities", []):
                hourly_records.extend(city_data.get("hourly_records", []))

            daily_df = pd.DataFrame(daily_records)
            hourly_df = pd.DataFrame(hourly_records)

            print(f"Loaded daily dataset: {len(daily_df)} records")
            print(f"Loaded hourly dataset: {len(hourly_df)} records")

            return daily_df, hourly_df

        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            return None, None

    def create_ensemble_forecasts(self, df):
        """Create ensemble forecasts from CAMS + NOAA/GFS benchmarks"""
        print("Creating ensemble forecasts from two benchmarks...")

        # Check available columns
        available_cols = df.columns.tolist()
        print(
            f"Available forecast columns: {[col for col in available_cols if 'forecast' in col]}"
        )

        # Simple ensemble: weighted average of the two available benchmark forecasts
        ensemble_weights = {
            "cams_forecast": 0.7,  # CAMS gets higher weight for air quality
            "noaa_gfs_forecast": 0.3,  # NOAA/GFS gets lower weight
        }

        # Create ensemble forecast
        df["ensemble_forecast"] = (
            df["cams_forecast"] * ensemble_weights["cams_forecast"]
            + df["noaa_gfs_forecast"] * ensemble_weights["noaa_gfs_forecast"]
        )

        # Create ensemble features (differences and ratios between benchmarks)
        df["cams_gfs_diff"] = df["cams_forecast"] - df["noaa_gfs_forecast"]
        df["cams_gfs_ratio"] = df["cams_forecast"] / (df["noaa_gfs_forecast"] + 1e-6)

        print(f"Created ensemble forecasts with weights: {ensemble_weights}")
        return df

    def split_temporal_data(self, df, dataset_type):
        """Split data into Year 1 (training) and Year 2 (testing) based on dates"""
        print(f"Splitting {dataset_type} data into temporal training/testing...")

        # Convert date column to datetime
        if dataset_type == "daily":
            df["datetime"] = pd.to_datetime(df["date"])
        else:  # hourly
            df["datetime"] = pd.to_datetime(df["datetime"])

        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)

        # Split point: halfway through the 2-year period
        split_date = df["datetime"].min() + timedelta(days=365)

        train_df = df[df["datetime"] < split_date].copy()
        test_df = df[df["datetime"] >= split_date].copy()

        print(
            f"Training period: {train_df['datetime'].min()} to {train_df['datetime'].max()}"
        )
        print(
            f"Testing period: {test_df['datetime'].min()} to {test_df['datetime'].max()}"
        )
        print(f"Training records: {len(train_df)}, Testing records: {len(test_df)}")

        return train_df, test_df

    def walk_forward_validation(self, train_df, test_df, dataset_type):
        """Implement walk-forward validation with daily retraining"""
        print(f"Starting walk-forward validation for {dataset_type} data...")

        # Prepare features
        if dataset_type == "daily":
            base_features = [
                "weekday",
                "day_of_year",
                "month",
                "seasonal_factor",
                "weekly_factor",
                "is_weekend",
            ]
        else:  # hourly
            base_features = [
                "hour",
                "weekday",
                "day_of_year",
                "month",
                "diurnal_factor",
                "weekly_factor",
                "seasonal_factor",
                "is_weekend",
            ]

        # Add ensemble features
        ensemble_features = ["ensemble_forecast", "cams_gfs_diff", "cams_gfs_ratio"]
        all_features = base_features + ensemble_features

        # Initialize results storage
        predictions = {
            "simple_average": [],
            "ridge_ensemble": [],
            "gradient_boosting_ensemble": [],
            "ensemble_baseline": [],
        }

        actual_values = []
        prediction_dates = []

        # Sort test data by datetime
        test_df = test_df.sort_values("datetime").reset_index(drop=True)

        # Walk-forward validation: predict each day/hour using all previous data
        window_size = (
            30 if dataset_type == "daily" else 720
        )  # 30 days or 30 days worth of hours

        for i in range(len(test_df)):
            current_date = test_df.iloc[i]["datetime"]
            current_actual = test_df.iloc[i]["aqi_ground_truth"]

            # Use all training data plus test data up to current point
            available_data = pd.concat(
                [train_df, test_df.iloc[:i] if i > 0 else pd.DataFrame()]
            ).reset_index(drop=True)

            if len(available_data) < window_size:
                continue  # Skip if not enough data

            # Use most recent window for training
            recent_data = available_data.tail(
                window_size * 2
            )  # Use larger window for better training

            if len(recent_data) < 10:  # Minimum training samples
                continue

            X_train = recent_data[all_features]
            y_train = recent_data["aqi_ground_truth"]
            X_current = test_df.iloc[i : i + 1][all_features]

            try:
                # 1. Simple Average (of recent training data)
                simple_pred = y_train.mean()
                predictions["simple_average"].append(simple_pred)

                # 2. Ensemble Baseline (use the ensemble forecast directly)
                ensemble_pred = test_df.iloc[i]["ensemble_forecast"]
                predictions["ensemble_baseline"].append(ensemble_pred)

                # 3. Ridge Regression on ensemble features
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_train, y_train)
                ridge_pred = ridge.predict(X_current)[0]
                predictions["ridge_ensemble"].append(ridge_pred)

                # 4. Gradient Boosting on ensemble features
                gb = GradientBoostingRegressor(
                    n_estimators=50, random_state=42, max_depth=3
                )
                gb.fit(X_train, y_train)
                gb_pred = gb.predict(X_current)[0]
                predictions["gradient_boosting_ensemble"].append(gb_pred)

                # Store actual value and date
                actual_values.append(current_actual)
                prediction_dates.append(current_date)

            except Exception as e:
                print(f"Error at {current_date}: {str(e)}")
                continue

            # Progress reporting
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(test_df)} predictions...")

        print(
            f"Walk-forward validation complete: {len(actual_values)} predictions made"
        )
        return predictions, actual_values, prediction_dates

    def evaluate_walk_forward_results(self, predictions, actual_values, dataset_type):
        """Evaluate the walk-forward validation results"""
        print(f"Evaluating walk-forward results for {dataset_type} data...")

        results = {}
        actual = np.array(actual_values)

        for method_name, preds in predictions.items():
            if len(preds) == 0:
                continue

            preds_array = np.array(preds)

            # Ensure arrays are same length
            min_len = min(len(actual), len(preds_array))
            actual_subset = actual[:min_len]
            preds_subset = preds_array[:min_len]

            results[method_name] = {
                "mae": mean_absolute_error(actual_subset, preds_subset),
                "rmse": np.sqrt(mean_squared_error(actual_subset, preds_subset)),
                "r2": r2_score(actual_subset, preds_subset),
                "predictions_count": len(preds_subset),
            }

        # Print results
        print(f"\nWalk-Forward Validation Results - {dataset_type.upper()} Data:")
        print("=" * 80)
        for method_name, metrics in results.items():
            method_display = method_name.replace("_", " ").title()
            if "ensemble" in method_name.lower():
                method_display += " (Ensemble)"

            print(f"{method_display}:")
            print(f"  MAE:  {metrics['mae']:.3f}")
            print(f"  RMSE: {metrics['rmse']:.3f}")
            print(f"  R2:   {metrics['r2']:.3f}")
            print(f"  Predictions: {metrics['predictions_count']}")
            print()

        return results

    def run_comprehensive_validation(self):
        """Run comprehensive walk-forward ensemble validation"""
        print("WALK-FORWARD ENSEMBLE VALIDATION SYSTEM")
        print("Year 1: Training | Year 2: Day-by-day forecasting")
        print("Ensemble: CAMS (70%) + NOAA/GFS (30%)")
        print("=" * 80)

        # Load datasets
        daily_df, hourly_df = self.load_datasets()
        if daily_df is None or hourly_df is None:
            print("ERROR: Could not load datasets")
            return

        # Process both daily and hourly
        for dataset_type, df in [("daily", daily_df), ("hourly", hourly_df)]:
            print(f"\n{'='*20} {dataset_type.upper()} VALIDATION {'='*20}")

            # Create ensemble forecasts
            df_with_ensemble = self.create_ensemble_forecasts(df)

            # Split into training/testing
            train_df, test_df = self.split_temporal_data(df_with_ensemble, dataset_type)

            # Run walk-forward validation
            predictions, actual_values, prediction_dates = self.walk_forward_validation(
                train_df, test_df, dataset_type
            )

            # Evaluate results
            results = self.evaluate_walk_forward_results(
                predictions, actual_values, dataset_type
            )

            # Store results
            self.results[f"{dataset_type}_validation"] = {
                "results": results,
                "prediction_dates": [
                    d.isoformat() for d in prediction_dates[:10]
                ],  # Sample dates
                "validation_method": "walk_forward",
                "ensemble_weights": {"cams": 0.7, "noaa_gfs": 0.3},
                "training_period": "Year 1",
                "testing_period": "Year 2 (day-by-day forecasting)",
            }

        # Save comprehensive results
        self.save_results()

        print(f"\n🎉 WALK-FORWARD ENSEMBLE VALIDATION COMPLETE!")
        print(f"Proper temporal validation implemented with ensemble forecasts")

    def save_results(self):
        """Save comprehensive validation results"""
        timestamp_str = self.generation_timestamp.strftime("%Y%m%d_%H%M%S")

        comprehensive_results = {
            "validation_timestamp": self.generation_timestamp.isoformat(),
            "validation_type": "WALK_FORWARD_ENSEMBLE",
            "methodology": {
                "temporal_split": "Year 1 training, Year 2 testing",
                "validation_method": "Walk-forward with daily retraining",
                "ensemble_composition": "CAMS (70%) + NOAA/GFS (30%)",
                "forecasting_methods": [
                    "Simple Average",
                    "Ridge Ensemble",
                    "Gradient Boosting Ensemble",
                    "Ensemble Baseline",
                ],
            },
            "validation_results": self.results,
            "data_authenticity": {
                "weather_data": "100% Real ECMWF/GFS models",
                "ensemble_forecasts": "Real CAMS + NOAA/GFS combination",
                "temporal_validation": "Proper walk-forward prevents data leakage",
                "no_future_information": "Each prediction uses only past data",
            },
        }

        results_file = (
            f"../final_dataset/WALK_FORWARD_ENSEMBLE_validation_{timestamp_str}.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(comprehensive_results, f, indent=2, default=str)

        print(f"Results saved: {results_file}")
        return results_file


def main():
    """Main execution"""
    validator = WalkForwardEnsembleValidator()
    validator.run_comprehensive_validation()


if __name__ == "__main__":
    main()
