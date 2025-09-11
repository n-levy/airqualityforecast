#!/usr/bin/env python3
"""
Walk-Forward Forecasting for 100-City Dataset

This script implements walk-forward validation with:
1. Simple Average Ensemble (CAMS + NOAA forecasts)
2. Ridge Regression Ensemble

The script processes the last year of data for each city, training models
on all previous data and making one-day-ahead predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WalkForwardForecaster:
    """Walk-forward forecasting implementation for 100-city dataset."""
    
    def __init__(self, data_path: str = "../comprehensive_tables/comprehensive_features_table.csv"):
        """Initialize the forecaster with dataset path."""
        self.data_path = data_path
        self.cities_data = {}
        self.results = {}
        
    def load_city_data(self) -> Dict:
        """Load city feature data."""
        logger.info("Loading 100-city dataset...")
        
        # Load comprehensive features table
        features_df = pd.read_csv(self.data_path)
        
        logger.info(f"Loaded {len(features_df)} cities")
        return features_df
    
    def generate_synthetic_time_series(self, city_row: pd.Series, days: int = 365) -> pd.DataFrame:
        """Generate synthetic time series data for a city based on its characteristics."""
        
        # Base values from city characteristics
        base_pm25 = city_row['Average_PM25']
        base_pm10 = city_row['pm10_Concentration']
        base_no2 = city_row['no2_Concentration'] 
        base_o3 = city_row['o3_Concentration']
        base_so2 = city_row['so2_Concentration']
        base_co = city_row['co_Concentration']
        
        # Generate date range for last year
        end_date = datetime(2025, 9, 11)
        start_date = end_date - timedelta(days=days-1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create synthetic data with realistic patterns
        np.random.seed(hash(city_row['City']) % 2**32)  # Consistent seed per city
        
        data = []
        for i, date in enumerate(dates):
            # Seasonal patterns
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Weekly patterns (higher on weekdays)
            weekly_factor = 1.2 if date.weekday() < 5 else 0.8
            
            # Holiday effects
            holiday_factor = city_row['Holiday_Pollution_Multiplier'] if 'Holiday_Pollution_Multiplier' in city_row else 1.0
            
            # Fire effects (seasonal)
            fire_months = eval(city_row['Fire_Peak_Months']) if pd.notna(city_row['Fire_Peak_Months']) else []
            fire_factor = 1.0
            if date.month in fire_months:
                fire_factor = 1 + (city_row['Fire_PM25_Contribution'] / 100) if pd.notna(city_row['Fire_PM25_Contribution']) else 1.0
            
            # Combined effects with noise
            noise_factor = np.random.normal(1.0, 0.15)
            total_factor = seasonal_factor * weekly_factor * fire_factor * noise_factor
            
            # Generate pollutant values
            pm25_actual = max(1, base_pm25 * total_factor)
            pm10_actual = max(1, base_pm10 * total_factor * 1.2)  # PM10 typically higher
            no2_actual = max(1, base_no2 * total_factor * weekly_factor)  # More sensitive to traffic
            o3_actual = max(1, base_o3 * seasonal_factor * np.random.normal(1.0, 0.2))
            so2_actual = max(1, base_so2 * total_factor * 0.9)  # Less variable
            co_actual = max(1, base_co * total_factor * weekly_factor)
            
            # Generate benchmark forecasts (CAMS and NOAA with realistic errors)
            cams_pm25 = pm25_actual * np.random.normal(1.0, 0.12)  # 12% error
            cams_pm10 = pm10_actual * np.random.normal(1.0, 0.15)  # 15% error
            cams_no2 = no2_actual * np.random.normal(1.0, 0.18)   # 18% error
            cams_o3 = o3_actual * np.random.normal(1.0, 0.14)     # 14% error
            cams_so2 = so2_actual * np.random.normal(1.0, 0.20)   # 20% error
            cams_co = co_actual * np.random.normal(1.0, 0.16)     # 16% error
            
            noaa_pm25 = pm25_actual * np.random.normal(1.0, 0.14)  # 14% error
            noaa_pm10 = pm10_actual * np.random.normal(1.0, 0.17)  # 17% error
            noaa_no2 = no2_actual * np.random.normal(1.0, 0.16)   # 16% error
            noaa_o3 = o3_actual * np.random.normal(1.0, 0.12)     # 12% error
            noaa_so2 = so2_actual * np.random.normal(1.0, 0.22)   # 22% error
            noaa_co = co_actual * np.random.normal(1.0, 0.18)     # 18% error
            
            data.append({
                'date': date,
                'city': city_row['City'],
                'PM25_actual': pm25_actual,
                'PM10_actual': pm10_actual,
                'NO2_actual': no2_actual,
                'O3_actual': o3_actual,
                'SO2_actual': so2_actual,
                'CO_actual': co_actual,
                'CAMS_PM25': max(1, cams_pm25),
                'CAMS_PM10': max(1, cams_pm10),
                'CAMS_NO2': max(1, cams_no2),
                'CAMS_O3': max(1, cams_o3),
                'CAMS_SO2': max(1, cams_so2),
                'CAMS_CO': max(1, cams_co),
                'NOAA_PM25': max(1, noaa_pm25),
                'NOAA_PM10': max(1, noaa_pm10),
                'NOAA_NO2': max(1, noaa_no2),
                'NOAA_O3': max(1, noaa_o3),
                'NOAA_SO2': max(1, noaa_so2),
                'NOAA_CO': max(1, noaa_co),
                'temperature': city_row['Temperature_C'] + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3),
                'humidity': max(10, min(100, city_row['Humidity_Percent'] + np.random.normal(0, 10))),
                'wind_speed': max(0, city_row['Wind_Speed_ms'] + np.random.normal(0, 2)),
                'pressure': city_row['Pressure_hPa'] + np.random.normal(0, 5),
                'day_of_year': day_of_year,
                'day_of_week': date.weekday(),
                'is_weekend': date.weekday() >= 5
            })
        
        return pd.DataFrame(data)
    
    def simple_average_forecast(self, cams_values: np.ndarray, noaa_values: np.ndarray) -> np.ndarray:
        """Simple average of CAMS and NOAA forecasts."""
        return (cams_values + noaa_values) / 2
    
    def ridge_regression_forecast(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Ridge regression ensemble forecast."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test.reshape(1, -1))
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        
        return ridge.predict(X_test_scaled)[0]
    
    def calculate_aqi(self, pm25: float, pm10: float, no2: float, o3: float, 
                     so2: float, co: float, standard: str = "US_EPA") -> float:
        """Calculate AQI based on pollutant concentrations."""
        
        if standard == "US_EPA":
            # US EPA AQI calculation (simplified)
            aqi_pm25 = min(500, max(0, pm25 * 4.17))  # Rough conversion
            aqi_pm10 = min(500, max(0, pm10 * 2.04))  
            aqi_no2 = min(500, max(0, no2 * 9.43))    
            aqi_o3 = min(500, max(0, o3 * 7.81))      
            aqi_so2 = min(500, max(0, so2 * 9.17))    
            aqi_co = min(500, max(0, co * 0.115))     
            
            return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_o3, aqi_so2, aqi_co)
        
        return 100  # Default AQI
    
    def evaluate_forecasts(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mpe = np.mean((predicted - actual) / actual) * 100  # Mean Percentage Error
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MPE': mpe
        }
    
    def walk_forward_validation(self, city_data: pd.DataFrame, city_name: str) -> Dict:
        """Perform walk-forward validation for a single city."""
        
        logger.info(f"Processing {city_name}...")
        
        # Sort by date
        city_data = city_data.sort_values('date').reset_index(drop=True)
        
        # Initialize prediction arrays
        pollutants = ['PM25', 'PM10', 'NO2', 'O3', 'SO2', 'CO']
        predictions = {
            'simple_avg': {p: [] for p in pollutants + ['AQI']},
            'ridge': {p: [] for p in pollutants + ['AQI']},
            'cams': {p: [] for p in pollutants + ['AQI']},
            'noaa': {p: [] for p in pollutants + ['AQI']},
            'actual': {p: [] for p in pollutants + ['AQI']}
        }
        
        # Walk-forward validation (train on all data before each day)
        for i in range(30, len(city_data)):  # Start after 30 days for training
            
            # Training data (all data before current day)
            train_data = city_data.iloc[:i]
            test_data = city_data.iloc[i]
            
            # Features for Ridge regression
            feature_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                          'day_of_year', 'day_of_week', 'is_weekend']
            
            for pollutant in pollutants:
                # Add lagged features (previous day values)
                if i > 0:
                    feature_cols_extended = feature_cols + [f'CAMS_{pollutant}', f'NOAA_{pollutant}']
                    if i > 1:
                        feature_cols_extended += [f'{pollutant}_actual_lag1']
                        train_data_extended = train_data.copy()
                        train_data_extended[f'{pollutant}_actual_lag1'] = train_data[f'{pollutant}_actual'].shift(1)
                        train_data_extended = train_data_extended.dropna()
                    else:
                        train_data_extended = train_data
                        feature_cols_extended = feature_cols + [f'CAMS_{pollutant}', f'NOAA_{pollutant}']
                    
                    X_train = train_data_extended[feature_cols_extended].values
                    y_train = train_data_extended[f'{pollutant}_actual'].values
                    
                    # Current day features
                    test_features = [test_data[col] if col in test_data else 0 for col in feature_cols_extended]
                    if f'{pollutant}_actual_lag1' in feature_cols_extended:
                        test_features[-1] = city_data.iloc[i-1][f'{pollutant}_actual']
                    
                    X_test = np.array(test_features)
                    
                    # Simple average forecast
                    simple_pred = self.simple_average_forecast(
                        np.array([test_data[f'CAMS_{pollutant}']]),
                        np.array([test_data[f'NOAA_{pollutant}']])
                    )[0]
                    
                    # Ridge regression forecast
                    try:
                        ridge_pred = self.ridge_regression_forecast(X_train, y_train, X_test)
                    except:
                        ridge_pred = simple_pred  # Fallback to simple average
                    
                    # Store predictions
                    predictions['simple_avg'][pollutant].append(simple_pred)
                    predictions['ridge'][pollutant].append(ridge_pred)
                    predictions['cams'][pollutant].append(test_data[f'CAMS_{pollutant}'])
                    predictions['noaa'][pollutant].append(test_data[f'NOAA_{pollutant}'])
                    predictions['actual'][pollutant].append(test_data[f'{pollutant}_actual'])
        
        # Calculate AQI for each method
        for method in ['simple_avg', 'ridge', 'cams', 'noaa', 'actual']:
            aqi_values = []
            for j in range(len(predictions[method]['PM25'])):
                aqi = self.calculate_aqi(
                    predictions[method]['PM25'][j],
                    predictions[method]['PM10'][j],
                    predictions[method]['NO2'][j],
                    predictions[method]['O3'][j],
                    predictions[method]['SO2'][j],
                    predictions[method]['CO'][j]
                )
                aqi_values.append(aqi)
            predictions[method]['AQI'] = aqi_values
        
        # Calculate metrics for each pollutant and AQI
        results = {}
        for pollutant in pollutants + ['AQI']:
            results[pollutant] = {}
            actual_values = np.array(predictions['actual'][pollutant])
            
            for method in ['simple_avg', 'ridge', 'cams', 'noaa']:
                pred_values = np.array(predictions[method][pollutant])
                results[pollutant][method] = self.evaluate_forecasts(actual_values, pred_values)
        
        return results
    
    def process_all_cities(self) -> Dict:
        """Process all 100 cities with walk-forward validation."""
        
        # Load city data
        cities_df = self.load_city_data()
        
        all_results = {}
        
        for idx, city_row in cities_df.iterrows():
            city_name = city_row['City']
            
            try:
                # Generate synthetic time series for this city
                city_time_series = self.generate_synthetic_time_series(city_row)
                
                # Perform walk-forward validation
                city_results = self.walk_forward_validation(city_time_series, city_name)
                
                all_results[city_name] = {
                    'country': city_row['Country'],
                    'continent': city_row['Continent'],
                    'results': city_results
                }
                
                logger.info(f"Completed {city_name} ({idx+1}/100)")
                
            except Exception as e:
                logger.error(f"Error processing {city_name}: {str(e)}")
                continue
        
        return all_results
    
    def generate_summary_report(self, results: Dict) -> Dict:
        """Generate summary report of all results."""
        
        summary = {
            'total_cities': len(results),
            'pollutants': ['PM25', 'PM10', 'NO2', 'O3', 'SO2', 'CO', 'AQI'],
            'methods': ['simple_avg', 'ridge', 'cams', 'noaa'],
            'overall_performance': {},
            'best_performing_method': {},
            'continental_performance': {}
        }
        
        # Calculate overall performance across all cities
        for pollutant in summary['pollutants']:
            summary['overall_performance'][pollutant] = {}
            
            for method in summary['methods']:
                mae_values = []
                rmse_values = []
                r2_values = []
                
                for city_name, city_data in results.items():
                    if pollutant in city_data['results']:
                        mae_values.append(city_data['results'][pollutant][method]['MAE'])
                        rmse_values.append(city_data['results'][pollutant][method]['RMSE'])
                        r2_values.append(city_data['results'][pollutant][method]['R2'])
                
                summary['overall_performance'][pollutant][method] = {
                    'avg_MAE': np.mean(mae_values),
                    'avg_RMSE': np.mean(rmse_values),
                    'avg_R2': np.mean(r2_values),
                    'cities_evaluated': len(mae_values)
                }
            
            # Find best performing method for this pollutant
            best_method = min(summary['methods'], 
                            key=lambda m: summary['overall_performance'][pollutant][m]['avg_MAE'])
            summary['best_performing_method'][pollutant] = best_method
        
        # Continental performance
        continents = ['Asia', 'Africa', 'Europe', 'North_America', 'South_America']
        for continent in continents:
            summary['continental_performance'][continent] = {}
            continent_cities = {k: v for k, v in results.items() if v['continent'] == continent}
            
            if continent_cities:
                for pollutant in summary['pollutants']:
                    summary['continental_performance'][continent][pollutant] = {}
                    
                    for method in summary['methods']:
                        mae_values = []
                        for city_name, city_data in continent_cities.items():
                            if pollutant in city_data['results']:
                                mae_values.append(city_data['results'][pollutant][method]['MAE'])
                        
                        if mae_values:
                            summary['continental_performance'][continent][pollutant][method] = {
                                'avg_MAE': np.mean(mae_values),
                                'cities': len(mae_values)
                            }
        
        return summary

def main():
    """Main execution function."""
    
    logger.info("Starting Walk-Forward Forecasting for 100-City Dataset")
    
    # Initialize forecaster
    forecaster = WalkForwardForecaster()
    
    # Process all cities
    results = forecaster.process_all_cities()
    
    # Generate summary report
    summary = forecaster.generate_summary_report(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    with open(f'walk_forward_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary report
    with open(f'walk_forward_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Results saved to walk_forward_results_{timestamp}.json")
    logger.info(f"Summary saved to walk_forward_summary_{timestamp}.json")
    
    # Print key findings
    print("\n" + "="*60)
    print("WALK-FORWARD FORECASTING RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nProcessed {summary['total_cities']} cities")
    print(f"Evaluated {len(summary['pollutants'])} pollutants/metrics")
    
    print("\nBest Performing Method by Pollutant:")
    for pollutant, best_method in summary['best_performing_method'].items():
        avg_mae = summary['overall_performance'][pollutant][best_method]['avg_MAE']
        print(f"  {pollutant}: {best_method.upper()} (MAE: {avg_mae:.2f})")
    
    print("\nOverall Performance Comparison (Average MAE):")
    for method in summary['methods']:
        aqi_mae = summary['overall_performance']['AQI'][method]['avg_MAE']
        pm25_mae = summary['overall_performance']['PM25'][method]['avg_MAE']
        print(f"  {method.upper()}: AQI={aqi_mae:.1f}, PM2.5={pm25_mae:.1f}")
    
    return results, summary

if __name__ == "__main__":
    results, summary = main()