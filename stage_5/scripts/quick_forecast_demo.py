#!/usr/bin/env python3
"""
Quick Forecasting Demo - Sample of 20 Cities

This script provides a quick demonstration of walk-forward forecasting
on a representative sample of 20 cities (4 from each continent).
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

class QuickForecastDemo:
    """Quick forecasting demo for representative cities."""
    
    def __init__(self, data_path: str = "../comprehensive_tables/comprehensive_features_table.csv"):
        """Initialize with dataset path."""
        self.data_path = data_path
        
        # Sample cities (4 per continent - highest AQI)
        self.sample_cities = [
            # Asia (highest AQI)
            'Delhi', 'Lahore', 'Dhaka', 'Kolkata',
            # Africa (highest AQI) 
            'Cairo', 'Khartoum', 'Giza', "N'Djamena",
            # Europe (highest AQI)
            'Skopje', 'Sarajevo', 'Tuzla', 'Zenica',
            # North America (representative)
            'Los Angeles', 'Phoenix', 'Houston', 'Mexico City',
            # South America (representative)
            'São Paulo', 'Lima', 'Santiago', 'Bogotá'
        ]
        
    def load_sample_cities(self) -> pd.DataFrame:
        """Load data for sample cities."""
        logger.info("Loading sample cities dataset...")
        
        features_df = pd.read_csv(self.data_path)
        
        # Filter for sample cities (allowing for name variations)
        sample_df = features_df[features_df['City'].isin(self.sample_cities)].copy()
        
        # If we don't have exact matches, get top cities by AQI from each continent
        if len(sample_df) < 15:  # Allow some flexibility
            by_continent = features_df.groupby('Continent').apply(
                lambda x: x.nlargest(4, 'Average_AQI')
            ).reset_index(drop=True)
            sample_df = by_continent
        
        logger.info(f"Loaded {len(sample_df)} sample cities")
        return sample_df
    
    def generate_time_series(self, city_row: pd.Series, days: int = 90) -> pd.DataFrame:
        """Generate synthetic time series (shorter for demo)."""
        
        base_pm25 = city_row['Average_PM25']
        base_pm10 = city_row['pm10_Concentration']
        base_no2 = city_row['no2_Concentration'] 
        base_o3 = city_row['o3_Concentration']
        base_so2 = city_row['so2_Concentration']
        base_co = city_row['co_Concentration']
        
        # Generate date range (last 90 days)
        end_date = datetime(2025, 9, 11)
        start_date = end_date - timedelta(days=days-1)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Consistent seed per city
        np.random.seed(hash(city_row['City']) % 2**32)
        
        data = []
        for i, date in enumerate(dates):
            # Simplified patterns
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            weekly_factor = 1.1 if date.weekday() < 5 else 0.9
            noise_factor = np.random.normal(1.0, 0.1)
            
            total_factor = seasonal_factor * weekly_factor * noise_factor
            
            # Generate pollutant values
            pm25_actual = max(1, base_pm25 * total_factor)
            pm10_actual = max(1, base_pm10 * total_factor * 1.2)
            no2_actual = max(1, base_no2 * total_factor * weekly_factor)
            o3_actual = max(1, base_o3 * seasonal_factor * np.random.normal(1.0, 0.15))
            so2_actual = max(1, base_so2 * total_factor * 0.9)
            co_actual = max(1, base_co * total_factor * weekly_factor)
            
            # Generate benchmark forecasts with realistic errors
            cams_pm25 = pm25_actual * np.random.normal(1.0, 0.12)
            cams_pm10 = pm10_actual * np.random.normal(1.0, 0.15)
            cams_no2 = no2_actual * np.random.normal(1.0, 0.18)
            cams_o3 = o3_actual * np.random.normal(1.0, 0.14)
            cams_so2 = so2_actual * np.random.normal(1.0, 0.20)
            cams_co = co_actual * np.random.normal(1.0, 0.16)
            
            noaa_pm25 = pm25_actual * np.random.normal(1.0, 0.14)
            noaa_pm10 = pm10_actual * np.random.normal(1.0, 0.17)
            noaa_no2 = no2_actual * np.random.normal(1.0, 0.16)
            noaa_o3 = o3_actual * np.random.normal(1.0, 0.12)
            noaa_so2 = so2_actual * np.random.normal(1.0, 0.22)
            noaa_co = co_actual * np.random.normal(1.0, 0.18)
            
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
                'temperature': 20 + 15 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365) + np.random.normal(0, 3),
                'humidity': max(10, min(100, 60 + np.random.normal(0, 15))),
                'wind_speed': max(0, 3 + np.random.normal(0, 2)),
                'pressure': 1013 + np.random.normal(0, 10),
                'day_of_year': date.timetuple().tm_yday,
                'day_of_week': date.weekday(),
                'is_weekend': float(date.weekday() >= 5)
            })
        
        return pd.DataFrame(data)
    
    def calculate_aqi(self, pm25: float, pm10: float, no2: float, o3: float, 
                     so2: float, co: float) -> float:
        """Simplified AQI calculation."""
        # US EPA simplified
        aqi_pm25 = min(500, max(0, pm25 * 4.17))
        aqi_pm10 = min(500, max(0, pm10 * 2.04))
        aqi_no2 = min(500, max(0, no2 * 9.43))
        aqi_o3 = min(500, max(0, o3 * 7.81))
        aqi_so2 = min(500, max(0, so2 * 9.17))
        aqi_co = min(500, max(0, co * 0.115))
        
        return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_o3, aqi_so2, aqi_co)
    
    def evaluate_forecasts(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted) if len(set(actual)) > 1 else 0
        mpe = np.mean((predicted - actual) / (actual + 1e-8)) * 100  # Avoid division by zero
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MPE': mpe
        }
    
    def walk_forward_city(self, city_data: pd.DataFrame, city_name: str) -> Dict:
        """Walk-forward validation for single city."""
        
        city_data = city_data.sort_values('date').reset_index(drop=True)
        
        pollutants = ['PM25', 'PM10', 'NO2', 'O3', 'SO2', 'CO']
        predictions = {
            'simple_avg': {p: [] for p in pollutants + ['AQI']},
            'ridge': {p: [] for p in pollutants + ['AQI']},
            'cams': {p: [] for p in pollutants + ['AQI']},
            'noaa': {p: [] for p in pollutants + ['AQI']},
            'actual': {p: [] for p in pollutants + ['AQI']}
        }
        
        # Walk-forward validation (last 30 days)
        for i in range(max(10, len(city_data)-30), len(city_data)):
            
            train_data = city_data.iloc[:i]
            test_data = city_data.iloc[i]
            
            for pollutant in pollutants:
                # Simple average
                simple_pred = (test_data[f'CAMS_{pollutant}'] + test_data[f'NOAA_{pollutant}']) / 2
                
                # Ridge regression with basic features
                try:
                    if len(train_data) > 5:
                        X_train = train_data[['temperature', 'humidity', 'wind_speed', 'pressure', 
                                            f'CAMS_{pollutant}', f'NOAA_{pollutant}']].values
                        y_train = train_data[f'{pollutant}_actual'].values
                        
                        X_test = np.array([test_data['temperature'], test_data['humidity'], 
                                         test_data['wind_speed'], test_data['pressure'],
                                         test_data[f'CAMS_{pollutant}'], test_data[f'NOAA_{pollutant}']])
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test.reshape(1, -1))
                        
                        ridge = Ridge(alpha=1.0)
                        ridge.fit(X_train_scaled, y_train)
                        ridge_pred = ridge.predict(X_test_scaled)[0]
                    else:
                        ridge_pred = simple_pred
                except:
                    ridge_pred = simple_pred
                
                # Store predictions
                predictions['simple_avg'][pollutant].append(simple_pred)
                predictions['ridge'][pollutant].append(ridge_pred)
                predictions['cams'][pollutant].append(test_data[f'CAMS_{pollutant}'])
                predictions['noaa'][pollutant].append(test_data[f'NOAA_{pollutant}'])
                predictions['actual'][pollutant].append(test_data[f'{pollutant}_actual'])
        
        # Calculate AQI
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
        
        # Evaluate performance
        results = {}
        for pollutant in pollutants + ['AQI']:
            results[pollutant] = {}
            actual_values = np.array(predictions['actual'][pollutant])
            
            for method in ['simple_avg', 'ridge', 'cams', 'noaa']:
                pred_values = np.array(predictions[method][pollutant])
                results[pollutant][method] = self.evaluate_forecasts(actual_values, pred_values)
        
        return results
    
    def process_sample_cities(self) -> Dict:
        """Process sample cities."""
        
        cities_df = self.load_sample_cities()
        all_results = {}
        
        for idx, city_row in cities_df.iterrows():
            city_name = city_row['City']
            
            try:
                logger.info(f"Processing {city_name}...")
                
                # Generate time series
                city_time_series = self.generate_time_series(city_row)
                
                # Walk-forward validation
                city_results = self.walk_forward_city(city_time_series, city_name)
                
                all_results[city_name] = {
                    'country': city_row['Country'],
                    'continent': city_row['Continent'],
                    'avg_aqi': city_row['Average_AQI'],
                    'results': city_results
                }
                
                logger.info(f"Completed {city_name} ({idx+1}/{len(cities_df)})")
                
            except Exception as e:
                logger.error(f"Error processing {city_name}: {str(e)}")
                continue
        
        return all_results
    
    def generate_summary(self, results: Dict) -> Dict:
        """Generate summary report."""
        
        summary = {
            'total_cities': len(results),
            'pollutants': ['PM25', 'PM10', 'NO2', 'O3', 'SO2', 'CO', 'AQI'],
            'methods': ['simple_avg', 'ridge', 'cams', 'noaa'],
            'overall_performance': {},
            'improvement_over_benchmarks': {},
            'best_method_by_pollutant': {}
        }
        
        # Calculate overall performance
        for pollutant in summary['pollutants']:
            summary['overall_performance'][pollutant] = {}
            
            for method in summary['methods']:
                mae_values = []
                r2_values = []
                
                for city_name, city_data in results.items():
                    if pollutant in city_data['results']:
                        mae_values.append(city_data['results'][pollutant][method]['MAE'])
                        r2_values.append(city_data['results'][pollutant][method]['R2'])
                
                summary['overall_performance'][pollutant][method] = {
                    'avg_MAE': np.mean(mae_values) if mae_values else 0,
                    'avg_R2': np.mean(r2_values) if r2_values else 0,
                    'cities': len(mae_values)
                }
            
            # Find best method
            best_method = min(summary['methods'], 
                            key=lambda m: summary['overall_performance'][pollutant][m]['avg_MAE'])
            summary['best_method_by_pollutant'][pollutant] = best_method
            
            # Calculate improvement over benchmarks
            ensemble_mae = min(summary['overall_performance'][pollutant]['simple_avg']['avg_MAE'],
                             summary['overall_performance'][pollutant]['ridge']['avg_MAE'])
            cams_mae = summary['overall_performance'][pollutant]['cams']['avg_MAE']
            noaa_mae = summary['overall_performance'][pollutant]['noaa']['avg_MAE']
            
            best_baseline = min(cams_mae, noaa_mae)
            improvement = ((best_baseline - ensemble_mae) / best_baseline) * 100 if best_baseline > 0 else 0
            
            summary['improvement_over_benchmarks'][pollutant] = {
                'best_ensemble_mae': ensemble_mae,
                'best_baseline_mae': best_baseline,
                'improvement_percent': improvement
            }
        
        return summary

def main():
    """Main execution."""
    
    logger.info("Starting Quick Forecasting Demo")
    
    # Initialize demo
    demo = QuickForecastDemo()
    
    # Process sample cities
    results = demo.process_sample_cities()
    
    # Generate summary
    summary = demo.generate_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'quick_forecast_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(f'quick_forecast_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print results
    print("\n" + "="*60)
    print("QUICK FORECASTING DEMO RESULTS")
    print("="*60)
    
    print(f"\nProcessed {summary['total_cities']} representative cities")
    
    print("\nBest Performing Method by Pollutant:")
    for pollutant, method in summary['best_method_by_pollutant'].items():
        mae = summary['overall_performance'][pollutant][method]['avg_MAE']
        print(f"  {pollutant}: {method.upper()} (MAE: {mae:.2f})")
    
    print("\nPerformance Improvement Over Benchmarks:")
    for pollutant, improvement in summary['improvement_over_benchmarks'].items():
        print(f"  {pollutant}: {improvement['improvement_percent']:.1f}% improvement")
    
    print("\nOverall Performance (Average MAE):")
    for method in summary['methods']:
        aqi_perf = summary['overall_performance']['AQI'][method]
        pm25_perf = summary['overall_performance']['PM25'][method]
        print(f"  {method.upper()}: AQI={aqi_perf['avg_MAE']:.1f} (R²={aqi_perf['avg_R2']:.3f}), "
              f"PM2.5={pm25_perf['avg_MAE']:.1f} (R²={pm25_perf['avg_R2']:.3f})")
    
    return results, summary

if __name__ == "__main__":
    results, summary = main()