import argparse, pandas as pd, glob
from stage1_forecast.env import data_root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', required=True)
    args = ap.parse_args()
    city = args.city.lower()

    root = data_root() / 'curated' / 'obs' / city / 'pm25'
    files = sorted(glob.glob(str(root / 'date=*/data.parquet')))[-3:]
    obs = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else None

    fc_path = data_root() / 'forecasts' / 'ours' / city / 'pm25' / 'forecast.parquet'
    fc = pd.read_parquet(fc_path) if fc_path.exists() else None

    dest = data_root() / 'exports' / city
    dest.mkdir(parents=True, exist_ok=True)

    if obs is not None and not obs.empty:
        obs.to_csv(dest / 'obs_pm25_recent.csv', index=False)
        obs.to_json(dest / 'obs_pm25_recent.json', orient='records', indent=2, date_format='iso')
        print('[publish] wrote obs_pm25_recent.csv/json')

    if fc is not None and not fc.empty:
        fc.to_csv(dest / 'forecast_pm25.csv', index=False)
        fc.to_json(dest / 'forecast_pm25.json', orient='records', indent=2, date_format='iso')
        print('[publish] wrote forecast_pm25.csv/json')

if __name__ == '__main__':
    main()
