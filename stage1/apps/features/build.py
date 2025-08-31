import argparse, pandas as pd, numpy as np, glob
from pathlib import Path
from stage1_forecast.env import data_root

def load_obs(city:str) -> pd.DataFrame:
    root = data_root() / 'curated' / 'obs' / city / 'pm25'
    files = sorted(glob.glob(str(root / 'date=*/data.parquet')))
    if not files: raise SystemExit(f'No observations found under {root}')
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('valid_time').copy()
    df['valid_time'] = pd.to_datetime(df['valid_time'], utc=True)
    idx = pd.date_range(df['valid_time'].min(), df['valid_time'].max(), freq='H', tz='UTC')
    s = df.set_index('valid_time')['value'].reindex(idx).interpolate(limit=3).ffill().bfill()
    feat = pd.DataFrame({'valid_time': idx, 'y': s.values})
    for k in [1,2,3,6,12,24,48]:
        feat[f'lag_{k}'] = s.shift(k).values
    feat['roll_6'] = s.rolling(6).mean().values
    feat['roll_24'] = s.rolling(24).mean().values
    feat['hour'] = feat['valid_time'].dt.hour
    feat['dow']  = feat['valid_time'].dt.dayofweek
    return feat.dropna().reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', required=True)
    args = ap.parse_args()
    city = args.city.lower()
    obs = load_obs(city)
    feat = build_features(obs)
    dest = data_root() / 'features' / city / 'pm25'
    dest.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(dest / 'features.parquet', index=False)
    print(f'[features] wrote {len(feat)} rows -> {dest}')

if __name__ == '__main__':
    main()
