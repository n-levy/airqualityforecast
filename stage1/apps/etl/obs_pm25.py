import argparse, pandas as pd
from pathlib import Path
from stage1_forecast.env import data_root
from apps.etl.providers import fetch_pm25_city

EXPECTED = ['city','valid_time','value','unit']

def _fetch(city:str, hours:int, provider:str) -> pd.DataFrame:
    raw = fetch_pm25_city(city=city.title(), hours=hours, prefer=provider)
    if not raw: raise SystemExit(f'No PM2.5 data for {city} in last {hours}h (provider={provider}).')
    df = pd.json_normalize(raw)
    ts = None
    for cand in ['date.utc','datetime','datetime.utc']:
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], utc=True, errors='coerce')
            break
    if ts is None and 'datetime.utc' in df.columns:
        ts = pd.to_datetime(df['datetime.utc'], utc=True, errors='coerce')
    if ts is None:
        raise SystemExit('Missing timestamp field (date.utc/datetime).')
    out = pd.DataFrame({
        'city': city.lower(),
        'valid_time': ts.dt.floor('h'),
        'value': pd.to_numeric(df['value'], errors='coerce'),
        'unit': df.get('unit','µg/m³')
    }).dropna(subset=['valid_time','value'])
    out = out.groupby(['city','valid_time','unit'], as_index=False).agg({'value':'mean'})
    return out[EXPECTED]

def _write(df: pd.DataFrame, city:str) -> None:
    dr = data_root()
    root = dr / 'curated' / 'obs' / city / 'pm25'
    if df['valid_time'].dt.tz is None:
        df['valid_time'] = df['valid_time'].dt.tz_localize('UTC')
    else:
        df['valid_time'] = df['valid_time'].dt.tz_convert('UTC')
    df['__date'] = df['valid_time'].dt.date.astype('string')
    for date_str, group in df.groupby('__date', sort=True):
        dest = root / f'date={date_str}'
        dest.mkdir(parents=True, exist_ok=True)
        (group.drop(columns=['__date']).to_parquet(dest / 'data.parquet', index=False))
        print(f'[obs] wrote {len(group)} rows -> {dest}')

def main():
    ap = argparse.ArgumentParser(description='Fetch PM2.5 and write curated parquet')
    ap.add_argument('--city', required=True, help='berlin|hamburg|munich')
    ap.add_argument('--hours', type=int, default=168)
    ap.add_argument('--provider', choices=['auto','openaq','openmeteo'], default='auto')
    args = ap.parse_args()
    city = args.city.lower()
    df = _fetch(city, args.hours, args.provider)
    _write(df, city)
    print('[obs] done.')

if __name__ == '__main__':
    main()
