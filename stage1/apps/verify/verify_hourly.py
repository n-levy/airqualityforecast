import argparse, pandas as pd, glob
from stage1_forecast.env import data_root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', required=True)
    args = ap.parse_args()
    city = args.city.lower()

    fpath = data_root() / 'forecasts' / 'ours' / city / 'pm25' / 'forecast.parquet'
    if not fpath.exists():
        print('[verify] forecast not found yet; skipping metrics.')
        raise SystemExit(0)
    df_fc = pd.read_parquet(fpath)

    root = data_root() / 'curated' / 'obs' / city / 'pm25'
    files = sorted(glob.glob(str(root / 'date=*/data.parquet')))
    if not files:
        print('[verify] no observations; skipping.')
        raise SystemExit(0)
    obs = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values('valid_time')

    m = df_fc.merge(obs[['valid_time','value']], on='valid_time', how='inner', suffixes=('_pred','_obs'))
    if m.empty:
        print('[verify] no overlap yet; metrics later.')
        raise SystemExit(0)

    mae = (m['yhat'] - m['value']).abs().mean()
    bias = (m['yhat'] - m['value']).mean()
    print(f'[verify] n={len(m)}  MAE={mae:.2f}  Bias={bias:.2f}')

if __name__ == '__main__':
    main()
