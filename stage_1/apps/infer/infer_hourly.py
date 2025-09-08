import argparse, pandas as pd
from joblib import load
from stage1_forecast.env import data_root, models_root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', required=True)
    ap.add_argument('--hours', type=int, default=24)
    args = ap.parse_args()
    city = args.city.lower()
    df = pd.read_parquet(data_root() / 'features' / city / 'pm25' / 'features.parquet').sort_values('valid_time').reset_index(drop=True)
    last_time = df['valid_time'].iloc[-1]
    model = load(models_root() / 'ridge' / city / 'pm25' / 'model.joblib')
    hist = df.copy()
    preds = []
    cur_time = last_time
    for h in range(1, args.hours+1):
        cur_time = cur_time + pd.Timedelta(hours=1)
        s = hist.set_index('valid_time')['y']
        row = {'valid_time': cur_time}
        for k in [1,2,3,6,12,24,48]:
            row[f'lag_{k}'] = s.iloc[-k]
        row['roll_6']  = s.iloc[-6:].mean()
        row['roll_24'] = s.iloc[-24:].mean()
        row['hour'] = cur_time.hour
        row['dow']  = cur_time.dayofweek
        X = pd.DataFrame([row]).drop(columns=['valid_time']).values
        yhat = float(model.predict(X)[0])
        preds.append({'valid_time': cur_time, 'yhat': yhat})
        new_row = row.copy(); new_row['y'] = yhat
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    out = pd.DataFrame(preds)
    dest = data_root() / 'forecasts' / 'ours' / city / 'pm25'
    dest.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest / 'forecast.parquet', index=False)
    print(f'[infer] wrote {len(out)} forecast rows -> {dest}')

if __name__ == '__main__':
    main()
