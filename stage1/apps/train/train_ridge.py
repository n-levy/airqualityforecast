import argparse, pandas as pd
from sklearn.linear_model import RidgeCV
from joblib import dump
from stage1_forecast.env import data_root, models_root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', required=True)
    args = ap.parse_args()
    city = args.city.lower()
    df = pd.read_parquet(data_root() / 'features' / city / 'pm25' / 'features.parquet')
    y = df['y'].values
    X = df.drop(columns=['y','valid_time']).values
    model = RidgeCV(alphas=[0.1,1.0,3.0,10.0], cv=5).fit(X,y)
    dest = models_root() / 'ridge' / city / 'pm25'
    dest.mkdir(parents=True, exist_ok=True)
    dump(model, dest / 'model.joblib')
    print(f'[train] saved ridge model -> {dest}')

if __name__ == '__main__':
    main()
