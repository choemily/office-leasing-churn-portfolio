
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', type=str, default='runs')
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--revenue_col', type=str, default='revenue_at_risk')
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    latest = sorted(runs_dir.glob('*/model.joblib'))
    if not latest:
        raise SystemExit('No trained models found. Run train.py first.')
    model_path = latest[-1]
    pipe = load(model_path)

    # Find the dataset used (assumes default path)
    data_path = Path('data/processed/tenant_month.csv')
    df = pd.read_csv(data_path, parse_dates=['ref_month','lease_start','lease_expiry'])

    # Simple holdout evaluation on last 20% months as proxy
    cutoff = df['ref_month'].quantile(0.8)
    df_eval = df[df['ref_month']>=cutoff].copy()
    y = df_eval['label'].values

    X = df_eval.drop(columns=['label','lease_start','lease_expiry'])
    y_proba = pipe.predict_proba(X)[:,1]

    # Top-K report
    idx = (-y_proba).argsort()[:args.k]
    out = df_eval.iloc[idx][['tenant_id','property_id','ref_month', args.revenue_col]].copy()
    out['churn_score'] = y_proba[idx]

    out_path = model_path.parent / f'top_{args.k}_risk_list_eval.csv'
    out.to_csv(out_path, index=False)
    print(f'Saved evaluation risk list to {out_path}')

if __name__ == '__main__':
    main()
