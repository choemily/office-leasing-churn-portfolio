
import argparse
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from joblib import dump
from datetime import datetime

from features import build_preprocessor, split_xy
from models import get_model
from utils import ensure_dir, save_json, metrics_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, choices=['logreg','lgbm'], required=True)
    ap.add_argument('--config', type=str, default='config/config.yaml')
    ap.add_argument('--data', type=str, default='data/processed/tenant_month.csv')
    ap.add_argument('--runs_dir', type=str, default='runs')
    ap.add_argument('--k', type=int, default=50)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(args.data, parse_dates=['ref_month','lease_start','lease_expiry'])
    label_col = cfg['data']['label_col']
    drop_cols = cfg['features']['drop']
    num_cols = cfg['features']['numeric']
    cat_cols = cfg['features']['categorical']

    # Train/valid/test split (stratified if configured)
    strat = df[label_col] if cfg['train'].get('stratify', True) else None
    df_train, df_test = train_test_split(df, test_size=cfg['train']['test_size'], random_state=cfg['random_seed'], stratify=strat)
    strat2 = df_train[label_col] if cfg['train'].get('stratify', True) else None
    df_train, df_valid = train_test_split(df_train, test_size=cfg['train']['valid_size'], random_state=cfg['random_seed'], stratify=strat2)

    X_train, y_train = split_xy(df_train, label_col, drop_cols=drop_cols)
    X_valid, y_valid = split_xy(df_valid, label_col, drop_cols=drop_cols)
    X_test, y_test = split_xy(df_test, label_col, drop_cols=drop_cols)

    pre = build_preprocessor(num_cols, cat_cols)
    model = get_model(args.model, cfg['models'][args.model])
    pipe = Pipeline([
        ('pre', pre),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)

    # Evaluate on validation
    import numpy as np
    y_val_proba = pipe.predict_proba(X_valid)[:,1]
    revenue_col = cfg['data']['revenue_col']
    revenue_valid = df_valid[revenue_col].values if revenue_col in df_valid.columns else None
    m_valid = metrics_dict(y_valid.values, y_val_proba, revenue=revenue_valid, k=args.k)

    # Evaluate on test
    y_test_proba = pipe.predict_proba(X_test)[:,1]
    revenue_test = df_test[revenue_col].values if revenue_col in df_test.columns else None
    m_test = metrics_dict(y_test.values, y_test_proba, revenue=revenue_test, k=args.k)

    # Save artifacts
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.runs_dir) / f"{args.model}_{ts}"
    ensure_dir(run_dir)

    dump(pipe, run_dir / 'model.joblib')

    # Save metrics
    save_json({'valid': m_valid, 'test': m_test}, run_dir / 'metrics.json')

    # Save top-K ranking on test
    topk = args.k
    idx = (-y_test_proba).argsort()[:topk]
    rank_df = df_test.iloc[idx][['tenant_id','property_id','ref_month', revenue_col]].copy()
    rank_df['churn_score'] = y_test_proba[idx]
    rank_df.to_csv(run_dir / f'top_{topk}_risk_list.csv', index=False)

    print(f"Saved run to {run_dir}")
    print('Validation metrics:', m_valid)
    print('Test metrics:', m_test)

if __name__ == '__main__':
    main()
