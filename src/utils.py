
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def precision_at_k(y_true, y_scores, k=50):
    idx = np.argsort(-y_scores)[:k]
    return float(np.mean(y_true[idx]))


def revenue_at_risk_topk(y_scores, revenue, k=50):
    idx = np.argsort(-y_scores)[:k]
    return float(np.sum(revenue[idx]))


def metrics_dict(y_true, y_proba, revenue=None, k=50):
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_at_k(y_true, y_proba, k=k)
    out = {'roc_auc': float(auc), f'precision@{k}': float(prec)}
    if revenue is not None:
        out[f'revenue_at_risk_top_{k}'] = float(revenue_at_risk_topk(y_proba, revenue, k))
    return out
