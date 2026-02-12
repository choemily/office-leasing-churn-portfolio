
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


def get_model(name: str, params: dict):
    name = name.lower()
    if name == 'logreg':
        return LogisticRegression(**params)
    elif name in ('lgbm','lightgbm','gbm'):
        return LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {name}")
