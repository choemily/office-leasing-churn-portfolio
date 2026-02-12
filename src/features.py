
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessor(numeric_cols, categorical_cols):
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median'))
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    pre = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ], remainder='drop')
    return pre


def split_xy(df: pd.DataFrame, label_col: str, drop_cols=None):
    drop_cols = drop_cols or []
    X = df.drop(columns=[label_col] + drop_cols)
    y = df[label_col].astype(int)
    return X, y
