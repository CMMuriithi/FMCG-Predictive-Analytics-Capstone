"""Model matrices (scaling + encoding) and model zoo."""
import re
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from . import config


def _clean_cols(cols):
    return [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in cols]


def make_matrices(train, test, num_cols, cat_cols):
    """One-hot encode categoricals; scale numeric columns with a TRAIN-ONLY scaler.
    Returns raw and scaled versions (trees use raw, linear models use scaled)."""
    X_tr = pd.get_dummies(train[num_cols + cat_cols], columns=cat_cols, dtype=int)
    X_te = pd.get_dummies(test[num_cols + cat_cols], columns=cat_cols, dtype=int)
    X_tr.columns = _clean_cols(X_tr.columns)
    X_te.columns = _clean_cols(X_te.columns)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    scaler = StandardScaler().fit(X_tr[num_cols])            # fit on train only
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    X_tr_s[num_cols] = scaler.transform(X_tr[num_cols])
    X_te_s[num_cols] = scaler.transform(X_te[num_cols])
    return X_tr, X_te, X_tr_s, X_te_s, scaler


def get_models():
    rs = config.RANDOM_STATE
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=rs),
        "LightGBM": LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, importance_type="gain",
            random_state=rs, verbose=-1),
    }
