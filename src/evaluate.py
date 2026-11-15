"""Forecast and classification metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error, precision_score,
                             recall_score, f1_score, average_precision_score, roc_auc_score)


def regression_metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = y > 0
    return {
        "MAE": mean_absolute_error(y, yhat),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "MAPE_%": float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100),
        "WAPE_%": float(np.abs(y - yhat).sum() / y.sum() * 100),
        "Bias_%": float((yhat - y).sum() / y.sum() * 100),   # negative = under-forecast
    }


def group_metrics(df, actual, pred, group):
    rows = []
    for key, g in df.groupby(group):
        rows.append({group: key, **regression_metrics(g[actual], g[pred])})
    return pd.DataFrame(rows)


def classification_summary(y, pred, proba):
    return {
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "PR_AUC": average_precision_score(y, proba),
        "ROC_AUC": roc_auc_score(y, proba),
    }
