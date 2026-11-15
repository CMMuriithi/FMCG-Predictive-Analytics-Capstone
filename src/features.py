"""Panel building, feature engineering, time split, stationarity test."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from . import config

NUM_FEATURES = [
    "price", "price_change", "discount", "promo",
    "lag_1", "lag_7", "lag_30", "roll_mean_7", "roll_std_7",
    "lead_time_lag1", "stockout_lag1",
    "dow", "month", "weekofyear", "is_weekend", "is_month_start", "is_month_end",
    "holiday_share",
]


def build_panel(df, skus, by_channel=False):
    """Daily panel per SKU (optionally per SKU x channel). Missing days become 0 sales."""
    keys = ["sku"] + (["channel"] if by_channel else [])
    d = df[df["sku"].isin(skus)]
    agg = (d.groupby(["date"] + keys)
             .agg(volume=("volume", "sum"), revenue=("revenue", "sum"),
                  promo=("promo", "mean"), discount=("discount", "mean"),
                  lead_time=("lead_time", "mean"), stockout_rate=("stockout", "mean"))
             .reset_index())
    agg["price"] = np.where(agg["volume"] > 0, agg["revenue"] / agg["volume"].where(agg["volume"] > 0), np.nan)

    dates = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max())})
    full = agg[keys].drop_duplicates().merge(dates, how="cross")
    p = full.merge(agg, on=["date"] + keys, how="left")

    for c in ("volume", "revenue", "promo", "discount", "stockout_rate"):
        p[c] = p[c].fillna(0)
    for c in ("price", "lead_time"):
        p[c] = p.groupby(keys)[c].transform(lambda s: s.ffill().bfill())

    hol = df.groupby("date")["is_holiday"].mean().rename("holiday_share").reset_index()
    p = p.merge(hol, on="date", how="left")
    p["holiday_share"] = p["holiday_share"].fillna(0)
    return p.sort_values(keys + ["date"]).reset_index(drop=True)


def add_features(p, lags=(1, 7, 30)):
    """Lags, rolling stats and calendar flags. Everything is shifted, so no leakage."""
    keys = [c for c in ("sku", "channel") if c in p.columns]
    p = p.sort_values(keys + ["date"]).reset_index(drop=True)
    g = p.groupby(keys)

    for l in lags:
        p[f"lag_{l}"] = g["volume"].shift(l)
    p["roll_mean_7"] = g["volume"].transform(lambda s: s.shift(1).rolling(7).mean())
    p["roll_std_7"] = g["volume"].transform(lambda s: s.shift(1).rolling(7).std())
    p["lead_time_lag1"] = g["lead_time"].shift(1)
    p["stockout_lag1"] = g["stockout_rate"].shift(1)
    p["price_change"] = (g["price"].pct_change()
                         .replace([np.inf, -np.inf], np.nan).fillna(0))

    d = p["date"]
    p["dow"] = d.dt.dayofweek
    p["month"] = d.dt.month
    p["weekofyear"] = d.dt.isocalendar().week.astype(int)
    p["is_weekend"] = (p["dow"] >= 5).astype(int)
    p["is_month_start"] = d.dt.is_month_start.astype(int)
    p["is_month_end"] = d.dt.is_month_end.astype(int)
    return p.dropna(subset=NUM_FEATURES).reset_index(drop=True)


def time_split(p, months=config.HOLDOUT_MONTHS):
    """Chronological split: the last `months` months are the hold-out set."""
    cutoff = p["date"].max() - pd.DateOffset(months=months)
    train = p[p["date"] <= cutoff].sort_values("date")
    test = p[p["date"] > cutoff].sort_values("date")
    return train, test, cutoff


def adf_test(series, alpha=0.05):
    stat, pval, *_ = adfuller(series.dropna(), autolag="AIC")
    return {"adf_stat": round(float(stat), 3), "p_value": round(float(pval), 4),
            "stationary": bool(pval < alpha)}


def build_stockout_frame(df, skus):
    """Row-level (store x SKU x day) frame with lagged stock signals for the classifier."""
    r = df[df["sku"].isin(skus)].sort_values(["store", "sku", "date"]).copy()
    g = r.groupby(["store", "sku"])
    r["soh_lag1"] = g["stock_on_hand"].shift(1)
    r["units_lag1"] = g["volume"].shift(1)
    r["units_roll7"] = g["volume"].transform(lambda s: s.shift(1).rolling(7).mean())
    r["stockout_lag1"] = g["stockout"].shift(1)
    r["cover_days_lag1"] = r["soh_lag1"] / (r["units_roll7"] + 1)
    r["dow"] = r["date"].dt.dayofweek
    return r.dropna(subset=["soh_lag1", "units_roll7", "stockout_lag1"]).reset_index(drop=True)
