"""ABC / XYZ segmentation and SKU selection."""
import numpy as np
import pandas as pd
from . import config


def abc_analysis(df, key_col, revenue_col="revenue"):
    """A = first 80% of revenue, B = next 15%, C = last 5%."""
    out = (df.groupby(key_col)[revenue_col].sum()
             .sort_values(ascending=False).rename("revenue").reset_index())
    out["share"] = out["revenue"] / out["revenue"].sum()
    out["cum_share"] = out["share"].cumsum()
    prev = out["cum_share"] - out["share"]
    out["abc"] = np.select([prev < 0.80, prev < 0.95], ["A", "B"], "C")
    return out


def abc_by_region(df, region_col, key_col, revenue_col="revenue"):
    parts = []
    for region, g in df.groupby(region_col):
        a = abc_analysis(g, key_col, revenue_col)
        a[region_col] = region
        parts.append(a)
    return pd.concat(parts, ignore_index=True)


def xyz_analysis(ts, key_col, value_col, x=0.5, y=1.0, min_per_class=3):
    """X stable / Y variable / Z erratic from the coefficient of variation.
    Uses the fixed cut-offs (x, y); if one class would be nearly empty, falls back to
    CV terciles so the matrix stays informative. Returns (table, info)."""
    g = ts.groupby(key_col)[value_col].agg(mean="mean", std="std")
    g["cv"] = g["std"] / g["mean"]

    def label(cv, a, b):
        return np.select([cv <= a, cv <= b], ["X", "Y"], "Z")

    g["xyz"] = label(g["cv"], x, y)
    counts = g["xyz"].value_counts().reindex(["X", "Y", "Z"], fill_value=0)
    info = {"method": "fixed", "x_cut": x, "y_cut": y}
    if counts.min() < min_per_class:
        a, b = g["cv"].quantile([1 / 3, 2 / 3])
        g["xyz"] = label(g["cv"], a, b)
        info = {"method": "terciles", "x_cut": round(a, 3), "y_cut": round(b, 3)}
    return g.reset_index(), info


def select_skus(seg, n_total=10, plan=None, min_obs=500):
    """Pick SKUs across diverse demand profiles (top revenue inside each segment)."""
    plan = plan or {"AX": 2, "AY": 2, "BY": 2, "CZ": 2, "BX": 1, "CY": 1, "AZ": 1, "BZ": 1}
    ok = seg[seg["n_obs"] >= min_obs]
    chosen = []
    for cls, n in plan.items():
        pool = ok[(ok["segment"] == cls) & (~ok["sku"].isin(chosen))]
        chosen += pool.sort_values("revenue", ascending=False).head(n)["sku"].tolist()
    chosen = chosen[:n_total]
    if len(chosen) < n_total:   # fill gaps, preferring segments not yet represented
        rest = ok[~ok["sku"].isin(chosen)].copy()
        rest["new_seg"] = ~rest["segment"].isin(seg[seg["sku"].isin(chosen)]["segment"])
        rest = rest.sort_values(["new_seg", "revenue"], ascending=False)
        chosen += rest.head(n_total - len(chosen))["sku"].tolist()
    return seg[seg["sku"].isin(chosen)].sort_values(["segment", "revenue"],
                                                   ascending=[True, False]).reset_index(drop=True)


def pick_small_format_channels(df):
    """Find the local / small-format channel for the bias audit. Returns (list, reason)."""
    chans = sorted(df["channel"].unique())
    hit = [c for c in chans if any(k in c.lower() for k in config.SMALL_FORMAT_KEYWORDS)]
    if hit:
        return hit, "name matches a small-format keyword"
    per_row = df.groupby("channel")["volume"].mean().sort_values()
    return [per_row.index[0]], "no keyword match: channel with the lowest average units per store-day"
