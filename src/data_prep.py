"""SQL loading, cleaning, standardising and merging."""
import sqlite3
import pandas as pd
from . import config


# SQL layer
def build_database(force=False, chunksize=200_000):
    """Stream the CSV into SQLite in chunks (low memory) and index it.
    Skipped automatically if the database already exists (use force=True to rebuild)."""
    db = config.DB_PATH
    if db.exists() and not force:
        return db
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    try:
        for chunk in pd.read_csv(config.CSV_PATH, chunksize=chunksize):
            chunk.to_sql(config.TABLE, con, if_exists="append", index=False)
        for real in (config.COLS["date"], config.COLS["sku"]):
            con.execute(f'CREATE INDEX IF NOT EXISTS idx_{real} ON {config.TABLE}("{real}")')
        con.commit()
    finally:
        con.close()
    return db


def run_query(sql, db_path=None):
    con = sqlite3.connect(db_path or config.DB_PATH)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def check_columns():
    """Fail loudly if config.COLS names a column that is not in the table.
    (SQLite silently turns an unknown "double-quoted" name into a text literal.)"""
    con = sqlite3.connect(config.DB_PATH)
    try:
        have = [r[1] for r in con.execute(f"PRAGMA table_info({config.TABLE})")]
    finally:
        con.close()
    missing = {k: v for k, v in config.COLS.items() if v not in have}
    if missing:
        raise KeyError(f"Columns missing from table: {missing}. Available: {have}")
    return have


def load_sales(where=""):
    """Select only the needed columns and return them under canonical names."""
    check_columns()
    select = ", ".join(f'"{real}" AS "{canon}"' for canon, real in config.COLS.items())
    return run_query(f"SELECT {select} FROM {config.TABLE} {where}")


# cleaning layer
def _to_flag(s):
    """Yes/No/True/False/1/0/NaN -> clean 0/1 integer."""
    if not pd.api.types.is_numeric_dtype(s):
        s = (s.astype("string").str.strip().str.lower()
             .map({"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}))
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1).astype(int)


def standardise_labels(df):
    for col in ("country", "city", "channel"):
        df[col] = (df[col].astype("string").str.strip()
                   .str.replace(r"\s+", " ", regex=True))
    df["country"] = df["country"].str.title()
    df["city"] = df["city"].str.title()
    mapped = df["channel"].str.lower().map(config.CHANNEL_MAP)
    df["channel"] = mapped.fillna(df["channel"].str.title())
    for col in ("country", "city", "channel"):
        df[col] = df[col].astype(object)
    return df


def clean(df):
    """Nulls, labels, invalid values, duplicates. Returns (clean_df, report)."""
    df = df.copy()
    rep = {"rows_in": len(df)}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    core = ["date", "store", "sku", "volume", "list_price"]
    rep["rows_dropped_missing_core"] = int(df[core].isna().any(axis=1).sum())
    df = df.dropna(subset=core)

    df = standardise_labels(df)

    # coordinates: fill from the city's own median, never from a global value
    for k in ("lat", "lon"):
        df[k] = df[k].fillna(df.groupby("city")[k].transform("median"))

    # lead time: median imputation (robust to outliers): country first, then global
    df["lead_time"] = df["lead_time"].fillna(
        df.groupby("country")["lead_time"].transform("median"))
    df["lead_time"] = df["lead_time"].fillna(df["lead_time"].median())

    for k in ("promo", "stockout", "is_holiday", "is_weekend"):
        df[k] = _to_flag(df[k])
    df["discount"] = df["discount"].fillna(0).clip(0, 1)
    df["stock_on_hand"] = df["stock_on_hand"].fillna(
        df.groupby("sku")["stock_on_hand"].transform("median")).fillna(0)

    bad = (df["volume"] < 0) | (df["list_price"] <= 0)
    rep["rows_dropped_invalid"] = int(bad.sum())
    df = df[~bad].copy()

    # realised (net) price per unit, and revenue
    df["price"] = df["list_price"] * (1 - df["discount"])
    df["revenue"] = df["revenue"].fillna(df["volume"] * df["price"])

    # true grain of this dataset = store x SKU x day
    rep["duplicates_removed"] = int(df.duplicated(["date", "store", "sku"]).sum())
    df = df.drop_duplicates(["date", "store", "sku"], keep="first")
    # the guide's key (SKU-Date-Channel-City) repeats when a city has several stores
    rep["rows_sharing_guide_key"] = int(df.duplicated(["date", "sku", "channel", "city"]).sum())

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    rep["rows_out"] = len(df)
    return df.reset_index(drop=True), rep


#  merging layer
def build_promo_calendar(df):
    """Promotional calendar at date x SKU x country level (built from the data itself)."""
    cal = (df.groupby(["date", "sku", "country"])
             .agg(promo_active_country=("promo", "max"),
                  avg_discount_country=("discount", "mean"))
             .reset_index())
    return cal


def merge_promo(df, calendar):
    """LEFT join: every sales row keeps its place and is flagged for promo activity."""
    out = df.merge(calendar, on=["date", "sku", "country"], how="left")
    out["promo_active_country"] = out["promo_active_country"].fillna(0).astype(int)
    out["avg_discount_country"] = out["avg_discount_country"].fillna(0.0)
    return out
