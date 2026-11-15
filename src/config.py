"""Project-wide paths, constants and the column map for the FMCG capstone."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "fmcg_sales_3years_1M_rows.csv"
DB_PATH = ROOT / "fmcg.db"
CLEAN_PATH = ROOT / "clean_sales.parquet"
TABLE = "sales"

OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
HOLDOUT_MONTHS = 6          # last 6 months = hold-out test set
SERVICE_Z = 1.65            # z-score for a 95% cycle service level

# canonical name used in the code  ->  real column name in the CSV / SQLite table
COLS = {
    "date": "date",
    "store": "store_id",
    "sku": "sku_id",
    "sku_name": "sku_name",
    "category": "category",
    "country": "country",
    "city": "city",
    "channel": "channel",
    "lat": "latitude",
    "lon": "longitude",
    "volume": "units_sold",
    "list_price": "list_price",
    "discount": "discount_pct",
    "promo": "promo_flag",
    "revenue": "net_sales",
    "stock_on_hand": "stock_on_hand",
    "stockout": "stock_out_flag",
    "lead_time": "lead_time_days",
    "is_holiday": "is_holiday",
    "is_weekend": "is_weekend",
}

# lower-case variants -> unified label (anything not listed is just Title-Cased)
CHANNEL_MAP = {
    "e-comm": "E-commerce", "ecomm": "E-commerce", "e-commerce": "E-commerce",
    "ecommerce": "E-commerce", "e commerce": "E-commerce", "online": "E-commerce",
    "modern trade": "Modern Trade", "traditional trade": "Traditional Trade",
    "wholesale": "Wholesale",
}

# used to find the small-format / local-retailer channel for the bias audit
SMALL_FORMAT_KEYWORDS = ("traditional", "kiosk", "duka", "convenience",
                         "mini", "corner", "small", "general trade")
