# FMCG Global Demand Planning and Forecasting (Capstone)

Time-series demand forecasting on 1.1M rows of daily FMCG sales (3 years, multi-country, store x SKU x day).

## Method
1. SQL retrieval (SQLite, chunked load, filtered/aggregated/joined queries) and cleaning
2. EDA, demand volatility (CV), geospatial map, lead time vs stockouts, ABC/XYZ segmentation
3. Hypothesis tests (Mann-Whitney U, Welch t, Wilcoxon) for promotional lift; price elasticity for A-SKUs
4. Feature engineering: lags 1/7/30, 7-day rolling mean/std, calendar flags; time-based split (last 6 months = test); scaler fit on train only
5. Forecasting: seasonal-naive baseline, Linear Regression, Random Forest, tuned LightGBM; MAE, RMSE, MAPE, WAPE, bias
6. Stockout classification (precision, recall, F1), feature importance and SHAP, bias audit by channel with mitigation
7. Safety-stock impact of forecast accuracy and recommendations

## Key results
See `outputs/headline.csv` and the final cells of the notebook (generated when you run it).

## How to run
1. `pip install -r requirements.txt`
2. Place `fmcg_sales_3years_1M_rows.csv` in the project root (not committed: 200 MB)
3. Open `fmcg_capstone.ipynb` and Run All. The first run builds `fmcg.db`.

## Structure
```
fmcg_capstone.ipynb   notebook (full story)
src/                  config, data_prep, segmentation, features, models, evaluate
outputs/              figures and result tables
```
