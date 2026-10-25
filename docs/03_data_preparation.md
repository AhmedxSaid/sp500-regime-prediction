# 3. Data Preparation

Pipeline:

```
data/raw/{gspc,sp500tr,spy,es}.csv   (committed snapshot; refresh with data_download.py)
data_prep.py      ->  data/processed/sp500_daily.parquet   (+ reports/data_quality.csv)
features.py       ->  data/processed/model_table.parquet   (features + labels + targets)
```

## Step 1 — Clean and align (`data_prep.py`)

- Remove duplicate dates and sort.
- Set non-positive prices to NaN; set rows where open = high = low = close to NaN for O/H/L (pre-1962 index data).
- Set zero volume to NaN.
- Align all series on the `^GSPC` trading calendar. Columns are prefixed (`gspc_close`, `spy_volume`, `es_close`, ...).
- Forward-fill **prices** for at most 2 days. Volume is not filled.
- Flag ES futures roll windows (`es_roll_window = 1`).

## Step 2 — Regime labels (`labels.py`)

### Bull / bear

Two definitions are implemented:

| Method | Bear period | Causal? | Use |
|---|---|---|---|
| `peak_to_trough` | From the peak before a ≥20% fall until the trough (the bear ends once price rallies 20% off the trough) | **No.** The peak and trough are only known later | Classic dating; use as a **label** only |
| `recovery` | From the day price closes 20% below the peak until it gets back to that peak | **Yes.** Known at day *t* | Label variant, and a "current state" feature / baseline |

By default the labels use `^SP500TR` (includes dividends), with `^GSPC` as a fallback.

### Calm / stressed

- 21-day realized volatility: `std(log returns over 21 days) × √252`.
- Threshold: the **expanding** 80th percentile of past realized volatility, shifted by one day. This uses only past data and needs 5 years of history before the first label.
- `stressed = 1` if the current realized volatility is above the threshold.

### Sanity check

Plot the labels against price and check that they catch the known bear markets: 2000–02, 2007–09, 2020 and 2022. This is done in the EDA notebook.

## Step 3 — Prediction target

`target_{label}_h{h} = label.shift(-h)` for h ∈ {5, 21}.

We predict the regime **h days ahead**, not today's regime. Labels are shifted forward; features are never shifted backward. The last *h* rows have no target and are dropped at training time.

## Step 4 — Features (`features.py`)

All features use only data available by the close of day *t*.

| Group | Features |
|---|---|
| Returns / momentum | `ret_1d`, `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m` (log returns over 21/63/126/252 days) |
| Trend | `dist_ma50`, `dist_ma200` (price / moving average − 1) |
| Drawdown | `drawdown` (price / running max − 1) |
| Volatility | `rv_21d`, `rv_63d`, `parkinson_21d`, `garman_klass_21d` (range-based, from 1962) |
| Gaps | `gap_open` (log open / previous close) |
| Volume | `spy_vol_ratio_20d` (log of SPY volume / its 20-day mean) |
| Futures | `es_ret_1d` (ES daily log return; NaN in roll windows) |

Features from later-starting series (SPY 1993, ES 2000) leave earlier rows as NaN. Each model chooses its own start date or drops those features.
