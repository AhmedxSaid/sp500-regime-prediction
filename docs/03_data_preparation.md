# 3. Data Preparation

Run all of it with `python process_data.py` (from the repo root). That script calls the modules below, runs the checks, and writes `reports/processing_summary.md`.

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

**Decision (2026-10-25): use both.** The standard 20% rule is the main label, and a merged version is the robustness check:

| Column | Definition | Causal? | Role |
|---|---|---|---|
| `bear` | **Main label.** From the peak before a ≥20% fall until the trough (the bear ends once price rallies 20% off the trough) | **No.** The peak and trough are only known later | Main **target**; all headline results use it |
| `bear_merged` | **Robustness label.** `bear`, but two bear markets are merged when the rally between them lasts ≤ 126 trading days (~6 months) **and** never closes above the peak where the first one started | **No** (hindsight) | Second **target**: results must hold on it too |
| `bear_recovery` | From the day price closes 20% below the peak until it gets back to that peak | **Yes.** Known at day *t* | Not a target. Used as the feature `below_peak_regime` and as the persistence baseline |

The 126-day gap is fixed in `config.BEAR_MERGE_MAX_GAP` and was chosen before any model results. **Do not tune it.** Changing it after seeing results would be overfitting the label.

Why both: the 20% rule is standard and has no extra setting, but it calls failed rallies inside long bear markets "bull" (e.g. Sept 2001 → Jan 2002). The merged label matches the risk-management view, at the cost of hindsight and fewer episodes. If the model beats the baselines on both, the result doesn't depend on that choice.

By default the labels use `^SP500TR` (includes dividends), with `^GSPC` as a fallback.

### Calm / stressed

- 21-day realized volatility: `std(log returns over 21 days) × √252`.
- Threshold: the **expanding** 80th percentile of past realized volatility, shifted by one day. This uses only past data and needs 5 years of history before the first label.
- `stressed = 1` if the current realized volatility is above the threshold.

### Sanity check

Plot the labels against price and check that they catch the known bear markets: 2000–02, 2007–09, 2020 and 2022. This is done in the EDA notebook.

**Results from the first run (data through 2026-10-23):**

| Label | Share of days | Episodes | Median length | Longest |
|---|---|---|---|---|
| `bear` (main) | 20.9% | 27 | 168 days | 437 days |
| `bear_merged` (robustness) | 23.6% | 17 | 268 days | 1,366 days |
| `bear_recovery` (feature) | 50.6% | 12 | 390 days | 6,219 days |
| `stressed` | 11.4% | 140 | 10 days | 241 days |

- `bear` fully captures 1973–74, 2020 and 2022, and 92% of 2007–09. It **splits** 1929–32 (73% covered) and 2000–02 (55%), because each had a bear-market rally of more than 20%.
- `bear_merged` joins these: 1929–35 becomes one episode, as do 2000–02 and 2007–09. 1929–32 is then 100% covered, and 2000–02 is 74%. The remaining dot-com gap has two causes:
  - The label uses the total-return index, which peaked in **Sept 2000** (dividends kept it rising), while the price index peaked in March 2000.
  - The final Aug → Oct 2002 drop was less than 20% from the August high, so it is not a new bear.
- `bear_recovery` marks half of all days as bear. After 1929 the index only regained its peak in 1954. That's why it's a feature, not a target.
- The full episode list, with which episodes were merged, is in [`reports/processing_summary.md`](../reports/processing_summary.md).

## Step 3 — Prediction target

`target_{label}_h{h} = label.shift(-h)` for label ∈ {`bear`, `bear_merged`, `stressed`} and h ∈ {5, 21}, giving 6 targets.

We predict the regime **h days ahead**, not today's regime. Labels are shifted forward; features are never shifted backward. The last *h* rows have no target and are dropped at training time.

## Step 4 — Features (`features.py`)

All features use only data available by the close of day *t*.

| Group | Features |
|---|---|
| Returns / momentum | `ret_1d`, `mom_1m`, `mom_3m`, `mom_6m`, `mom_12m` (log returns over 21/63/126/252 days) |
| Trend | `dist_ma50`, `dist_ma200` (price / moving average − 1) |
| Drawdown | `drawdown` (price / running max − 1), `below_peak_regime` (1 if price fell 20% and hasn't regained the old high; causal) |
| Volatility | `rv_21d`, `rv_63d`, `parkinson_21d`, `garman_klass_21d` (range-based, from 1962) |
| Gaps | `gap_open` (log open / previous close) |
| Volume | `spy_vol_ratio_20d` (log of SPY volume / its 20-day mean) |
| Futures | `es_ret_1d` (ES daily log return; NaN in roll windows) |

Features from later-starting series (SPY 1993, ES 2000) leave earlier rows as NaN. Each model chooses its own start date or drops those features.
