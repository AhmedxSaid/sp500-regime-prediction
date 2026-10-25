# Processing summary

Generated 2026-10-25 15:50 by `process_data.py`.

## Outputs

| Table | Rows | Columns | From | To |
|---|---|---|---|---|
| `sp500_daily` | 24,821 | 32 | 1927-12-30 | 2026-10-23 |
| `model_table` | 24,821 | 34 | 1927-12-30 | 2026-10-23 |

Final holdout (do not use for tuning): 2022-01-01 onward, 1,207 rows.

## Raw data quality

| series | rows | start | end | non_positive_prices | flat_ohlc_rows | inconsistent_hlc_rows | zero_volume_rows | missing_close | moves_gt_15pct | largest_up_move | largest_down_move | largest_down_date | bad_prints_fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gspc | 24821 | 1927-12-30 | 2026-10-23 | 0 | 8547 | 0 | 5497 | 0 | 2 | 0.1537 | -0.229 | 1987-10-19 | 4 |
| sp500tr | 9776 | 1988-01-04 | 2026-10-23 | 0 | 5379 | 0 | 9776 | 0 | 0 | 0.1096 | -0.1276 | 2020-03-16 | 0 |
| spy | 8492 | 1993-01-29 | 2026-10-23 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1356 | -0.1159 | 2020-03-16 | 0 |
| es | 6590 | 2000-09-18 | 2026-10-23 | 0 | 0 | 10 | 11 | 0 | 0 | 0.132 | -0.1096 | 2020-03-16 | 0 |
| vix | 9273 | 1990-01-02 | 2026-10-23 | 0 | 508 | 0 | 9273 | 0 | 321 | 0.7682 | -0.4424 | 2025-04-09 | 0 |

## Regimes

| Label | Share of days | Episodes | Median length (days) | Longest (days) |
|---|---|---|---|---|
| `bear` | 20.9% | 27 | 168 | 437 |
| `bear_merged` | 23.6% | 17 | 268 | 1366 |
| `bear_recovery` | 50.6% | 12 | 390 | 6219 |
| `stressed` | 22.0% | 266 | 7 | 256 |
| `stressed_fixed` | 11.7% | 116 | 14 | 344 |

- `bear`: **main label**. Standard 20% rule, peak to trough.
- `bear_merged`: **robustness label**. `bear`, with bear markets merged when the rally between them lasts at most 126 trading days and stays below the starting peak.
- `bear_recovery`: causal "20% down and not yet back to the old high". Used as the feature `below_peak_regime`, not as a target.
- `stressed`: **main label**. 21-day realized vol above the 80th percentile of the **previous 5 years**' realized vol (unusually volatile for the recent market).
- `stressed_fixed`: **robustness label**. 21-day realized vol above a fixed 25% a year.

### Stressed share by decade

| Decade | `stressed` (5-year window) | `stressed_fixed` (> 25% vol) |
|---|---|---|
| 1920s | – | 19% |
| 1930s | 17% | 57% |
| 1940s | 15% | 6% |
| 1950s | 16% | 2% |
| 1960s | 18% | 1% |
| 1970s | 26% | 4% |
| 1980s | 30% | 5% |
| 1990s | 29% | 3% |
| 2000s | 32% | 18% |
| 2010s | 16% | 7% |
| 2020s | 19% | 12% |

### Bear markets found

`bear` (main) episodes. The last column shows which `bear_merged` episode each belongs to; rows sharing a number were merged.

| Start | End | Trading days | Merged episode |
|---|---|---|---|
| 1929-09-16 | 1929-11-13 | 41 | 1 |
| 1930-04-10 | 1930-12-16 | 172 | 1 |
| 1931-02-24 | 1931-06-02 | 70 | 1 |
| 1931-06-26 | 1931-10-05 | 71 | 1 |
| 1931-11-09 | 1932-06-01 | 141 | 1 |
| 1932-09-07 | 1933-02-27 | 117 | 1 |
| 1933-07-18 | 1933-10-19 | 66 | 1 |
| 1934-02-06 | 1935-03-14 | 274 | 1 |
| 1937-03-10 | 1938-03-31 | 268 | 2 |
| 1938-10-10 | 1939-04-11 | 123 | 3 |
| 1939-10-25 | 1940-06-10 | 156 | 4 |
| 1940-11-07 | 1942-04-28 | 366 | 4 |
| 1946-05-29 | 1947-05-19 | 244 | 5 |
| 1948-06-15 | 1949-06-13 | 251 | 6 |
| 1956-08-02 | 1957-10-22 | 308 | 7 |
| 1961-12-12 | 1962-06-26 | 136 | 8 |
| 1966-02-09 | 1966-10-07 | 168 | 9 |
| 1968-11-29 | 1970-05-26 | 370 | 10 |
| 1973-01-11 | 1974-10-03 | 437 | 11 |
| 1980-11-28 | 1982-08-12 | 431 | 12 |
| 1987-08-25 | 1987-12-04 | 72 | 13 |
| 2000-09-01 | 2001-09-21 | 262 | 14 |
| 2002-03-19 | 2002-07-23 | 88 | 14 |
| 2007-10-09 | 2008-11-20 | 284 | 15 |
| 2009-01-06 | 2009-03-09 | 43 | 15 |
| 2020-02-19 | 2020-03-23 | 24 | 16 |
| 2022-01-03 | 2022-10-12 | 196 | 17 |

## Targets

| Target | Non-missing | Share = 1 |
|---|---|---|
| `target_bear_h5` | 24,816 | 20.9% |
| `target_bear_merged_h5` | 24,816 | 23.6% |
| `target_stressed_h5` | 23,540 | 22.0% |
| `target_stressed_fixed_h5` | 24,800 | 11.7% |
| `target_bear_h21` | 24,800 | 20.9% |
| `target_bear_merged_h21` | 24,800 | 23.6% |
| `target_stressed_h21` | 23,540 | 22.0% |
| `target_stressed_fixed_h21` | 24,800 | 11.7% |

## Feature availability

| Feature | First valid date | Non-missing rows |
|---|---|---|
| `ret_1d` | 1928-01-03 | 24,820 |
| `mom_1m` | 1928-01-31 | 24,800 |
| `mom_3m` | 1928-04-02 | 24,758 |
| `mom_6m` | 1928-07-02 | 24,695 |
| `mom_12m` | 1929-01-03 | 24,569 |
| `dist_ma50` | 1928-03-13 | 24,772 |
| `dist_ma200` | 1928-10-16 | 24,622 |
| `drawdown` | 1927-12-30 | 24,821 |
| `rv_21d` | 1928-01-31 | 24,800 |
| `rv_63d` | 1928-04-02 | 24,758 |
| `parkinson_21d` | 1962-01-30 | 16,008 |
| `garman_klass_21d` | 1962-01-30 | 16,008 |
| `gap_open` | 1962-01-02 | 16,274 |
| `spy_vol_ratio_20d` | 1993-02-26 | 8,473 |
| `vix` | 1990-01-02 | 9,271 |
| `vix_rv_spread` | 1990-01-02 | 9,271 |
| `vix_chg_5d` | 1990-01-09 | 9,266 |
| `es_ret_1d` | 2000-09-19 | 6,041 |
| `below_peak_regime` | 1927-12-30 | 24,821 |
| `rv_vs_5y_p80` | 1933-02-10 | 23,540 |

## Checks (46 pass, 6 warn, 0 fail)

WARN on a known-bear-market check means that label splits the bear market at a 20%+ rally. This is expected for `bear` (the main label) and is the reason `bear_merged` exists. WARN on a medium sell-off for `stressed_fixed` is also expected: a fixed 25% level misses stress in calm eras, which is why the main stress label is relative.

| Check | Result | Detail |
|---|---|---|
| target_bear_h5 == bear.shift(-5) | PASS |  |
| target_bear_h5 is NaN in the last 5 rows | PASS |  |
| target_bear_merged_h5 == bear_merged.shift(-5) | PASS |  |
| target_bear_merged_h5 is NaN in the last 5 rows | PASS |  |
| target_stressed_h5 == stressed.shift(-5) | PASS |  |
| target_stressed_h5 is NaN in the last 5 rows | PASS |  |
| target_stressed_fixed_h5 == stressed_fixed.shift(-5) | PASS |  |
| target_stressed_fixed_h5 is NaN in the last 5 rows | PASS |  |
| target_bear_h21 == bear.shift(-21) | PASS |  |
| target_bear_h21 is NaN in the last 21 rows | PASS |  |
| target_bear_merged_h21 == bear_merged.shift(-21) | PASS |  |
| target_bear_merged_h21 is NaN in the last 21 rows | PASS |  |
| target_stressed_h21 == stressed.shift(-21) | PASS |  |
| target_stressed_h21 is NaN in the last 21 rows | PASS |  |
| target_stressed_fixed_h21 == stressed_fixed.shift(-21) | PASS |  |
| target_stressed_fixed_h21 is NaN in the last 21 rows | PASS |  |
| features + stress label unchanged when data after 2008-06-30 is removed | PASS |  |
| `bear` covers Great Depression 1929-32 | WARN | 73% of days labeled bear |
| `bear` covers 1973-74 | PASS | 100% of days labeled bear |
| `bear` covers Dot-com 2000-02 | WARN | 55% of days labeled bear |
| `bear` covers GFC 2007-09 | PASS | 92% of days labeled bear |
| `bear` covers COVID 2020 | PASS | 100% of days labeled bear |
| `bear` covers Inflation 2022 | PASS | 100% of days labeled bear |
| `bear_merged` covers Great Depression 1929-32 | PASS | 100% of days labeled bear |
| `bear_merged` covers 1973-74 | PASS | 100% of days labeled bear |
| `bear_merged` covers Dot-com 2000-02 | WARN | 74% of days labeled bear |
| `bear_merged` covers GFC 2007-09 | PASS | 100% of days labeled bear |
| `bear_merged` covers COVID 2020 | PASS | 100% of days labeled bear |
| `bear_merged` covers Inflation 2022 | PASS | 100% of days labeled bear |
| `stressed` catches Black Monday 1987 | PASS | 100% of days stressed |
| `stressed` catches GFC Oct-Nov 2008 | PASS | 100% of days stressed |
| `stressed` catches COVID Mar 2020 | PASS | 100% of days stressed |
| `stressed` catches 1962 flash crash | PASS | 91% of days stressed |
| `stressed` catches 1966 sell-off | PASS | 100% of days stressed |
| `stressed` catches 1970 bear | PASS | 85% of days stressed |
| `stressed` is calm in 2017 | PASS | 0% of days stressed |
| `stressed_fixed` catches Black Monday 1987 | PASS | 97% of days stressed |
| `stressed_fixed` catches GFC Oct-Nov 2008 | PASS | 100% of days stressed |
| `stressed_fixed` catches COVID Mar 2020 | PASS | 100% of days stressed |
| `stressed_fixed` catches 1962 flash crash | WARN | 48% of days stressed |
| `stressed_fixed` catches 1966 sell-off | WARN | 0% of days stressed |
| `stressed_fixed` catches 1970 bear | WARN | 38% of days stressed |
| `stressed_fixed` is calm in 2017 | PASS | 0% of days stressed |
| `bear_merged` >= `bear` on every day | PASS |  |
| drawdown is between -1 and 0 | PASS |  |
| realized vol is non-negative | PASS |  |
| bear only takes values 0/1 | PASS | [0, 1] |
| bear_merged only takes values 0/1 | PASS | [0, 1] |
| bear_recovery only takes values 0/1 | PASS | [0, 1] |
| stressed only takes values 0/1 | PASS | [0, 1] |
| stressed_fixed only takes values 0/1 | PASS | [0, 1] |
| dates are unique and sorted | PASS |  |
