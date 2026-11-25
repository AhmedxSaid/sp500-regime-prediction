# Live check

Generated 2026-11-25 20:49 by `run_live_check.py` (0.6 min). Models trained only on data known before **2026-10-23** (the last day of the data every regime result used); forecasts made at each close from 2026-10-23 to 2026-11-24 (23 trading days). Same features, settings and gaps as validation; nothing tuned.

S&P 500: 7,624 at the cut-off → 7,201 on 2026-11-24 (-5.5%); lowest close 6,910 on 2026-11-20.

With this few days, the numbers are a sanity check of the reported results, not a new estimate of accuracy: one stress onset is a single event.

## What happened

- `stressed`: 2026-11-05 → 2026-11-24 (14 days)
- `stressed_fixed`: 2026-11-20 → 2026-11-24 (3 days)
- `bear`: not on at any time in the live period
- `bear_merged`: not on at any time in the live period

## Regime forecasts made at the cut-off close

Probability, at the 2026-10-23 close, that the regime is on within *h* days, and whether it was.

| Target | Happened? | Baseline: no change | Baseline: price < 200-day MA | Baseline: 2-state HMM | Logistic regression | LightGBM | SVM (RBF) |
|---|---|---|---|---|---|---|---|
| `stressed` in 5d | no | 0.00 | – | 0.06 | 0.00 | 0.01 | 0.01 |
| `stressed` in 21d | **yes** | 0.00 | – | 0.15 | 0.10 | 0.04 | 0.14 |
| `stressed_fixed` in 5d | no | 0.00 | – | 0.06 | 0.00 | 0.00 | 0.02 |
| `stressed_fixed` in 21d | **yes** | 0.00 | – | 0.15 | 0.02 | 0.01 | 0.06 |
| `bear` in 5d | no | 0.00 | 0.00 | 0.06 | 0.06 | 0.00 | 0.02 |
| `bear` in 21d | no | 0.00 | 0.00 | 0.15 | 0.07 | 0.01 | 0.03 |
| `bear_merged` in 5d | no | 0.00 | 0.00 | 0.06 | 0.06 | 0.00 | 0.02 |
| `bear_merged` in 21d | no | 0.00 | 0.00 | 0.15 | 0.07 | 0.01 | 0.03 |

## Warning before each regime start in the live period

First day the probability reached 0.5, in trading days relative to the start (negative = warned in advance; the forecast must come before the start to count as a warning).

| Regime start | Horizon | Baseline: no change | Baseline: 2-state HMM | Logistic regression | LightGBM | SVM (RBF) |
|---|---|---|---|---|---|---|
| `stressed` 2026-11-05 | 5d | +0 (2026-11-05) | +0 (2026-11-05) | +0 (2026-11-05) | +0 (2026-11-05) | +0 (2026-11-05) |
| `stressed` 2026-11-05 | 21d | +0 (2026-11-05) | +0 (2026-11-05) | +10 (2026-11-19) | +0 (2026-11-05) | +0 (2026-11-05) |
| `stressed_fixed` 2026-11-20 | 5d | +0 (2026-11-20) | -11 (2026-11-05) | +0 (2026-11-20) | +0 (2026-11-20) | -11 (2026-11-05) |
| `stressed_fixed` 2026-11-20 | 21d | +0 (2026-11-20) | -11 (2026-11-05) | never | -11 (2026-11-05) | never |

## Daily `stressed` probabilities

| Date | Close | Stressed | Within 5d? | LogReg 5d | LightGBM 5d | HMM 5d | Within 21d? | LogReg 21d | LightGBM 21d | HMM 21d |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | 7,624 | no | no | 0.00 | 0.01 | 0.06 | yes | 0.10 | 0.04 | 0.15 |
| 2026-10-26 | 7,677 | no | no | 0.00 | 0.01 | 0.06 | yes | 0.08 | 0.03 | 0.15 |
| 2026-10-27 | 7,723 | no | no | 0.00 | 0.01 | 0.06 | – | 0.07 | 0.03 | 0.14 |
| 2026-10-28 | 7,744 | no | no | 0.00 | 0.00 | 0.05 | – | 0.06 | 0.02 | 0.14 |
| 2026-10-29 | 7,720 | no | yes | 0.00 | 0.01 | 0.05 | – | 0.07 | 0.03 | 0.14 |
| 2026-10-30 | 7,580 | no | yes | 0.01 | 0.01 | 0.18 | – | 0.12 | 0.07 | 0.20 |
| 2026-11-02 | 7,541 | no | yes | 0.01 | 0.00 | 0.12 | – | 0.09 | 0.04 | 0.18 |
| 2026-11-03 | 7,569 | no | yes | 0.01 | 0.00 | 0.08 | – | 0.08 | 0.05 | 0.16 |
| 2026-11-04 | 7,689 | no | yes | 0.01 | 0.01 | 0.16 | – | 0.07 | 0.05 | 0.19 |
| 2026-11-05 | 7,354 | yes | yes | 0.59 | 0.66 | 0.84 | – | 0.44 | 0.62 | 0.52 |
| 2026-11-06 | 7,377 | yes | yes | 0.56 | 0.59 | 0.77 | – | 0.27 | 0.37 | 0.49 |
| 2026-11-09 | 7,550 | yes | yes | 0.71 | 0.86 | 0.84 | – | 0.20 | 0.34 | 0.52 |
| 2026-11-10 | 7,521 | yes | yes | 0.73 | 0.81 | 0.78 | – | 0.28 | 0.41 | 0.49 |
| 2026-11-11 | 7,619 | yes | yes | 0.74 | 0.84 | 0.78 | – | 0.21 | 0.30 | 0.49 |
| 2026-11-12 | 7,592 | yes | yes | 0.69 | 0.81 | 0.67 | – | 0.25 | 0.36 | 0.44 |
| 2026-11-13 | 7,562 | yes | yes | 0.71 | 0.80 | 0.52 | – | 0.27 | 0.40 | 0.37 |
| 2026-11-16 | 7,417 | yes | yes | 0.81 | 0.91 | 0.81 | – | 0.38 | 0.65 | 0.50 |
| 2026-11-17 | 7,341 | yes | yes | 0.84 | 0.90 | 0.79 | – | 0.39 | 0.73 | 0.49 |
| 2026-11-18 | 7,324 | yes | – | 0.85 | 0.89 | 0.68 | – | 0.38 | 0.56 | 0.44 |
| 2026-11-19 | 7,176 | yes | – | 0.93 | 0.92 | 0.83 | – | 0.55 | 0.81 | 0.51 |
| 2026-11-20 | 6,910 | yes | – | 0.99 | 0.99 | 0.84 | – | 0.79 | 0.90 | 0.52 |
| 2026-11-23 | 7,237 | yes | – | 1.00 | 0.98 | 0.84 | – | 0.50 | 0.71 | 0.52 |
| 2026-11-24 | 7,201 | yes | – | 1.00 | 0.98 | 0.79 | – | 0.70 | 0.79 | 0.49 |

"–" in the *Within* columns: the answer needs days after the last date in the data.

## Min / max forecasts made at the cut-off close

10–90% band and median for the lowest and highest close-to-extreme move over the next *h* days, in S&P 500 points (close 7,624).

| Horizon | Target | Actual | Baseline: historical quantiles | Baseline: volatility band | XGBoost | CatBoost | LSTM | Ensemble (XGB + Cat + LSTM) |
|---|---|---|---|---|---|---|---|---|
| 5d | max high | 7,758 | 7,640 / 7,723 / 7,887 ✓ | 7,634 / 7,686 / 7,774 ✓ | 7,640 / 7,722 / 7,827 ✓ | 7,638 / 7,722 / 7,836 ✓ | 7,631 / 7,703 / 7,829 ✓ | 7,636 / 7,716 / 7,831 ✓ |
| 5d | min low | 7,528 | 7,345 / 7,539 / 7,617 ✓ | 7,462 / 7,570 / 7,619 ✓ | 7,427 / 7,558 / 7,629 ✓ | 7,416 / 7,553 / 7,625 ✓ | 7,395 / 7,542 / 7,621 ✓ | 7,413 / 7,551 / 7,625 ✓ |
| 21d | max high | 7,758 | 7,674 / 7,848 / 8,149 ✓ | 7,654 / 7,764 / 7,938 ✓ | 7,677 / 7,846 / 8,011 ✓ | 7,684 / 7,850 / 8,036 ✓ | 7,667 / 7,812 / 7,999 ✓ | 7,676 / 7,836 / 8,015 ✓ |
| 21d | min low | 6,883 | 7,060 / 7,449 / 7,600 ✗ | 7,287 / 7,513 / 7,608 ✗ | 7,259 / 7,481 / 7,614 ✗ | 7,268 / 7,481 / 7,611 ✗ | 7,221 / 7,488 / 7,615 ✗ | 7,250 / 7,483 / 7,613 ✗ |

Each cell: 10% / 50% / 90% quantile. ✓ = actual inside the 10–90% band, ✗ = outside.

## Min / max: every live day with a known outcome

Coverage should be near 80% for a well-calibrated 10–90% band (with this few, overlapping days, expect a lot of noise). Error = mean absolute error of the median, in % of the close.

| Horizon | Target | Days | Model | Inside band | Median error | Validation coverage |
|---|---|---|---|---|---|---|
| 5d | max high | 18 | Baseline: historical quantiles | 61% | 1.03% | 78% |
| 5d | max high | 18 | Baseline: volatility band | 56% | 1.06% | 77% |
| 5d | max high | 18 | XGBoost | 67% | 0.92% | 79% |
| 5d | max high | 18 | CatBoost | 72% | 0.93% | 78% |
| 5d | max high | 18 | LSTM | 72% | 0.95% | 80% |
| 5d | max high | 18 | Ensemble (XGB + Cat + LSTM) | 72% | 0.93% | 79% |
| 5d | min low | 18 | Baseline: historical quantiles | 50% | 2.80% | 73% |
| 5d | min low | 18 | Baseline: volatility band | 22% | 2.96% | 71% |
| 5d | min low | 18 | XGBoost | 28% | 2.99% | 72% |
| 5d | min low | 18 | CatBoost | 28% | 2.95% | 72% |
| 5d | min low | 18 | LSTM | 50% | 2.92% | 73% |
| 5d | min low | 18 | Ensemble (XGB + Cat + LSTM) | 33% | 2.95% | 72% |
| 21d | max high | 2 | Baseline: historical quantiles | 100% | 1.50% | 81% |
| 21d | max high | 2 | Baseline: volatility band | 100% | 0.42% | 80% |
| 21d | max high | 2 | XGBoost | 100% | 1.25% | 74% |
| 21d | max high | 2 | CatBoost | 100% | 1.35% | 70% |
| 21d | max high | 2 | LSTM | 100% | 1.03% | 82% |
| 21d | max high | 2 | Ensemble (XGB + Cat + LSTM) | 100% | 1.21% | 76% |
| 21d | min low | 2 | Baseline: historical quantiles | 0% | 8.25% | 74% |
| 21d | min low | 2 | Baseline: volatility band | 0% | 9.11% | 73% |
| 21d | min low | 2 | XGBoost | 0% | 8.68% | 67% |
| 21d | min low | 2 | CatBoost | 0% | 8.67% | 63% |
| 21d | min low | 2 | LSTM | 0% | 8.78% | 74% |
| 21d | min low | 2 | Ensemble (XGB + Cat + LSTM) | 0% | 8.71% | 69% |

