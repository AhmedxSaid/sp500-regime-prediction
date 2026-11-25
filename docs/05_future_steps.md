# 5. Future Steps

## Status (2026-11-25)

| Step | Status |
|---|---|
| 5.1 Validation design | ✅ Done: `validation.py` + per-fold bear labels in `models.py` |
| 5.2 Baselines | ✅ Done: no change, 200-day MA rule, 2-state HMM |
| 5.3 Models: logistic regression, LightGBM, SVM | ✅ Done. Results in [06_modeling.md](06_modeling.md) |
| 5.3 LSTM | ⏸ Not needed for regimes: simple models already match LightGBM. (Used in the min / max extension, 5.8.) |
| 5.4 Evaluation: PR-AUC, onset PR-AUC, Brier, flips, lead time | ✅ Done |
| 5.4 Per-crisis results | ✅ Done: 15 stress crises, 7 bear markets ([06_modeling.md](06_modeling.md)) |
| 5.5 Interpretation (coefficients / SHAP) | ✅ Done: `interpret_models.py` |
| 5.5 Feature groups: SPY volume, VIX | ✅ Done: both dropped. VIX passes on a common 1993+ sample but loses to the main model in every full-history setup ([06_modeling.md](06_modeling.md)) |
| 5.5 Other groups (VIX term structure, sector ETFs, breadth) | ⬜ Optional: VIX3M only exists from ~2007, too short to test reliably |
| 5.6 Literature extensions | ⬜ Optional |
| 5.7 Final holdout (2022+) | ✅ Done once, 2026-10-27 ([07_final_evaluation.md](07_final_evaluation.md)) |
| 5.7 Final written report / presentation | ✅ Done: written report (Claude Doc) and 16-slide deck, 2026-10-27 |
| 5.8 Extension: min / max price (XGBoost, CatBoost, LSTM) + dashboard | ✅ Done ([08_price_range.md](08_price_range.md)) |
| 5.9 Data refresh to 2026-11-24 + quality fixes (ES range, SPY bad closes, close-only rows) | ✅ Done, 2026-11-25; regime reports unchanged (holdout capped at `HOLDOUT_END`) |
| 5.9 Live check on the days after the 2026-10-23 snapshot | ✅ First run 2026-11-25 ([`reports/live_check.md`](../reports/live_check.md)); re-run after each refresh |
| 5.10 Calibrate the min / max bands (conformal) | ⬜ Next: the 80% bands cover the low only 63–74% of the time |

## 5.1 Validation design

- **Walk-forward:** train up to year *N*, test on year *N + 1*, then roll forward (`sp500.validation.walk_forward_splits`).
- **Gap:** at least *h* trading days between the end of training and the start of testing, so that overlapping targets don't leak.
- **Final holdout:** 2022-01-01 onward. Use it once, at the end.
- **Class imbalance:** use class weights. Do not oversample, because that mixes time periods.

## 5.2 Baselines (any model must beat these)

1. **Persistence:** future regime = current regime (the causal `below_peak_regime` feature for bear / current stress state).
2. **Threshold rule:** bear if price < 200-day MA; stressed if realized volatility is above its threshold today.
3. **2-state Hidden Markov Model** on daily returns (e.g. `hmmlearn`), fitted on training data only.

## 5.3 Models (start simple)

1. Regularized logistic regression (L1/L2) on standardized features.
2. Gradient boosting (LightGBM or XGBoost).
3. **SVM (RBF kernel)** as a comparison with the literature (see [literature.md](literature.md)).
4. LSTM only if the simpler models leave clear room to improve.

## 5.4 Evaluation

- Precision-recall AUC (the classes are imbalanced, so prefer it over ROC-AUC).
- Brier score and calibration curves.
- Results within each crisis: 2000–02, 2007–09, 2011, 2015–16, 2018 Q4, 2020, 2022.
- **Early-warning lead time:** days between the probability crossing a threshold and the start of the episode.
- **Stability:** number of signal flips per year.

## 5.5 Interpretation and robustness

- SHAP values / feature importance.
- **Main vs. robustness labels:** report every bear result on both `target_bear_h*` (main) and `target_bear_merged_h*` (robustness), and every stress result on both `target_stressed_h*` (5-year window) and `target_stressed_fixed_h*` (> 25% vol). The conclusion "model beats baselines" should hold on both. Keep the merge gap (126 days), stress lookback (5 years) and fixed level (25%) unchanged.
- Vary the label thresholds (15/20/25% drawdown; 75/80/90th volatility percentile), the horizon *h* (5, 10, 21, 63) and the start date.
- Add feature groups one at a time (VIX term structure → sector ETFs → breadth) and keep a group only if it improves out-of-sample results.

## 5.6 Extensions from the literature

- **SVR price/return forecasting** (Sheta et al.): reproduce the setup with `r_{t+1} = f(X_t)` and compare with linear regression and MLP using MAE / RMSE / R². Report returns, not price levels, because price-level R² is misleadingly high.
- **SVM + news sentiment** (Chen et al., 2017): does adding sentiment improve directional prediction? Compare `SVM_market` with `SVM_market+sentiment`.
- **GARCH-simulated markets** (Rosillo et al.): fit a GARCH model to the S&P 500, simulate bull, bear, high-volatility and low-volatility markets, and test how SVM performance changes by regime.

## 5.7 Final report

- Model vs. baselines table (walk-forward and holdout).
- A timeline of predicted probabilities against the actual regimes.
- Limitations: few bear markets in the sample, dependence on the label choice, and regime shifts in market structure.

## 5.8 Extension: min / max price

Done; see [08_price_range.md](08_price_range.md). Quantile models for the highest and lowest price over the next 5 and 21 days, validated walk-forward 2000–2021 and on the 2022+ holdout, shown in `dashboard/app.py`.

## 5.9 Keeping the results honest on new data

- The 2022+ regime holdout was used once, on data through **2026-10-23** (`HOLDOUT_END` in `config.py`). `run_holdout.py` and `run_statistical_tests.py` stop there, so refreshed data reproduces the published numbers.
- Every later day is genuinely new. After each refresh, run `python run_live_check.py`: it trains only on data known before 2026-10-23 and scores every forecast made since.
- The first live check covers 23 days and one stress onset, so it is a sanity check, not a new accuracy estimate. Re-running it as data arrives builds a real out-of-sample record.

## 5.10 Next: calibrated min / max bands

The 80% bands are too narrow for the low (fat left tail; the 21-day low after 2026-10-23 fell below every model's 10% quantile). Conformal calibration on the walk-forward residuals would widen them in a principled way. Fit it on 2000–2021 only and check it on the holdout and the live check.
