# 5. Future Steps

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
- **Main vs. robustness bear label:** report every bear result on both `target_bear_h*` (main) and `target_bear_merged_h*` (robustness). The conclusion "model beats baselines" should hold on both. Keep the merge gap fixed at 126 days.
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
