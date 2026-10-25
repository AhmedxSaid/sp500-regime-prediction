# 6. Modeling and Evaluation (first round)

Code: [`src/sp500/models.py`](../src/sp500/models.py), [`run_models.py`](../run_models.py). Full tables: [`reports/model_results_main.md`](../reports/model_results_main.md).

```bash
python process_data.py
python run_models.py            # ~2 minutes
```

## Setup (fixed before any results)

| Item | Choice | Why |
|---|---|---|
| Research focus | Regime prediction; **SVM is one of the compared models** | Links the project to the literature review (Sheta et al., Chen et al., Rosillo et al.) |
| Features | 13 "main" features available from 1962 (`config.FEATURES_MAIN`) | Real OHL starts in 1962; near-duplicates from EDA removed |
| Training data | 1962 → year before the test year (expanding) | The first fold already includes 4 bear markets (1962, 1966, 1968–70, 1973–74) |
| Test years | **1980–2021**, one year at a time (42 folds) | 2022+ is the untouched final holdout |
| Gap | 21 trading days between train and test | ≥ the longest horizon, so overlapping targets can't leak |
| Bear labels in training | **Recomputed in each fold using only prices known at the cut-off** | The bear label needs hindsight; this stops training from seeing how later bears were dated |
| Hyperparameters | Fixed defaults, no tuning | Tuning on the test years would overstate results |
| Class imbalance | No reweighting | Keeps probabilities calibrated (Brier); PR-AUC handles imbalance |

**Methods**

| Type | Method | Details |
|---|---|---|
| Baseline | No change (persistence) | Future regime = current *known* state (stress now; for bear, "20% below the old peak") |
| Baseline | Price < 200-day MA | Bear targets only |
| Baseline | 2-state HMM | Gaussian HMM on daily returns; causal forward filter, projected *h* days ahead |
| Model | Logistic regression | Robust scaling, L2 (C = 1) |
| Model | LightGBM | 200 trees, depth-limited (15 leaves, ≥ 200 rows per leaf) |
| Model | SVM (RBF) | Trained on weekly rows (as in Sheta et al.), Platt-scaled probabilities |

## Results (walk-forward, 1980–2021)

### Stress: models clearly beat the baselines

| `stressed` (main label) | PR-AUC, 5 days | PR-AUC, 21 days | Onset PR-AUC, 21 days | Brier, 21 days |
|---|---|---|---|---|
| Base rate (random guess) | 0.266 | 0.266 | 0.128 | – |
| Baseline: no change | 0.776 | 0.512 | 0.128 | 0.188 |
| Baseline: 2-state HMM | 0.749 | 0.531 | 0.191 | 0.161 |
| **Logistic regression** | **0.930** | **0.661** | 0.260 | **0.143** |
| **LightGBM** | 0.927 | 0.648 | **0.265** | 0.148 |
| SVM (RBF) | 0.847 | 0.555 | 0.211 | 0.157 |

- **Big improvement over "no change":** at 21 days, PR-AUC rises from 0.51 to 0.66. On days that are still calm, the models identify upcoming stress about **twice as well** as "no change" (onset PR-AUC 0.26 vs. 0.13).
- **The robustness label agrees:** on `stressed_fixed` (> 25% vol), logistic regression scores 0.56 vs. 0.32 for "no change" at 21 days. The conclusion doesn't depend on the stress definition.
- **Simple wins:** logistic regression is as good as LightGBM, and the SVM is clearly behind both. This is consistent with EDA finding 1: stress is driven by volatility persistence, which is close to linear.

### Bear markets: models recognize them, but can't predict their start

| `bear` (main label) | PR-AUC, 5 days | PR-AUC, 21 days | Onset PR-AUC, 21 days | Brier, 21 days |
|---|---|---|---|---|
| Base rate (random guess) | 0.114 | 0.116 | 0.018 | – |
| Baseline: no change | 0.131 | 0.122 | 0.017 | 0.335 |
| Baseline: price < 200-day MA | 0.299 | 0.261 | 0.018 | 0.204 |
| Baseline: 2-state HMM | 0.277 | 0.280 | **0.044** | 0.116 |
| Logistic regression | **0.361** | 0.293 | 0.032 | 0.100 |
| **LightGBM** | 0.358 | **0.340** | 0.014 | 0.103 |
| SVM (RBF) | 0.355 | 0.313 | 0.019 | **0.091** |

- **Overall PR-AUC is about 3× random** (0.34–0.36 vs. 0.11) and beats every baseline. That comes from recognizing a bear market **while it's underway** (deep drawdown, below trend, high volatility).
- **Onset PR-AUC is at the random level** (0.014–0.032 vs. a base rate of 0.018). No method can tell, from a normal-looking day, that a bear market will start within 21 days.
- This is expected. The bear label starts at the **market peak**, the day the market looks strongest. Only 7 bear markets start in the test years, so there's very little to learn from.
- **The robustness label agrees** (`bear_merged`: same pattern).

![Predicted probability vs. actual regimes, LightGBM, 21 days](figures/model_timeline_main.png)

**Timeline (2000–2021):**
- **Stress (top):** the probability rises at almost every stress episode (2000–02, 2007–09, 2011, 2015, 2018, 2020) and stays low in calm years (2004–06, 2013–14, 2017).
- **Bear (bottom):** the model is high during 2008 but mostly low during the 2000–02 bear market. It gives false alarms in **2015–16 and late 2018**, which were sharp sell-offs of 14–20% that never became official bear markets.

### Early warning

Days of warning before a regime started (probability ≥ 50% on the days before the start):

| | Stress starts (130) with ≥ 5 days warning | Bear starts (7) with ≥ 5 days warning |
|---|---|---|
| Best model | 13% (LightGBM) | 0% |

Stress usually starts suddenly (one bad day pushes volatility above the threshold), so true early warning is rare. The models' value is mostly in **recognizing stress within the first days** and in **ranking risk** (higher probability before stress than before calm), not in long advance warning.

## Feature group tests (SPY volume, VIX)

Code: [`run_feature_groups.py`](../run_feature_groups.py). Full tables: [`reports/feature_groups.md`](../reports/feature_groups.md).

**Method.** Each group is added on top of the 13 main features and compared with the main features alone, on **exactly the same rows**: training from 1993 (when both groups exist), test years 2000–2021, logistic regression and LightGBM. The decision rule was fixed in `config.py` before running: keep a group only if mean PR-AUC over the 4 main targets improves by **≥ 0.01 for both models**, and onset PR-AUC drops by no more than 0.01.

**New data:** `^VIX` (CBOE Volatility Index: the market's expected volatility over the next 30 days, from option prices), 1990-01-02 → 2026-10-23, in `data/raw/vix.csv`. Features: `vix` (level), `vix_rv_spread` (VIX minus realized vol: the "fear premium"), `vix_chg_5d` (1-week change).

| Group | Δ PR-AUC (logistic / LightGBM) | Δ onset PR-AUC | Decision |
|---|---|---|---|
| SPY volume (`spy_vol_ratio_20d`) | +0.005 / +0.002 | +0.011 / +0.010 | ❌ **Drop** (below the 0.01 bar) |
| VIX (3 features) | **+0.020 / +0.018** | **+0.043 / +0.018** | ✅ **Passes the rule** |

**VIX: useful, but not worth its shorter history.**
- **Where it helps:** logistic regression gains mostly on stress in 21 days (0.569 → 0.626); LightGBM gains mostly on bear (0.233 → 0.266).
- **The gain is not robust.**
  - On the robustness labels it nearly disappears (mean +0.001 / +0.002).
  - Year by year, LightGBM improves in only 9 of 16 test years with both classes. Logistic regression improves in 14 of 16.
- **Using VIX means training from 1990, and that costs more than VIX adds.** On the same test years 2000–2021:

| Target (2000–2021) | Main model: 1962+, no VIX | "Modern" model: 1990+, with VIX |
|---|---|---|
| Stressed in 21 days (logistic regression) | **0.725** | 0.674 |
| Stressed in 21 days (LightGBM) | **0.721** | 0.688 |
| Bear in 5 days (logistic regression) | **0.375** | 0.153 |
| Bear in 21 days (LightGBM) | **0.247** | 0.207 |

  With training from 1990, the early folds contain only one short bear market (1990), so the bear models barely learn. Even for stress, 28 extra years of history beat VIX.

**Decision:** the **main 1962 model (13 features, no VIX) stays the primary model**. The VIX model is kept for reference ([`reports/model_results_modern.md`](../reports/model_results_modern.md), `python run_models.py --features modern`).

**Last check: VIX on the full history.** LightGBM can train with missing values, so it was trained from 1962 with the VIX features left empty before 1990, and compared with LightGBM on the main features on identical rows and test years 2000–2021. The same thresholds were used, applied to LightGBM only, since logistic regression can't take missing values. Full tables: [`reports/feature_groups_vix_hybrid.md`](../reports/feature_groups_vix_hybrid.md).

| LightGBM, 2000–2021 | Bear 5d | Bear 21d | Stressed 5d | Stressed 21d | Mean (main) |
|---|---|---|---|---|---|
| Main features | **0.312** | **0.247** | **0.955** | **0.721** | **0.559** |
| + VIX (missing before 1990) | 0.275 | 0.230 | 0.952 | 0.703 | 0.540 |

- **Result: −0.018 mean PR-AUC (−0.018 on the robustness labels too) → ❌ drop.**
- Onset PR-AUC is unchanged (+0.001). Only `stressed_fixed` at 21 days improves (0.602 → 0.629).
- Year by year it beats the main set in 12 of 16 years on stressed-21d, but it's worse when all years are pooled. Its probabilities seem less consistent across years: before 1990 the model learned from rows with VIX missing, after 1990 from rows with VIX present.

**Final decision on VIX: not used.** All three ways of adding it (common 1993+ sample, separate 1990+ model, full history with VIX missing) either fail the rule or lose to the main model on the same test years.

**Side finding (2000–2021 only):** for **bear**, the simple baselines are as good as or better than the models. At 21 days, the 2-state HMM scores 0.352 and the 200-day-MA rule 0.317, vs. 0.268 for logistic regression. The models' bear advantage in the full 1980–2021 test comes mostly from the 1980s–90s.

## Conclusions

1. **Stress is predictable; bear-market starts are not.** For the report: the useful product is a calibrated "stress in the next 1–4 weeks" probability.
2. **Logistic regression is the recommended main model:** best or tied-best on most metrics, simple, and interpretable. LightGBM is a close second, and the SVM is third.
3. **Long history beats extra data sources.** SPY volume doesn't help. VIX helps a little on a common 1993+ sample, but it loses to the main model whether it's used in a 1990+ model or on the full history with missing values. The primary model keeps the 13 main features.
4. **For bear markets, simple rules are hard to beat** in recent decades (2000–2021).
3. **The SVM result is relevant to the literature:** in this regime task, an RBF SVM doesn't beat simpler models, unlike the price-level forecasts in Sheta et al. Note that we use a much longer test period (42 years vs. 43 weekly points).

## Next steps

See [05_future_steps.md](05_future_steps.md). In short: interpret logistic regression and LightGBM (coefficients / SHAP), check the per-crisis results, then run the 2022+ holdout **once**.
