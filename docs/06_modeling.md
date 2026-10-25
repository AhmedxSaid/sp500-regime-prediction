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

## Conclusions

1. **Stress is predictable; bear-market starts are not.** For the report: the useful product is a calibrated "stress in the next 1–4 weeks" probability.
2. **Logistic regression is the recommended main model:** best or tied-best on most metrics, simple, and interpretable. LightGBM is a close second, and the SVM is third.
3. **The SVM result is relevant to the literature:** in this regime task, an RBF SVM doesn't beat simpler models, unlike the price-level forecasts in Sheta et al. Note that we use a much longer test period (42 years vs. 43 weekly points).

## Next steps

See [05_future_steps.md](05_future_steps.md). In short: interpret logistic regression and LightGBM (coefficients / SHAP), test the extended feature set (`python run_models.py --features extended`) and new feature groups (VIX), check the per-crisis results, then run the 2022+ holdout **once**.
