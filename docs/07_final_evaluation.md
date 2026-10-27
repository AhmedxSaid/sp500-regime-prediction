# 7. Final Evaluation: 2022+ holdout (one-time test)

Code: [`run_holdout.py`](../run_holdout.py). Full tables: [`reports/holdout_results.md`](../reports/holdout_results.md).

## What was tested

The holdout period, **2022-01-01 → 2026-10-16** (the last day whose 21-day target is known), was set aside at the start of the project and **never used for any decision**: not for labels, features, feature groups, models, hyperparameters or the keep rules.

The setup is the one fixed before testing:
- **Features:** 13 main features, training from 1962.
- **Training:** models refit each January on all earlier data, with a 21-day gap.
- **Methods:** 3 baselines and 3 models.
- **Targets:** all 4 labels × 2 horizons.

The models were fitted **once** (26 seconds). The report can be rebuilt from the saved predictions without refitting (`--report-only`).

**What the period contains:** 1,202 trading days, **1 bear market** (Jan–Oct 2022, −25%), and **5 stress episodes of 10+ days** (three in 2022, plus the April 2025 tariff shock). That's a small sample, so the results below come with bootstrap confidence intervals.

## Results

### Main targets: PR-AUC, holdout vs. validation

| Target | Holdout base rate | Best baseline | Logistic regression | LightGBM | SVM | Validation (1980–2021): logistic / LightGBM |
|---|---|---|---|---|---|---|
| Stressed in 5 days | 15.6% | 0.780 (no change) | 0.938 | 0.947 | 0.954 | 0.930 / 0.927 |
| Stressed in 21 days | 15.8% | 0.485 (HMM) | 0.549 | 0.590 | 0.602 | 0.661 / 0.648 |
| Bear in 5 days | 15.9% | 0.465 (MA200 rule) | 0.481 | 0.448 | 0.612 | 0.361 / 0.358 |
| Bear in 21 days | 14.8% | 0.413 (MA200 rule) | 0.433 | 0.345 | 0.534 | 0.293 / 0.340 |

**Onset PR-AUC** (predicting stress *starting* on a calm day), holdout:

| Target | No change | HMM | Logistic regression | LightGBM |
|---|---|---|---|---|
| Stressed in 5 days | 0.024 | 0.165 | **0.611** | 0.434 |
| Stressed in 21 days | 0.082 | 0.300 | 0.516 | **0.622** |

### Is the improvement real? (block bootstrap, 95% CI)

| Target | Model vs. best baseline | Difference | 95% CI | Verdict |
|---|---|---|---|---|
| Stressed in 5 days | Logistic vs. no change | +0.158 | **+0.097 to +0.281** | ✅ Clearly better |
| Stressed in 5 days | LightGBM vs. no change | +0.167 | **+0.107 to +0.283** | ✅ Clearly better |
| Stressed in 21 days | Logistic vs. HMM | +0.064 | −0.136 to +0.350 | Better, but not conclusive |
| Stressed in 21 days | LightGBM vs. HMM | +0.105 | −0.056 to +0.343 | Better, but not conclusive |
| Bear in 5 / 21 days | Models vs. MA200 rule | −0.07 to +0.02 | all include 0 | Tie |

### Timeline

![Holdout timeline](figures/holdout_timeline.png)

**Stress (top):**

| Stress episode | Length | Logistic | LightGBM | HMM |
|---|---|---|---|---|
| 2022-02-08 (Russia–Ukraine, rate hikes) | 39 days | +2 | +2 | +12 |
| 2022-04-28 | 55 days | +1 | +1 | +0 |
| 2022-09-13 | 10 days | +0 | +0 | +0 |
| 2022-09-28 | 52 days | +1 | **−11** | +1 |
| 2025-04-03 (tariff shock) | 27 days | +0 | +0 | +0 |

*(Signal start relative to the first stressed day: negative = warned in advance.)*

- Every stress episode was recognized within 0–2 days. The only advance warning was LightGBM's signal before the late-September 2022 episode, still on from the turbulence just before.
- In the calm years (2023–2024) the probabilities stayed low, with no false stress signals above 50%.
- The April 2025 tariff shock wasn't anticipated (probability under 0.2 beforehand), but it was flagged on day 0.

**Bear (bottom):**

| Method | First "bear within 21 days" signal | Days after the Jan 2022 peak | Market fall at signal |
|---|---|---|---|
| Price < 200-day MA | 2022-01-21 | 13 | −8% |
| Logistic regression | 2022-01-27 | 17 | −10% |
| LightGBM | 2022-01-25 | 15 | −9% |
| 2-state HMM | 2022-02-03 | 22 | −7% |

- In 2022 the models flagged the bear market much faster than their historical average: 2–3 weeks after the peak, vs. 1–3 months in validation.
- **False alarms:**
  - **LightGBM** stayed high from Oct 2022 to Mar 2023, *after* the bear market's trough. The market was still deep below its peak, but the labeled bear market was already over.
  - **Logistic regression** gave a short false alarm in April 2025: the tariff sell-off fell about 19%, just short of a bear market.

## Validation vs. holdout

- **Stress, 5 days:** almost identical (0.94–0.95 vs. 0.93). The strongest and most stable result.
- **Stress, 21 days:** PR-AUC is lower than in validation (0.55–0.59 vs. 0.65–0.66), but stress was much rarer in the holdout (16% of days vs. 27%), and the baselines fell even more (no change: 0.38 vs. 0.51). The models' **margin over "no change" grew** (+0.17 to +0.21 vs. +0.14 to +0.15).
- **Bear:** all methods score higher than in validation, because one long, clear bear market dominates the period. Models and the 200-day-MA rule are statistically tied, as in 2000–2021.
- **SVM:** the best on several holdout targets, but the worst on fixed-stress 21-day (0.192). It was third in validation. Choosing it now would mean picking a model *because of* the holdout, which the holdout exists to prevent. **The recommended models stay logistic regression and LightGBM.** SVM's holdout result is reported as an observation for the literature discussion.

## Final conclusions

1. **Short-term stress is reliably predictable.** For "stressed within 5 trading days", the models beat all baselines in 42 years of validation *and* in the untouched holdout, with a bootstrap interval clearly above zero.
2. **One-month stress prediction adds value, with more uncertainty.** The models beat the baselines in validation and the holdout, and are much better at predicting stress *starting* (onset PR-AUC 0.52–0.62 vs. 0.08 for "no change"). The holdout alone is too short to make the 21-day gain statistically conclusive.
3. **Bear-market starts can't be predicted.** Models recognize bear markets after the market has already fallen (2–3 weeks in 2022, 1–3 months historically) and are no better than a simple 200-day-moving-average rule.
4. **The models are fast recognizers, not prophets.** Advance warning before a stress episode is rare. Their value is turning "something is happening" into a calibrated probability within days, with fewer false alarms than simple rules.
5. **Recommended setup:** logistic regression (simple, interpretable, stable) with LightGBM as a check. 13 price-based features from 1962; no VIX or volume.

## Limitations

- **Small holdout:** 1 bear market and 5 major stress episodes, so confidence intervals are wide.
- **Hindsight in the bear label:** the start is dated at the peak, which is only known later. The models were trained without it, but "days after peak" can only be measured after the fact.
- **Label choices:** results depend on the definitions (20% rule, 5-year stress window). The robustness labels gave the same picture in validation. In the holdout, LightGBM slipped below the HMM on the fixed-stress 21-day target (0.359 vs. 0.375), while logistic regression stayed ahead (0.437).
- **Data:** a single source (Yahoo Finance); 4 suspected bad prints before 1962 were corrected by rule.
- **No trading test:** the project measures prediction quality, not profits after costs.
