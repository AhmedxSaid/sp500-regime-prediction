# Model results (modern features)

Generated 2026-10-25 15:53 by `run_models.py`. Walk-forward out-of-sample predictions for **2000–2021** (one test year at a time; models retrained each year on all earlier data from 1990, with a 21-day gap). The 2022+ holdout is not used.

Features (16): `ret_1d`, `mom_1m`, `mom_3m`, `mom_12m`, `dist_ma50`, `dist_ma200`, `drawdown`, `below_peak_regime`, `rv_21d`, `rv_63d`, `parkinson_21d`, `rv_vs_5y_p80`, `gap_open`, `vix`, `vix_rv_spread`, `vix_chg_5d`

**How to read the metrics**

- **PR-AUC:** how well the method ranks future regime days above normal days (1 = perfect; a random guess scores the base rate).
- **Onset PR-AUC:** the same, but only on days when the regime is **not active yet**: can it predict a regime *starting*? This is where a model must beat "no change".
- **Brier:** mean squared error of the probability (lower is better; 0 = perfect).
- **Flips/yr:** how often the 50% signal switches on/off per year (compare with the true number).
- Bold = best in the column.

## Main targets

### `bear` in 5 days

Base rate 12.8% of days; on not-yet-bear days, 0.6% turn bear within 5 days. True regime switches: 0.5 per year.

| Method | PR-AUC | Onset PR-AUC | Brier | Flips/yr |
|---|---|---|---|---|
| Baseline: no change | 0.135 | 0.006 | 0.445 | 0.3 |
| Baseline: price < 200-day MA | 0.362 | 0.006 | 0.188 | 7.4 |
| Baseline: 2-state HMM | **0.382** | 0.014 | 0.164 | 10.0 |
| Logistic regression | 0.153 | **0.082** | 0.146 | 2.0 |
| LightGBM | 0.225 | 0.005 | 0.128 | 6.2 |
| SVM (RBF) | 0.290 | 0.011 | **0.113** | 5.2 |

### `bear` in 21 days

Base rate 13.0% of days; on not-yet-bear days, 2.6% turn bear within 21 days. True regime switches: 0.5 per year.

| Method | PR-AUC | Onset PR-AUC | Brier | Flips/yr |
|---|---|---|---|---|
| Baseline: no change | 0.129 | 0.024 | 0.460 | 0.3 |
| Baseline: price < 200-day MA | 0.317 | 0.028 | 0.208 | 7.4 |
| Baseline: 2-state HMM | **0.402** | **0.108** | 0.128 | 11.5 |
| Logistic regression | 0.121 | 0.040 | 0.161 | 2.3 |
| LightGBM | 0.207 | 0.023 | 0.129 | 5.6 |
| SVM (RBF) | 0.294 | 0.045 | **0.117** | 5.8 |

### `stressed` in 5 days

Base rate 24.1% of days; on not-yet-stressed days, 3.5% turn stressed within 5 days. True regime switches: 4.1 per year.

| Method | PR-AUC | Onset PR-AUC | Brier | Flips/yr |
|---|---|---|---|---|
| Baseline: no change | 0.816 | 0.035 | 0.054 | 4.1 |
| Baseline: price < 200-day MA | – | – | – | – |
| Baseline: 2-state HMM | 0.780 | 0.136 | 0.101 | 10.0 |
| Logistic regression | 0.951 | 0.276 | 0.043 | 4.8 |
| LightGBM | **0.954** | **0.278** | **0.039** | 5.7 |
| SVM (RBF) | 0.738 | 0.145 | 0.144 | 6.8 |

### `stressed` in 21 days

Base rate 23.8% of days; on not-yet-stressed days, 10.2% turn stressed within 21 days. True regime switches: 4.1 per year.

| Method | PR-AUC | Onset PR-AUC | Brier | Flips/yr |
|---|---|---|---|---|
| Baseline: no change | 0.526 | 0.102 | 0.158 | 4.1 |
| Baseline: price < 200-day MA | – | – | – | – |
| Baseline: 2-state HMM | 0.618 | 0.158 | 0.139 | 11.5 |
| Logistic regression | 0.674 | 0.223 | 0.130 | 9.5 |
| LightGBM | **0.688** | **0.283** | **0.121** | 10.7 |
| SVM (RBF) | 0.583 | 0.148 | 0.168 | 12.5 |

## Robustness targets

### `bear_merged` in 5 days

Base rate 15.5% of days; on not-yet-bear_merged days, 0.4% turn bear_merged within 5 days. True regime switches: 0.3 per year.

| Method | PR-AUC | Onset PR-AUC | Brier | Flips/yr |
|---|---|---|---|---|
| Baseline: no change | 0.183 | 0.004 | 0.418 | 0.3 |
| Baseline: price < 200-day MA | 0.451 | 0.004 | 0.163 | 7.4 |
| Baseline: 2-state HMM | **0.467** | **0.006** | 0.157 | 10.0 |
| Logistic regression | 0.174 | 0.003 | 0.191 | 3.7 |
| LightGBM | 0.184 | 0.003 | 0.190 | 6.8 |
| SVM (RBF) | 0.307 | 0.004 | **0.150** | 8.4 |

### `bear_merged` in 21 days

Base rate 15.8% of days; on not-yet-bear_merged days, 1.8% turn bear_merged within 21 days. True regime switches: 0.3 per year.

| Method | PR-AUC | Onset PR-AUC | Brier | Flips/yr |
|---|---|---|---|---|
| Baseline: no change | 0.174 | 0.018 | 0.433 | 0.3 |
| Baseline: price < 200-day MA | 0.408 | 0.018 | 0.181 | 7.4 |
| Baseline: 2-state HMM | **0.482** | **0.039** | **0.129** | 11.5 |
| Logistic regression | 0.136 | 0.011 | 0.203 | 2.2 |
| LightGBM | 0.170 | 0.012 | 0.189 | 7.4 |
| SVM (RBF) | 0.257 | 0.014 | 0.157 | 8.4 |

### `stressed_fixed` in 5 days

Base rate 13.2% of days; on not-yet-stressed_fixed days, 2.3% turn stressed_fixed within 5 days. True regime switches: 2.9 per year.

| Method | PR-AUC | Onset PR-AUC | Brier | Flips/yr |
|---|---|---|---|---|
| Baseline: no change | 0.743 | 0.023 | 0.039 | 2.9 |
| Baseline: price < 200-day MA | – | – | – | – |
| Baseline: 2-state HMM | 0.589 | 0.142 | 0.123 | 10.0 |
| Logistic regression | 0.899 | **0.316** | 0.039 | 3.2 |
| LightGBM | **0.914** | 0.305 | **0.034** | 3.5 |
| SVM (RBF) | 0.691 | 0.170 | 0.073 | 3.4 |

### `stressed_fixed` in 21 days

Base rate 13.1% of days; on not-yet-stressed_fixed days, 6.7% turn stressed_fixed within 21 days. True regime switches: 2.9 per year.

| Method | PR-AUC | Onset PR-AUC | Brier | Flips/yr |
|---|---|---|---|---|
| Baseline: no change | 0.370 | 0.067 | 0.116 | 2.9 |
| Baseline: price < 200-day MA | – | – | – | – |
| Baseline: 2-state HMM | 0.426 | 0.178 | 0.118 | 11.5 |
| Logistic regression | **0.644** | **0.255** | **0.071** | 3.5 |
| LightGBM | 0.616 | 0.236 | 0.079 | 3.1 |
| SVM (RBF) | 0.382 | 0.144 | 0.102 | 1.1 |

## Early warning (21-day horizon)

For each regime start in the test years: consecutive trading days **before** the start on which the method's probability was already ≥ 50% (0 = no warning). Median over all starts, and the share of starts with at least 5 days of warning.

| Method | Stress starts: median days | ≥ 5 days warned | Bear starts: median days | ≥ 5 days warned |
|---|---|---|---|---|
| Baseline: no change | 0 | 0% | 0 | 40% |
| Baseline: price < 200-day MA | – | – | 0 | 20% |
| Baseline: 2-state HMM | 0 | 20% | 0 | 20% |
| Logistic regression | 0 | 4% | 0 | 20% |
| LightGBM | 0 | 7% | 0 | 0% |
| SVM (RBF) | 0 | 36% | 0 | 20% |

(45 stress starts and 5 bear starts in the test years. A bear start is dated at the market peak, which is only known in hindsight, so "warning" before it means the signal was already on at the top, e.g. because the market was still below an older high.)

## Timeline

Best model by mean PR-AUC on the main targets: **LightGBM**.

![timeline](../docs/figures/model_timeline_modern.png)

