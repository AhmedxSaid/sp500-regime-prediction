# Feature group tests

Generated 2026-10-25 15:52 by `run_feature_groups.py`. Same rows for every feature set: training from 1993-02-26, walk-forward test years **2000–2021**, 21-day gap. Models: logistic regression and LightGBM (the two best in the main run). The 2022+ holdout is not used.

**Decision rule (fixed before testing):** keep a group only if mean PR-AUC over the 4 main targets (bear and stressed, 5 and 21 days) improves by **≥ 0.01** for **both** models, and mean onset PR-AUC drops by no more than 0.01 for either.

| Group | Features | Logistic regression: Δ PR-AUC | LightGBM: Δ PR-AUC | Δ onset PR-AUC (LR / LGBM) | Decision |
|---|---|---|---|---|---|
| spy_volume | `spy_vol_ratio_20d` | +0.005 | +0.002 | +0.011 / +0.010 | ❌ drop |
| vix | `vix`, `vix_rv_spread`, `vix_chg_5d` | +0.020 | +0.018 | +0.043 / +0.018 | ✅ keep |

## Main targets: PR-AUC by target

| Feature set | Model | bear 5d | bear 21d | stressed 5d | stressed 21d | Mean |
|---|---|---|---|---|---|---|
| base | Logistic regression | 0.143 | 0.113 | 0.938 | 0.569 | **0.441** |
| base | LightGBM | 0.237 | 0.233 | 0.952 | 0.685 | **0.527** |
| base + spy_volume | Logistic regression | 0.143 | 0.113 | 0.946 | 0.582 | **0.446** |
| base + spy_volume | LightGBM | 0.240 | 0.229 | 0.954 | 0.692 | **0.529** |
| base + vix | Logistic regression | 0.149 | 0.118 | 0.951 | 0.626 | **0.461** |
| base + vix | LightGBM | 0.276 | 0.266 | 0.954 | 0.683 | **0.545** |

Onset PR-AUC (main targets):

| Feature set | Model | bear 5d | bear 21d | stressed 5d | stressed 21d | Mean |
|---|---|---|---|---|---|---|
| base | Logistic regression | 0.077 | 0.033 | 0.171 | 0.155 | **0.109** |
| base | LightGBM | 0.005 | 0.022 | 0.263 | 0.263 | **0.138** |
| base + spy_volume | Logistic regression | 0.077 | 0.033 | 0.208 | 0.161 | **0.120** |
| base + spy_volume | LightGBM | 0.006 | 0.023 | 0.299 | 0.268 | **0.149** |
| base + vix | Logistic regression | 0.081 | 0.035 | 0.288 | 0.204 | **0.152** |
| base + vix | LightGBM | 0.006 | 0.024 | 0.289 | 0.305 | **0.156** |

## Robustness targets: PR-AUC by target

| Feature set | Model | bear_merged 5d | bear_merged 21d | stressed_fixed 5d | stressed_fixed 21d | Mean |
|---|---|---|---|---|---|---|
| base | Logistic regression | 0.167 | 0.132 | 0.900 | 0.626 | **0.456** |
| base | LightGBM | 0.265 | 0.279 | 0.921 | 0.612 | **0.519** |
| base + spy_volume | Logistic regression | 0.166 | 0.132 | 0.907 | 0.639 | **0.461** |
| base + spy_volume | LightGBM | 0.277 | 0.284 | 0.927 | 0.619 | **0.527** |
| base + vix | Logistic regression | 0.169 | 0.129 | 0.891 | 0.640 | **0.457** |
| base + vix | LightGBM | 0.275 | 0.282 | 0.907 | 0.620 | **0.521** |

Onset PR-AUC (robustness targets):

| Feature set | Model | bear_merged 5d | bear_merged 21d | stressed_fixed 5d | stressed_fixed 21d | Mean |
|---|---|---|---|---|---|---|
| base | Logistic regression | 0.003 | 0.011 | 0.225 | 0.256 | **0.124** |
| base | LightGBM | 0.003 | 0.012 | 0.333 | 0.270 | **0.155** |
| base + spy_volume | Logistic regression | 0.003 | 0.011 | 0.285 | 0.268 | **0.141** |
| base + spy_volume | LightGBM | 0.003 | 0.012 | 0.370 | 0.279 | **0.166** |
| base + vix | Logistic regression | 0.003 | 0.011 | 0.278 | 0.246 | **0.134** |
| base + vix | LightGBM | 0.003 | 0.013 | 0.308 | 0.279 | **0.151** |

## Year by year (`stressed` in 21 days)

Number of test years in which each set beat the base set on PR-AUC (years with both classes):

- Logistic regression: base + spy_volume: 14/16; base + vix: 14/16
- LightGBM: base + spy_volume: 9/16; base + vix: 9/16
