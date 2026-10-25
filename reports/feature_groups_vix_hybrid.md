# VIX on the full history (LightGBM, VIX missing before 1990)

Generated 2026-10-25 15:59 by `run_feature_groups.py --vix-hybrid`. Both sets train LightGBM from 1962-01-01 on identical rows; the VIX features are empty (missing) before 1990 and LightGBM learns to handle that. Test years **2000–2021**. The 2022+ holdout is not used.

**Rule (same thresholds as the group tests, LightGBM only):** keep VIX if mean PR-AUC over the 4 main targets improves by ≥ 0.01 and mean onset PR-AUC drops by no more than 0.01.

| | Δ mean PR-AUC (main) | Δ mean onset PR-AUC (main) | Δ mean PR-AUC (robustness) | Years better (`stressed` 21d) | Decision |
|---|---|---|---|---|---|
| VIX added | -0.018 | +0.001 | -0.018 | 12/16 | ❌ drop |

## PR-AUC

| Feature set | bear 5d | bear 21d | bear_merged 5d | bear_merged 21d | stressed 5d | stressed 21d | stressed_fixed 5d | stressed_fixed 21d |
|---|---|---|---|---|---|---|---|---|
| main (13 features) | 0.312 | 0.247 | 0.340 | 0.260 | 0.955 | 0.721 | 0.929 | 0.602 |
| main + vix, missing before 1990 | 0.275 | 0.230 | 0.280 | 0.224 | 0.952 | 0.703 | 0.928 | 0.629 |

## Onset PR-AUC

| Feature set | bear 5d | bear 21d | bear_merged 5d | bear_merged 21d | stressed 5d | stressed 21d | stressed_fixed 5d | stressed_fixed 21d |
|---|---|---|---|---|---|---|---|---|
| main (13 features) | 0.005 | 0.020 | 0.003 | 0.012 | 0.313 | 0.302 | 0.330 | 0.206 |
| main + vix, missing before 1990 | 0.006 | 0.025 | 0.003 | 0.012 | 0.304 | 0.307 | 0.340 | 0.210 |

## Brier (lower is better)

| Feature set | bear 5d | bear 21d | bear_merged 5d | bear_merged 21d | stressed 5d | stressed 21d | stressed_fixed 5d | stressed_fixed 21d |
|---|---|---|---|---|---|---|---|---|
| main (13 features) | 0.108 | 0.117 | 0.130 | 0.142 | 0.039 | 0.113 | 0.031 | 0.080 |
| main + vix, missing before 1990 | 0.113 | 0.120 | 0.140 | 0.150 | 0.040 | 0.117 | 0.030 | 0.076 |

