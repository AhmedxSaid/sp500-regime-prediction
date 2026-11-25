# Min / max price prediction: results

Generated on 2026-11-25 20:07 by `run_range_models.py` (7.3 min).

Target: highest high (`up`) and lowest low (`down`) over the next *h* trading days, as a log
move from today's close. Models give the 10%, 50% and 90% quantiles.

* **MAE %**: mean absolute error of the median forecast, in % of price (lower is better).
* **Pinball**: average quantile loss x100 (lower is better; scores the whole band).
* **Coverage 80%**: share of days where the actual value fell inside the 10-90% band (target 0.80).
* **Skill**: 1 - MAE / MAE of the volatility-band baseline (> 0 beats it).
* **Corr**: correlation of the median forecast with the actual value.

## Walk-forward 2000-2021

### max high, next 5 days (n = 5536)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| Ensemble (XGB + Cat + LSTM) | 0.934 | **0.288** | 0.79 | 2.92 | 0.53 | +0.017 |
| CatBoost | 0.933 | 0.289 | 0.78 | 2.83 | 0.53 | +0.018 |
| XGBoost | 0.944 | 0.290 | 0.79 | 2.94 | 0.51 | +0.007 |
| LSTM | 0.941 | 0.291 | 0.80 | 3.00 | 0.51 | +0.010 |
| Baseline: volatility band | 0.950 | 0.297 | 0.77 | 2.96 | 0.48 | +0.000 |
| Baseline: historical quantiles | 1.038 | 0.346 | 0.78 | 3.20 | -0.04 | -0.092 |

### min low, next 5 days (n = 5536)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| Ensemble (XGB + Cat + LSTM) | 1.248 | **0.404** | 0.72 | 3.21 | 0.44 | +0.012 |
| XGBoost | 1.251 | 0.405 | 0.72 | 3.16 | 0.44 | +0.010 |
| LSTM | 1.257 | 0.405 | 0.73 | 3.34 | 0.43 | +0.005 |
| CatBoost | 1.249 | 0.407 | 0.72 | 3.11 | 0.43 | +0.011 |
| Baseline: volatility band | 1.263 | 0.410 | 0.71 | 3.23 | 0.42 | +0.000 |
| Baseline: historical quantiles | 1.352 | 0.459 | 0.73 | 3.53 | -0.10 | -0.071 |

### max high, next 21 days (n = 5536)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| LSTM | 1.733 | **0.538** | 0.82 | 5.84 | 0.47 | +0.001 |
| Ensemble (XGB + Cat + LSTM) | 1.718 | 0.541 | 0.76 | 5.22 | 0.50 | +0.010 |
| Baseline: volatility band | 1.736 | 0.544 | 0.80 | 5.99 | 0.47 | +0.000 |
| XGBoost | 1.740 | 0.562 | 0.74 | 5.16 | 0.50 | -0.002 |
| CatBoost | 1.757 | 0.565 | 0.70 | 4.66 | 0.48 | -0.012 |
| Baseline: historical quantiles | 1.870 | 0.605 | 0.81 | 6.10 | -0.11 | -0.077 |

### min low, next 21 days (n = 5536)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| Baseline: volatility band | 2.684 | **0.905** | 0.73 | 6.60 | 0.34 | +0.000 |
| LSTM | 2.705 | 0.907 | 0.74 | 6.43 | 0.34 | -0.008 |
| Ensemble (XGB + Cat + LSTM) | 2.672 | 0.912 | 0.69 | 5.75 | 0.35 | +0.005 |
| XGBoost | 2.678 | 0.924 | 0.67 | 5.64 | 0.34 | +0.002 |
| CatBoost | 2.687 | 0.940 | 0.63 | 5.18 | 0.32 | -0.001 |
| Baseline: historical quantiles | 2.823 | 0.971 | 0.74 | 7.17 | -0.17 | -0.052 |

## Holdout 2022+

### max high, next 5 days (n = 1224)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| CatBoost | 0.943 | **0.296** | 0.73 | 2.63 | 0.46 | +0.051 |
| XGBoost | 0.943 | 0.296 | 0.74 | 2.67 | 0.45 | +0.050 |
| Ensemble (XGB + Cat + LSTM) | 0.946 | 0.297 | 0.74 | 2.68 | 0.45 | +0.047 |
| LSTM | 0.961 | 0.302 | 0.76 | 2.75 | 0.42 | +0.033 |
| Baseline: volatility band | 0.993 | 0.313 | 0.73 | 2.74 | 0.35 | +0.000 |
| Baseline: historical quantiles | 1.030 | 0.334 | 0.77 | 3.16 | nan | -0.037 |

### min low, next 5 days (n = 1224)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| Ensemble (XGB + Cat + LSTM) | 1.239 | **0.401** | 0.73 | 3.24 | 0.24 | +0.009 |
| CatBoost | 1.242 | 0.401 | 0.73 | 3.27 | 0.24 | +0.007 |
| XGBoost | 1.246 | 0.402 | 0.72 | 3.26 | 0.24 | +0.004 |
| LSTM | 1.240 | 0.403 | 0.72 | 3.19 | 0.24 | +0.008 |
| Baseline: volatility band | 1.251 | 0.408 | 0.70 | 3.13 | 0.24 | +0.000 |
| Baseline: historical quantiles | 1.268 | 0.421 | 0.75 | 3.62 | -0.00 | -0.014 |

### max high, next 21 days (n = 1208)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| CatBoost | 1.783 | **0.546** | 0.72 | 4.79 | 0.46 | +0.049 |
| XGBoost | 1.802 | 0.552 | 0.72 | 4.95 | 0.44 | +0.039 |
| Ensemble (XGB + Cat + LSTM) | 1.803 | 0.552 | 0.72 | 4.93 | 0.44 | +0.038 |
| LSTM | 1.847 | 0.567 | 0.74 | 5.06 | 0.36 | +0.015 |
| Baseline: volatility band | 1.875 | 0.569 | 0.75 | 5.52 | 0.41 | +0.000 |
| Baseline: historical quantiles | 1.927 | 0.609 | 0.79 | 6.00 | nan | -0.028 |

### min low, next 21 days (n = 1208)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| Baseline: volatility band | 2.512 | **0.809** | 0.71 | 6.40 | 0.23 | +0.000 |
| XGBoost | 2.519 | 0.818 | 0.69 | 6.11 | 0.20 | -0.002 |
| Ensemble (XGB + Cat + LSTM) | 2.532 | 0.822 | 0.68 | 6.07 | 0.20 | -0.008 |
| LSTM | 2.549 | 0.825 | 0.70 | 6.30 | 0.19 | -0.014 |
| Baseline: historical quantiles | 2.540 | 0.830 | 0.75 | 7.27 | nan | -0.011 |
| CatBoost | 2.556 | 0.838 | 0.65 | 5.81 | 0.17 | -0.017 |
