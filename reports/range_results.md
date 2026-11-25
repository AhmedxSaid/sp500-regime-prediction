# Min / max price prediction: results

Generated on 2026-11-25 18:41 by `notebooks/15_range_model_comparison.ipynb`.

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
| Ensemble (XGB + Cat + LSTM) | 0.934 | **0.288** | 0.79 | 2.92 | 0.53 | +0.018 |
| CatBoost | 0.933 | 0.289 | 0.78 | 2.83 | 0.53 | +0.018 |
| XGBoost | 0.943 | 0.290 | 0.79 | 2.94 | 0.52 | +0.008 |
| LSTM | 0.941 | 0.291 | 0.80 | 3.00 | 0.51 | +0.010 |
| Baseline: volatility band | 0.950 | 0.297 | 0.77 | 2.96 | 0.48 | +0.000 |
| Baseline: historical quantiles | 1.038 | 0.346 | 0.78 | 3.20 | -0.04 | -0.092 |

### min low, next 5 days (n = 5536)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| Ensemble (XGB + Cat + LSTM) | 1.248 | **0.404** | 0.72 | 3.21 | 0.44 | +0.012 |
| XGBoost | 1.250 | 0.405 | 0.72 | 3.17 | 0.44 | +0.010 |
| LSTM | 1.257 | 0.405 | 0.73 | 3.34 | 0.43 | +0.005 |
| CatBoost | 1.249 | 0.407 | 0.72 | 3.11 | 0.43 | +0.011 |
| Baseline: volatility band | 1.263 | 0.410 | 0.71 | 3.23 | 0.42 | +0.000 |
| Baseline: historical quantiles | 1.352 | 0.459 | 0.73 | 3.53 | -0.10 | -0.071 |

### max high, next 21 days (n = 5536)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| LSTM | 1.733 | **0.538** | 0.82 | 5.84 | 0.48 | +0.002 |
| Ensemble (XGB + Cat + LSTM) | 1.723 | 0.542 | 0.76 | 5.22 | 0.50 | +0.007 |
| Baseline: volatility band | 1.736 | 0.544 | 0.80 | 5.99 | 0.47 | +0.000 |
| CatBoost | 1.757 | 0.565 | 0.70 | 4.66 | 0.48 | -0.012 |
| XGBoost | 1.756 | 0.567 | 0.74 | 5.17 | 0.49 | -0.012 |
| Baseline: historical quantiles | 1.870 | 0.605 | 0.81 | 6.10 | -0.11 | -0.077 |

### min low, next 21 days (n = 5536)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| Baseline: volatility band | 2.684 | **0.905** | 0.73 | 6.60 | 0.34 | +0.000 |
| LSTM | 2.706 | 0.906 | 0.74 | 6.44 | 0.34 | -0.008 |
| Ensemble (XGB + Cat + LSTM) | 2.673 | 0.911 | 0.69 | 5.76 | 0.35 | +0.004 |
| XGBoost | 2.680 | 0.921 | 0.67 | 5.67 | 0.34 | +0.002 |
| CatBoost | 2.687 | 0.940 | 0.63 | 5.18 | 0.32 | -0.001 |
| Baseline: historical quantiles | 2.823 | 0.971 | 0.74 | 7.17 | -0.17 | -0.052 |

## Holdout 2022+

### max high, next 5 days (n = 1202)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| CatBoost | 0.945 | **0.297** | 0.73 | 2.63 | 0.45 | +0.050 |
| XGBoost | 0.946 | 0.297 | 0.73 | 2.68 | 0.45 | +0.049 |
| Ensemble (XGB + Cat + LSTM) | 0.948 | 0.298 | 0.74 | 2.68 | 0.45 | +0.046 |
| LSTM | 0.962 | 0.303 | 0.76 | 2.75 | 0.42 | +0.032 |
| Baseline: volatility band | 0.994 | 0.313 | 0.74 | 2.74 | 0.35 | +0.000 |
| Baseline: historical quantiles | 1.031 | 0.335 | 0.77 | 3.16 | nan | -0.038 |

### min low, next 5 days (n = 1202)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| CatBoost | 1.218 | **0.393** | 0.73 | 3.27 | 0.25 | +0.007 |
| Ensemble (XGB + Cat + LSTM) | 1.216 | 0.393 | 0.73 | 3.24 | 0.25 | +0.009 |
| XGBoost | 1.223 | 0.394 | 0.73 | 3.25 | 0.24 | +0.003 |
| LSTM | 1.217 | 0.395 | 0.73 | 3.19 | 0.24 | +0.008 |
| Baseline: volatility band | 1.227 | 0.400 | 0.71 | 3.14 | 0.25 | +0.000 |
| Baseline: historical quantiles | 1.248 | 0.414 | 0.75 | 3.62 | nan | -0.017 |

### max high, next 21 days (n = 1186)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| CatBoost | 1.793 | **0.549** | 0.72 | 4.81 | 0.46 | +0.053 |
| XGBoost | 1.811 | 0.554 | 0.73 | 4.98 | 0.44 | +0.043 |
| Ensemble (XGB + Cat + LSTM) | 1.814 | 0.555 | 0.73 | 4.96 | 0.43 | +0.042 |
| LSTM | 1.859 | 0.570 | 0.74 | 5.08 | 0.35 | +0.018 |
| Baseline: volatility band | 1.893 | 0.575 | 0.75 | 5.56 | 0.40 | +0.000 |
| Baseline: historical quantiles | 1.928 | 0.610 | 0.79 | 6.00 | nan | -0.019 |

### min low, next 21 days (n = 1186)

| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |
|---|---|---|---|---|---|---|
| Baseline: volatility band | 2.489 | **0.799** | 0.72 | 6.44 | 0.24 | +0.000 |
| XGBoost | 2.498 | 0.811 | 0.70 | 6.12 | 0.21 | -0.004 |
| Ensemble (XGB + Cat + LSTM) | 2.514 | 0.816 | 0.68 | 6.09 | 0.21 | -0.010 |
| LSTM | 2.529 | 0.819 | 0.70 | 6.33 | 0.20 | -0.016 |
| Baseline: historical quantiles | 2.528 | 0.828 | 0.75 | 7.27 | 0.00 | -0.016 |
| CatBoost | 2.539 | 0.833 | 0.66 | 5.83 | 0.18 | -0.020 |
