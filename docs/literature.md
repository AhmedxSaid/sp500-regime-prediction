# Related Work

## SVM for S&P 500 index prediction — Sheta et al.

The study compared Support Vector Regression (RBF kernel) with Multiple Linear Regression and a Multilayer Perceptron for predicting the S&P 500 index.

- Data: 27 features and 1,192 daily observations (Dec 2009 – Sep 2014), sampled weekly; 100 observations for training and 43 for testing.
- Result: SVR outperformed MLP and MLR on both training and test sets.
- Source: ScienceDirect.

**Possible reproduction:**

| Item | Choice |
|---|---|
| Inputs | Open, high, low, close, volume, technical indicators, lagged market variables |
| Model | SVR with RBF kernel |
| Baselines | Multiple linear regression, MLP |
| Target | `P_{t+1} = f(X_t)` or, better, `r_{t+1} = f(X_t)` |
| Metrics | MAE, RMSE, MAPE, R² |

*Caveat:* the test set is very small (43 points). Forecasting price levels gives R² close to 1 even for a naive "tomorrow = today" model, so any comparison should use returns and include the naive baseline.

## SVM + financial news — Chen et al. (2017)

The approach combined SVM, AdaBoost.M2, Naive Bayes and a Quantum Genetic Algorithm to test whether financial news improves trading-signal prediction.

- Their best model was **AdaBoost QGA-SVM *without* news**. More information does not automatically improve forecasting.

**Possible research question:** *Does financial-news sentiment improve SVM-based prediction of S&P 500 direction?* Compare `SVM_market` with `SVM_market+sentiment`.

## SVM on simulated S&P 500-like markets — Rosillo et al.

"Stock Market Simulation Using Support Vector Machines" (Wiley Online Library).

- GARCH models fitted to the S&P 500 were used to generate simulated markets: high volatility, bearish, bullish and low volatility.
- The SVM used technical rules (RSI, MACD) to classify the market as bullish or bearish.

```
S&P 500 → GARCH → simulated markets → technical indicators → SVM → bull/bear
```

This is relevant to our project because it measures how classifier performance changes across market regimes.

<!-- TODO: add full citations (authors, year, journal, DOI) -->
