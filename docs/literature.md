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

## What our results add

- **Sheta et al.:** in our regime task, with 42 test years instead of 43 weekly points, the RBF SVM ranked **third**, behind logistic regression and LightGBM (see [06_modeling.md](06_modeling.md)).
- **Chen et al.:** consistent with their finding, adding data sources (SPY volume, VIX) did not improve our final model ([06_modeling.md](06_modeling.md), feature group tests).
- **Rosillo et al.:** our per-crisis analysis gives a real-data counterpart to their per-regime simulation; their GARCH approach remains future work.

## References

1. Rosillo, R., Giner, J., & de la Fuente, D. (2014). Stock market simulation using support vector machines. *Journal of Forecasting*, 33(6), 488–500. [doi:10.1002/for.2302](https://doi.org/10.1002/for.2302) (verified).
2. Sheta, A. F., Ahmed, S. E. M., & Faris, H. (2015). A comparison between regression, artificial neural networks and support vector machines for predicting stock market index. **TODO: verify journal, volume and pages** (secondary sources disagree on the venue).
3. Chen et al. (2017). AdaBoost QGA-SVM study on financial news and trading signals. **TODO: complete from the team's original source** (a DOI was in the initial research notes; not found by web search).
4. Pagan, A. R., & Sossounov, K. A. (2003). A simple framework for analysing bull and bear markets. *Journal of Applied Econometrics*, 18(1), 23–46.
5. Lunde, A., & Timmermann, A. (2004). Duration dependence in stock prices: an analysis of bull and bear markets. *Journal of Business & Economic Statistics*, 22(3), 253–273.
6. Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61–65.
7. Ke, G., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
8. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
