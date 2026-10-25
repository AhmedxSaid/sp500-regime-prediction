# 1. Business Understanding

> Draft. Refine the questions and success criteria as a team.

## Context

The S&P 500 is the main benchmark for US equities. Index funds, pensions and risk systems all track it. Day-to-day price moves are close to unpredictable: returns have almost no autocorrelation, and many published "price prediction" results do not hold up out of sample. Market **regimes** are a different matter. Volatility clusters, drawdowns last for months, and stressed periods tend to persist, which gives a model something to learn.

For an investor or risk manager, knowing *which regime we are entering* is often more useful than a point forecast of tomorrow's price:

- **Bear markets** (a fall of more than 20% from a peak) cause the largest losses. Examples: 2000–02, 2007–09, 2020, 2022.
- **Stressed periods** (high realized volatility) are when correlations rise, liquidity drops, and position sizes or hedges need adjusting.

## Problem statement

Given market data available at the close of day *t*, estimate the probability that the S&P 500 will be in a **bear** and/or **stressed** regime on day *t + h* (h = 5 and 21 trading days).

## Questions

1. Can regime models built on price, volatility and volume features predict the regime *h* days ahead better than simple rules?
2. How many days of warning does the model give before known stress and bear episodes?
3. Which features drive the predictions (momentum, distance from moving averages, volatility, volume, futures)?
4. Do results stay stable across decades, label definitions and horizons?
5. *(Extension)* Do extra data sources (VIX term structure, sector ETFs, news sentiment) improve out-of-sample results?
6. How does an SVM compare with logistic regression and gradient boosting on this task (see [literature.md](literature.md))? **Decision:** the SVM is one of the three compared models, not a separate track.

## Success criteria

**Data-science criteria** (measured on walk-forward tests and the 2022+ holdout):

| Criterion | Target |
|---|---|
| PR-AUC for the stressed class | Clearly above the best baseline (persistence, 200-day MA rule, 2-state HMM) |
| Brier score | Lower than the baselines; calibration curve close to the diagonal |
| Early warning | Positive median lead time (days) before the start of each stress episode |
| Signal stability | Few flip-flops (number of regime switches per year close to the true number) |
| Robustness | Ranking vs. baselines holds when thresholds, *h* and start date change |

**Project criteria:**

- A fully reproducible pipeline (download → prep → features → model) that runs from the README.
- No look-ahead bias anywhere in labels, features or validation.
- A clear report: model vs. baselines, a probability timeline against actual regimes, and limitations.

## Scope and limitations

- **In scope:** daily data, index-level regimes, classification of future regime.
- **Out of scope:** intraday trading, individual stocks, live trading.
- **Known risks:** the sample has only a handful of bear markets, results depend on the label definition, and market structure changes over time (non-stationarity).
