# Statistical tests and validation vs. holdout

Generated 2026-10-27 16:08 by `run_statistical_tests.py`. Test 1 uses the full history; Tests 2–4 use 1962–2021, the modeling period. Daily observations are autocorrelated and samples are large, so p-values overstate the evidence: read the effect sizes.

## Hypothesis tests

| Test | H0 | n | Statistic | p-value | Effect size | Decision |
|---|---|---|---|---|---|---|
| Ljung–Box, returns | No autocorrelation up to lag 20 | 24,820 | Q(20) = 98.4 | < 0.001 | lag-1 autocorrelation -0.007 | Reject H0 |
| Ljung–Box, absolute returns | No autocorrelation up to lag 20 | 24,820 | Q(20) = 37,890.5 | < 0.001 | lag-1 autocorrelation +0.318 | Reject H0 |
| Chi-square, stressed today vs. in 21 days | Independent | 15,105 | χ²(1) = 3,804.9 | < 0.001 | Cramér's V = 0.50 | Reject H0 |
| Mann–Whitney, 21-day realized vol, by stressed in 21 days | Same distribution | 3,875 vs. 11,230 | U = 3.31e+07 | < 0.001 | r = +0.52 (medians +16.7% vs. +10.8%) | Reject H0 |
| Mann–Whitney, distance from 200-day average, by bear in 21 days | Same distribution | 2,302 vs. 12,803 | U = 6.3e+06 | < 0.001 | r = -0.57 (medians -4.0% vs. +4.8%) | Reject H0 |

Smallest expected count in the chi-square test: 994 (condition: ≥ 5).

Jarque–Bera normality test of daily returns: JB = 361,599, p < 0.001 (skew -0.46, excess kurtosis 18.7).

Contingency table (rows: stressed today; columns: stressed in 21 days):

| | Not stressed in 21 days | Stressed in 21 days |
|---|---|---|
| Not stressed today | 9,795 | 1,435 |
| Stressed today | 1,435 | 2,440 |

## Validation vs. holdout

| Measure | Validation 1962–2021 | Holdout 2022–2026 |
|---|---|---|
| Trading days | 15,105 | 1,207 |
| Annual return / volatility | 7.0% / 16.4% | 9.8% / 17.1% |
| Share of days in a bear market | 15.2% | 16.2% |
| Share of days stressed (main label) | 25.7% | 15.5% |
| Share of days stressed (fixed 25% label) | 7.1% | 10.4% |
| Median 21-day realized volatility | 11.8% | 13.6% |
| Deepest drawdown from a peak | -56.8% | -25.4% |
| Worst day (log return) | -22.9% | -6.2% |

Kolmogorov–Smirnov test on 21-day realized volatility, validation vs. holdout: D = 0.176, p < 0.001.
