# 4. Exploratory Data Analysis

Notebook: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). Figures are saved to `reports/figures/`.

## Checklist

### Structure and quality
- [ ] Rows, date range and columns for each series
- [ ] Missing values per column over time (when each series and feature becomes available)
- [ ] Review the data quality report (`reports/data_quality.csv`)
- [ ] Inspect flagged large moves: genuine (1987, 2008, 2020) or errors?

### Price and returns
- [ ] Log-price chart of `^GSPC` for the full history
- [ ] Distribution of daily log returns vs. a normal distribution (fat tails, skew, kurtosis)
- [ ] Autocorrelation of returns (≈ 0) vs. squared/absolute returns (volatility clustering)
- [ ] Return statistics by decade

### Volatility
- [ ] 21-day realized volatility over time, with the 5-year rolling 80th-percentile threshold and the fixed 25% level
- [ ] Close-to-close vs. Parkinson vs. Garman-Klass volatility

### Regimes
- [ ] Price chart with bear periods shaded (both methods)
- [ ] Price chart with stressed periods shaded
- [ ] Check that 2000–02, 2007–09, 2020 and 2022 are captured
- [ ] Class balance: % of days bear / stressed
- [ ] Regime durations (how long episodes last) and number of episodes
- [ ] How often bear and stressed overlap
- [ ] Returns and volatility by regime

### Features vs. target
- [ ] Correlation heatmap of features (redundancy)
- [ ] Feature distributions split by future regime (`target_*_h21`)
- [ ] Persistence: P(regime at t+h | regime at t). This shows how strong the "no-change" baseline is

## Findings

<!-- Write key findings here as you go, with a figure link for each, e.g.
- Daily returns have excess kurtosis of ~X; tails are much fatter than normal ([fig](../reports/figures/returns_hist.png)).
-->
