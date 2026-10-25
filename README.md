# S&P 500 Regime Prediction

Predicting **market regimes** of the S&P 500 a few days to a month ahead:

- **Bull / bear** — based on 20% drawdowns from a peak.
- **Calm / stressed** — based on 21-day realized volatility above its historical 80th percentile.

Our main goal is risk management: an early, well-calibrated warning that the market is entering a bear or high-stress period. We do not try to forecast exact prices.

The project follows a CRISP-DM–style workflow:

| Phase | Document | Code |
|---|---|---|
| 1. Business understanding | [docs/01_business_understanding.md](docs/01_business_understanding.md) | – |
| 2. Data understanding | [docs/02_data_understanding.md](docs/02_data_understanding.md) | `src/sp500/data_download.py` |
| 3. Data preparation | [docs/03_data_preparation.md](docs/03_data_preparation.md) | `src/sp500/data_prep.py`, `labels.py`, `features.py` |
| 4. Exploratory data analysis | [docs/04_eda.md](docs/04_eda.md) | `notebooks/01_eda.ipynb` |
| 5. Future steps (modeling & evaluation) | [docs/05_future_steps.md](docs/05_future_steps.md) | `src/sp500/validation.py` |
| Related work | [docs/literature.md](docs/literature.md) | – |

## Repository layout

```
sp500-regime-prediction/
├── data/
│   ├── raw/           # downloaded parquet files (not committed)
│   └── processed/     # cleaned, aligned data + model table (not committed)
├── docs/              # one document per project phase
├── notebooks/
│   └── 01_eda.ipynb   # exploratory data analysis
├── reports/
│   └── figures/       # saved plots
├── src/sp500/
│   ├── config.py         # paths, tickers, constants
│   ├── data_download.py  # step 1: download raw data (yfinance, Stooq fallback)
│   ├── data_prep.py      # step 2: clean, quality-check, align calendars
│   ├── labels.py         # bull/bear and calm/stressed labels, forward targets
│   ├── features.py       # step 3: features + labels -> model table
│   └── validation.py     # walk-forward splits with a gap
├── pyproject.toml
└── requirements.txt
```

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the pipeline:

```bash
python -m sp500.data_download   # -> data/raw/*.parquet
python -m sp500.data_prep       # -> data/processed/sp500_daily.parquet + reports/data_quality.csv
python -m sp500.features        # -> data/processed/model_table.parquet
```

Then open `notebooks/01_eda.ipynb`.

## Data

| Series | Ticker | From | Role |
|---|---|---|---|
| S&P 500 price index | `^GSPC` | 1927 (OHL from ~1962) | Main series: labels and features |
| S&P 500 total return | `^SP500TR` | 1988 | Bull/bear labels including dividends |
| SPY ETF | `SPY` | 1993 | Real traded volume |
| E-mini S&P 500 futures | `ES=F` | ~2000 | Futures moves (watch quarterly roll jumps) |

Raw data is not committed. Yahoo data can change or disappear, so every run caches to `data/raw/`. Details are in [docs/02_data_understanding.md](docs/02_data_understanding.md).

## Key rules (avoiding look-ahead bias)

1. Features use only data available at the close of day *t*.
2. Targets are labels shifted **forward** (`label.shift(-h)`); features are never shifted backward.
3. Volatility percentiles use only past data (an expanding window).
4. Validation is walk-forward in time order, with a gap of at least *h* days between train and test.
5. **2022 onward is the final holdout.** Do not touch it during tuning.

## Team

<!-- Add team members, roles and contact here -->

## License

MIT. See [LICENSE](LICENSE).
