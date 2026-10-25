"""Project paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# short name -> Yahoo ticker
TICKERS = {
    "gspc": "^GSPC",       # S&P 500 price index
    "sp500tr": "^SP500TR", # S&P 500 total return index
    "spy": "SPY",          # SPDR S&P 500 ETF
    "es": "ES=F",          # E-mini S&P 500 continuous futures
}

# short name -> Stooq symbol (fallback source; no total return index there)
STOOQ_SYMBOLS = {
    "gspc": "^spx",
    "spy": "spy.us",
    "es": "es.f",
}

START_DATE = "1927-01-01"
TRADING_DAYS = 252

# Forward-fill prices for at most this many missing days after calendar alignment
FFILL_LIMIT = 2

# Regime label settings
BEAR_THRESHOLD = 0.20          # 20% drawdown
STRESS_WINDOW = 21             # days for realized volatility
STRESS_QUANTILE = 0.80         # percentile of past realized volatility
STRESS_MIN_HISTORY = 5 * TRADING_DAYS

# Prediction horizons (trading days ahead)
HORIZONS = (5, 21)

# Final holdout: never used for tuning
HOLDOUT_START = "2022-01-01"
