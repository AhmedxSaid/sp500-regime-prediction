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
# Robustness label: merge two bear markets if the rally between them lasts at most this many
# trading days (~6 months) and never gets back above the first bear's starting peak.
# Fixed on 2026-10-25 BEFORE any model results; do not tune it.
BEAR_MERGE_MAX_GAP = 126
STRESS_WINDOW = 21             # days for realized volatility
STRESS_QUANTILE = 0.80         # percentile of past realized volatility
# Main stress label: compare with the PAST 5 YEARS only, so "stressed" means "unusually volatile
# for the recent market". (An all-history threshold was anchored on the 1930s: 1-2% stressed in
# the 1950s-60s vs 37% in the 2000s.) Fixed on 2026-10-25 BEFORE any model results; do not tune.
STRESS_LOOKBACK = 5 * TRADING_DAYS
STRESS_MIN_HISTORY = 5 * TRADING_DAYS
# Robustness stress label: absolute level, same meaning in every era
STRESS_FIXED_VOL = 0.25

# Prediction horizons (trading days ahead)
HORIZONS = (5, 21)

# Final holdout: never used for tuning
HOLDOUT_START = "2022-01-01"

# ---------------------------------------------------------------------------
# Modeling setup. Fixed on 2026-10-25 BEFORE any model results; do not tune.
# ---------------------------------------------------------------------------
# Main feature set: everything available from 1962 (real OHL starts then), without the
# near-duplicates found in EDA (Garman-Klass ~ Parkinson 0.97, mom_6m ~ dist_ma200 0.94,
# es_ret_1d ~ ret_1d 0.97).
FEATURES_MAIN = [
    "ret_1d", "mom_1m", "mom_3m", "mom_12m",          # returns / momentum
    "dist_ma50", "dist_ma200",                          # trend
    "drawdown", "below_peak_regime",                    # drawdown
    "rv_21d", "rv_63d", "parkinson_21d", "rv_vs_5y_p80",  # volatility
    "gap_open",                                         # overnight gap
]
# Extended set adds SPY volume (from 1993); tested later as a feature group
FEATURES_EXTENDED = FEATURES_MAIN + ["spy_vol_ratio_20d"]

MODEL_START = "1962-01-01"      # first rows with all main features
FIRST_TEST_YEAR = 1980          # walk-forward: train on 1962..N-1, test on year N
WALK_FORWARD_GAP = 21           # trading days between train end and test start (>= max horizon)
SVM_TRAIN_EVERY = 5             # SVM trains on every 5th day (weekly), as in Sheta et al.
