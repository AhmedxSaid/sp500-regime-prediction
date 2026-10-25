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
    "vix": "^VIX",         # CBOE Volatility Index (30-day implied vol), from 1990
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
# Candidate feature groups, each tested on top of FEATURES_MAIN (see run_feature_groups.py)
FEATURE_GROUPS = {
    "spy_volume": ["spy_vol_ratio_20d"],                    # from 1993
    "vix": ["vix", "vix_rv_spread", "vix_chg_5d"],          # from 1990
}
# Kept only if it beats the base set out of sample (decision rule fixed before testing):
# mean PR-AUC over the 4 main targets improves by >= 0.01 for BOTH logistic regression and
# LightGBM, and mean onset PR-AUC drops by no more than 0.01 for either.
GROUP_MIN_GAIN = 0.01
GROUP_MAX_ONSET_LOSS = 0.01
# Result of run_feature_groups.py (2026-10-25, test years 2000-2021):
#   spy_volume: +0.005 / +0.002 PR-AUC (logreg / LightGBM) -> DROPPED (below the 0.01 bar)
#   vix:        +0.020 / +0.018 PR-AUC, onset +0.043 / +0.018 -> KEPT
# VIX only exists from 1990, so it forms a separate "modern" model rather than joining the 1962 one.
# On the same test years (2000-2021) the 1962 model WITHOUT VIX beats the 1990 model WITH VIX
# (e.g. stressed 21d PR-AUC 0.725 vs 0.674): the extra history is worth more than VIX.
# LightGBM from 1962 with VIX missing before 1990 (run_feature_groups.py --vix-hybrid):
# -0.018 mean PR-AUC vs. the main features -> DROPPED.
# -> FEATURES_MAIN (1962) stays the primary model; VIX is not used. FEATURES_MODERN is kept for reference.
FEATURES_MODERN = FEATURES_MAIN + FEATURE_GROUPS["vix"]
MODERN_START = "1990-02-01"
MODERN_FIRST_TEST_YEAR = 2000

MODEL_START = "1962-01-01"      # first rows with all main features
FIRST_TEST_YEAR = 1980          # walk-forward: train on 1962..N-1, test on year N
WALK_FORWARD_GAP = 21           # trading days between train end and test start (>= max horizon)
SVM_TRAIN_EVERY = 5             # SVM trains on every 5th day (weekly), as in Sheta et al.
