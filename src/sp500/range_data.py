"""Features and targets for predicting the MIN and MAX price over the next h trading days.

Targets (known only after day t + h, so never used as features):
    up_h   = log( max(high[t+1 .. t+h]) / close[t] )   >= 0 most of the time
    down_h = log( min(low [t+1 .. t+h]) / close[t] )   <= 0 most of the time
so that  max_price = close[t] * exp(up_h)  and  min_price = close[t] * exp(down_h).

Models predict the target divided by a causal volatility scale
    sigma_h = parkinson_21d * sqrt(h / 252)
which makes 1987, 2008 and 2017 comparable; predictions are multiplied back afterwards.

Usage:
    python -m sp500.range_data        # writes data/processed/range_table.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candles import multi_timeframe_features
from .config import DATA_PROCESSED, FEATURES_MAIN, TRADING_DAYS
from .features import build_model_table
from .patterns import PATTERNS, detect_patterns, pattern_score

RANGE_HORIZONS = (5, 21)
RANGE_START = "1962-07-01"   # real OHL from 1962 (+ warm-up for 21-day windows)


def forward_extremes(df: pd.DataFrame, h: int, prefix: str = "gspc_") -> pd.DataFrame:
    hi, lo, c = df[f"{prefix}high"], df[f"{prefix}low"], df[f"{prefix}close"]
    # max over t+1..t+h  ==  rolling max over h days, shifted back by h
    fmax = hi.rolling(h, min_periods=h).max().shift(-h)
    fmin = lo.rolling(h, min_periods=h).min().shift(-h)
    fclose = c.shift(-h)
    return pd.DataFrame({
        f"up_{h}": np.log(fmax / c),
        f"down_{h}": np.log(fmin / c),
        f"ret_{h}": np.log(fclose / c),
        f"max_high_{h}": fmax,
        f"min_low_{h}": fmin,
    })


def range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Regime features from the main project + range / ATR + patterns + multi-timeframe."""
    base = build_model_table(df)[FEATURES_MAIN]
    o, h, l, c = (df[f"gspc_{k}"] for k in ("open", "high", "low", "close"))
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f = pd.DataFrame(index=df.index)
    f["atr_14_pct"] = tr.rolling(14).mean() / c
    f["range_1d"] = np.log(h / l)
    f["range_5d"] = np.log(h.rolling(5).max() / l.rolling(5).min())
    f["range_21d"] = np.log(h.rolling(21).max() / l.rolling(21).min())
    f["past_up_5d"] = np.log(h.rolling(5).max() / c)       # how far below the 1-week high
    f["past_down_5d"] = np.log(l.rolling(5).min() / c)     # how far above the 1-week low
    f["past_up_21d"] = np.log(h.rolling(21).max() / c)
    f["past_down_21d"] = np.log(l.rolling(21).min() / c)
    f["close_pos_1d"] = ((c - l) / (h - l)).where(h > l, 0.5)
    f["upper_shadow"] = (h - np.maximum(o, c)) / c
    f["lower_shadow"] = (np.minimum(o, c) - l) / c
    f["body"] = np.log(c / o)
    f["vol_ratio_5_63"] = np.log(base["rv_21d"].rolling(5).mean() / base["rv_63d"])
    f["dow"] = df.index.dayofweek.astype(float)
    f["month"] = df.index.month.astype(float)

    ohlc = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    pats = detect_patterns(ohlc)
    pats.columns = [f"pat_{k}" for k in pats.columns]
    # pattern seen in the last 3 candles (a single-day flag is very sparse)
    recent = pats.rolling(3, min_periods=1).max()
    recent.columns = [f"{k}_3d" for k in pats.columns]
    score = pattern_score(detect_patterns(ohlc)).rename("pattern_score_d")
    mtf = multi_timeframe_features(ohlc)
    return pd.concat([base, f, pats, recent, score, mtf], axis=1)


def build_range_table(df: pd.DataFrame, horizons=RANGE_HORIZONS) -> pd.DataFrame:
    feats = range_features(df)
    parts = [feats]
    for h in horizons:
        tg = forward_extremes(df, h)
        sigma = (feats["parkinson_21d"] * np.sqrt(h / TRADING_DAYS)).rename(f"sigma_{h}")
        parts += [tg, sigma]
    prices = df[[f"gspc_{k}" for k in ("open", "high", "low", "close")]]
    prices.columns = ["open", "high", "low", "close"]
    table = pd.concat([prices, *parts], axis=1)
    return table.loc[RANGE_START:]


def feature_columns(table: pd.DataFrame) -> list[str]:
    skip = {"open", "high", "low", "close"}
    prefixes = ("up_", "down_", "ret_", "max_high_", "min_low_", "sigma_")
    return [c for c in table.columns if c not in skip and not c.startswith(prefixes)]


def load_range_table() -> pd.DataFrame:
    path = DATA_PROCESSED / "range_table.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return build_range_table(pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet"))


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    table = build_range_table(df)
    out = DATA_PROCESSED / "range_table.parquet"
    table.to_parquet(out)
    feats = feature_columns(table)
    print(f"Saved {table.shape[0]} rows x {table.shape[1]} cols ({len(feats)} features) -> {out}")
    for h in RANGE_HORIZONS:
        print(f"h={h}: median up {table[f'up_{h}'].median():.4f}, median down {table[f'down_{h}'].median():.4f}")
    print("Patterns found (daily):", {k: int(table[f'pat_{k}'].sum()) for k in PATTERNS})


if __name__ == "__main__":
    main()
