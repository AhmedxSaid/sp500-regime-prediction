"""Build features (known at the close of day t), labels and forward targets.

Usage:
    python -m sp500.features
Output:
    data/processed/model_table.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, HORIZONS, TRADING_DAYS
from .labels import (
    bear_market_labels,
    make_target,
    merge_bear_episodes,
    stress_fixed_labels,
    stress_labels,
)

# Targets are built for these labels (main + robustness version of each)
TARGET_LABELS = ("bear", "bear_merged", "stressed", "stressed_fixed")

MOMENTUM_WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["gspc_close"]
    log_c = np.log(c)
    r = log_c.diff()

    f = pd.DataFrame(index=df.index)
    f["ret_1d"] = r
    for name, w in MOMENTUM_WINDOWS.items():
        f[f"mom_{name}"] = log_c.diff(w)
    for w in (50, 200):
        f[f"dist_ma{w}"] = c / c.rolling(w).mean() - 1
    f["drawdown"] = c / c.cummax() - 1
    f["rv_21d"] = r.rolling(21).std() * np.sqrt(TRADING_DAYS)
    f["rv_63d"] = r.rolling(63).std() * np.sqrt(TRADING_DAYS)

    if {"gspc_open", "gspc_high", "gspc_low"}.issubset(df.columns):
        o, h, l = df["gspc_open"], df["gspc_high"], df["gspc_low"]
        hl = np.log(h / l)
        co = np.log(c / o)
        # range-based volatility estimators (need real OHLC, i.e. from ~1962)
        f["parkinson_21d"] = np.sqrt((hl**2).rolling(21).mean() / (4 * np.log(2)) * TRADING_DAYS)
        gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
        f["garman_klass_21d"] = np.sqrt(gk.rolling(21).mean().clip(lower=0) * TRADING_DAYS)
        f["gap_open"] = np.log(o / c.shift(1))

    if "spy_volume" in df.columns:
        v = df["spy_volume"]
        f["spy_vol_ratio_20d"] = np.log(v / v.rolling(20).mean())

    if "es_close" in df.columns:
        es_ret = np.log(df["es_close"]).diff()
        if "es_roll_window" in df.columns:
            es_ret = es_ret.mask(df["es_roll_window"] == 1)
        f["es_ret_1d"] = es_ret

    return f


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Bull/bear on the total return index where available (from 1988), price index before
    tr = df.get("sp500tr_close")
    bear_price = df["gspc_close"]
    if tr is not None and tr.notna().sum() > TRADING_DAYS:
        # splice: scale the price index to the TR level at the TR start so returns stay continuous
        first = tr.first_valid_index()
        scale = tr.loc[first] / df["gspc_close"].loc[first]
        bear_price = pd.concat([df["gspc_close"].loc[:first].iloc[:-1] * scale, tr.loc[first:]])
        bear_price = bear_price.reindex(df.index).ffill()

    labels = pd.DataFrame(index=df.index)
    labels["bear"] = bear_market_labels(bear_price, method="peak_to_trough")        # main label
    labels["bear_merged"] = merge_bear_episodes(labels["bear"], bear_price)         # robustness label
    labels["bear_recovery"] = bear_market_labels(bear_price, method="recovery")     # causal state
    st = stress_labels(df["gspc_close"])                          # main: vs. past 5 years
    labels["stressed"] = st["stressed"]
    labels["stress_threshold"] = st["stress_threshold"]
    labels["stressed_fixed"] = stress_fixed_labels(df["gspc_close"])  # robustness: vol > 25%
    return labels


def build_model_table(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    feats = build_features(df)
    labels = build_labels(df)
    # Causal "still in a 20% drawdown and not yet back to the old high" flag, known at day t.
    # Lets models learn that a rally deep below the peak is suspicious, without changing the label.
    feats["below_peak_regime"] = labels["bear_recovery"].astype(float)
    targets = [make_target(labels[name], h) for h in horizons for name in TARGET_LABELS]
    return pd.concat([feats, labels, *targets], axis=1)


def main() -> None:
    df = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    table = build_model_table(df)
    out = DATA_PROCESSED / "model_table.parquet"
    table.to_parquet(out)
    print(f"Saved {table.shape[0]} rows x {table.shape[1]} cols -> {out}")
    print("\nShare of days in each regime:")
    print(table[["bear", "bear_merged", "bear_recovery", "stressed", "stressed_fixed"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
