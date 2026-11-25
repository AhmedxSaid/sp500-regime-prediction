"""Move candles between timeframes: daily -> weekly / monthly and back again.

Two directions:

* `resample_candles` aggregates daily candles into bigger ones (open = first, high = max,
  low = min, close = last). A weekly candle is labelled with its LAST trading day, so it is only
  "known" at that date.
* `upsample_to_daily` maps a higher-timeframe series back onto the daily calendar so that each
  day sees only higher-timeframe candles that were already closed (causal forward-fill).
  `partial_candle` builds the still-open candle (week-to-date, month-to-date) as of each day.

`multi_timeframe_features` combines both into daily features for the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .patterns import detect_patterns, pattern_score

TIMEFRAMES = {"D": "Daily", "W": "Weekly", "M": "Monthly"}
_RULES = {"W": "W-FRI", "M": "ME"}


def _period_key(index: pd.DatetimeIndex, tf: str) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    if tf == "W":
        return pd.Series(idx.to_period("W-FRI").astype(str), index=idx)
    if tf == "M":
        return pd.Series(idx.to_period("M").astype(str), index=idx)
    raise ValueError("tf must be 'W' or 'M'")


def resample_candles(ohlc: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate daily OHLC(V) into weekly ('W') or monthly ('M') candles.

    The index of each candle is the last trading day inside it (not the calendar period end).
    """
    if tf == "D":
        return ohlc.copy()
    key = _period_key(ohlc.index, tf)
    g = ohlc.groupby(key.values)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in ohlc.columns:
        agg["volume"] = "sum"
    out = g.agg({k: v for k, v in agg.items() if k in ohlc.columns})
    out.index = pd.DatetimeIndex(g.apply(lambda x: x.index.max()).values)
    out.index.name = "date"
    return out.sort_index()


def partial_candle(ohlc: pd.DataFrame, tf: str) -> pd.DataFrame:
    """The still-forming higher-timeframe candle as of each day (e.g. week-to-date)."""
    key = _period_key(ohlc.index, tf).values
    g = ohlc.groupby(key)
    return pd.DataFrame({
        "open": g["open"].transform("first"),
        "high": g["high"].cummax(),
        "low": g["low"].cummin(),
        "close": ohlc["close"],
    }, index=ohlc.index)


def upsample_to_daily(htf: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill closed higher-timeframe rows onto the daily calendar.

    A weekly candle labelled Friday becomes visible from that Friday's close onward (same day
    is fine: features are known at the close of day t).
    """
    return htf.reindex(htf.index.union(daily_index)).ffill().reindex(daily_index)


def multi_timeframe_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Daily features built from weekly and monthly candles (all causal)."""
    feats = {}
    c = ohlc["close"]
    for tf in ("W", "M"):
        t = tf.lower()
        closed = resample_candles(ohlc[["open", "high", "low", "close"]], tf)
        # last CLOSED candle: shift by one candle so the still-open one is never used here
        last = closed.copy()
        last_rng = np.log(last["high"] / last["low"])
        last_body = np.log(last["close"] / last["open"])
        avg_rng = last_rng.rolling(8, min_periods=4).mean()
        pats = detect_patterns(closed)
        htf = pd.DataFrame({
            f"{t}_prev_range": last_rng,
            f"{t}_prev_body": last_body,
            f"{t}_range_avg8": avg_rng,
            f"{t}_pattern_score": pattern_score(pats),
            f"{t}_prev_high": last["high"],
            f"{t}_prev_low": last["low"],
        })
        # keep values only from the candle's last day on (closed candles); the current
        # period's own candle closes only on its last day, so shift the daily map by one day
        up = upsample_to_daily(htf, ohlc.index).shift(1)
        feats[f"{t}_prev_range"] = up[f"{t}_prev_range"]
        feats[f"{t}_prev_body"] = up[f"{t}_prev_body"]
        feats[f"{t}_range_avg8"] = up[f"{t}_range_avg8"]
        feats[f"{t}_pattern_score"] = up[f"{t}_pattern_score"]
        # where is today's close relative to the previous candle's high / low
        feats[f"{t}_dist_prev_high"] = np.log(c / up[f"{t}_prev_high"])
        feats[f"{t}_dist_prev_low"] = np.log(c / up[f"{t}_prev_low"])
        # the candle that is still forming (week-to-date / month-to-date)
        part = partial_candle(ohlc[["open", "high", "low", "close"]], tf)
        feats[f"{t}_td_range"] = np.log(part["high"] / part["low"])
        feats[f"{t}_td_ret"] = np.log(c / part["open"])
        feats[f"{t}_td_pos"] = ((c - part["low"]) / (part["high"] - part["low"])).fillna(0.5)
    return pd.DataFrame(feats, index=ohlc.index)
