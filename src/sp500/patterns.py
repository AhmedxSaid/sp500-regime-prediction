"""Candlestick pattern recognition (vectorized, causal).

Every pattern uses only candles up to and including day t, so the flags are known at the close
of day t and can be used as features. Works on any candle size (daily, weekly, monthly) as long
as the frame has open/high/low/close columns.

Bullish patterns are +1, bearish -1, neutral (doji, inside bar) +1 when present.

Usage:
    from sp500.patterns import detect_patterns
    flags = detect_patterns(ohlc)          # one column per pattern
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# name -> (direction, short description) ; direction is used for the dashboard colours
PATTERNS = {
    "doji":               ("neutral", "Body < 10% of the range: indecision"),
    "hammer":             ("bullish", "Small body on top, long lower shadow, after a fall"),
    "shooting_star":      ("bearish", "Small body at the bottom, long upper shadow, after a rise"),
    "bullish_engulfing":  ("bullish", "Up candle whose body engulfs the previous down body"),
    "bearish_engulfing":  ("bearish", "Down candle whose body engulfs the previous up body"),
    "morning_star":       ("bullish", "Down candle, small candle, strong up candle"),
    "evening_star":       ("bearish", "Up candle, small candle, strong down candle"),
    "three_white_soldiers": ("bullish", "Three up candles, each closing higher"),
    "three_black_crows":  ("bearish", "Three down candles, each closing lower"),
    "inside_bar":         ("neutral", "Range inside the previous range: volatility squeeze"),
    "outside_bar":        ("neutral", "Range engulfs the previous range: volatility expansion"),
    "bullish_marubozu":   ("bullish", "Big up body, almost no shadows"),
    "bearish_marubozu":   ("bearish", "Big down body, almost no shadows"),
}


def _cols(df: pd.DataFrame, prefix: str = "") -> tuple[pd.Series, ...]:
    return tuple(df[f"{prefix}{c}"] for c in ("open", "high", "low", "close"))


def detect_patterns(df: pd.DataFrame, prefix: str = "", trend_window: int = 5) -> pd.DataFrame:
    """Return a 0/1 frame with one column per pattern in `PATTERNS`.

    `prefix` lets you pass e.g. the processed table with `gspc_open`, ... (prefix="gspc_").
    `trend_window` defines the short trend used by hammer / shooting star (close vs. close
    `trend_window` candles ago).
    """
    o, h, l, c = _cols(df, prefix)
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    up = c > o
    down = c < o
    avg_body = body.rolling(20, min_periods=5).mean()
    trend = c / c.shift(trend_window) - 1

    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)
    body1, body2 = body.shift(1), body.shift(2)
    up1, down1, up2, down2 = up.shift(1, fill_value=False), down.shift(1, fill_value=False), \
        up.shift(2, fill_value=False), down.shift(2, fill_value=False)

    out = pd.DataFrame(index=df.index)
    out["doji"] = body <= 0.1 * rng
    small_body = body <= 0.35 * rng
    out["hammer"] = small_body & (lower >= 2 * body) & (upper <= 0.25 * rng) & (trend < 0)
    out["shooting_star"] = small_body & (upper >= 2 * body) & (lower <= 0.25 * rng) & (trend > 0)
    out["bullish_engulfing"] = up & down1 & (o <= c1) & (c >= o1) & (body > body1)
    out["bearish_engulfing"] = down & up1 & (o >= c1) & (c <= o1) & (body > body1)
    mid2 = (o2 + c2) / 2
    star = body1 <= 0.5 * avg_body
    out["morning_star"] = down2 & (body2 > avg_body) & star & up & (c > mid2)
    out["evening_star"] = up2 & (body2 > avg_body) & star & down & (c < mid2)
    out["three_white_soldiers"] = up & up1 & up2 & (c > c1) & (c1 > c2) & (body > 0.5 * avg_body)
    out["three_black_crows"] = down & down1 & down2 & (c < c1) & (c1 < c2) & (body > 0.5 * avg_body)
    out["inside_bar"] = (h < h1) & (l > l1)
    out["outside_bar"] = (h > h1) & (l < l1)
    maru = (body >= 0.9 * rng) & (body > avg_body)
    out["bullish_marubozu"] = maru & up
    out["bearish_marubozu"] = maru & down

    valid = o.notna() & h.notna() & l.notna() & c.notna()
    out = out.astype(float).where(valid)
    return out[list(PATTERNS)]


def pattern_score(flags: pd.DataFrame) -> pd.Series:
    """Net bullish (+) minus bearish (-) pattern count on each candle."""
    sign = {k: (1 if d == "bullish" else -1 if d == "bearish" else 0) for k, (d, _) in PATTERNS.items()}
    return sum(flags[k].fillna(0) * s for k, s in sign.items()).rename("pattern_score")


def pattern_stats(flags: pd.DataFrame, fwd_up: pd.Series, fwd_down: pd.Series,
                  fwd_ret: pd.Series) -> pd.DataFrame:
    """How the market behaved after each pattern vs. all days (for the dashboard).

    fwd_up / fwd_down: log(max high / close) and log(min low / close) over the next h days.
    fwd_ret: log return over the next h days.
    """
    base = pd.DataFrame({"up": fwd_up, "down": fwd_down, "ret": fwd_ret}).dropna()
    rows = [{
        "pattern": "all days", "direction": "", "count": len(base),
        "hit_rate_up": (base["ret"] > 0).mean(),
        "avg_ret": base["ret"].mean(), "avg_max_up": base["up"].mean(),
        "avg_max_down": base["down"].mean(),
    }]
    for name, (direction, _) in PATTERNS.items():
        sel = base[flags[name].reindex(base.index) == 1]
        if len(sel) == 0:
            continue
        rows.append({
            "pattern": name, "direction": direction, "count": len(sel),
            "hit_rate_up": (sel["ret"] > 0).mean(), "avg_ret": sel["ret"].mean(),
            "avg_max_up": sel["up"].mean(), "avg_max_down": sel["down"].mean(),
        })
    return pd.DataFrame(rows)
