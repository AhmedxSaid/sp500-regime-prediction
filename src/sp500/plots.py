"""Small matplotlib helpers shared by the notebooks (static, so they render on GitHub)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UP_EDGE, DOWN_FILL = "#7a7a7a", "#3d3d3d"
C_MAX, C_MIN = "#257de0", "#f47230"          # max high = blue, min low = orange
PAT_COLORS = {"bullish": "#008a17", "bearish": "#ec4647", "neutral": "#666666"}
PAT_MARKERS = {"bullish": "^", "bearish": "v", "neutral": "D"}


def plot_candles(ax, ohlc: pd.DataFrame, width: float = 0.6, n_ticks: int = 8):
    """Draw candles at integer positions 0..n-1 (no weekend gaps). Returns the x positions."""
    x = np.arange(len(ohlc))
    o, h, l, c = (ohlc[k].to_numpy() for k in ("open", "high", "low", "close"))
    up = c >= o
    ax.vlines(x, l, h, color=np.where(up, UP_EDGE, DOWN_FILL), linewidth=0.8)
    ax.bar(x[up], (c - o)[up], bottom=o[up], width=width, color="white", edgecolor=UP_EDGE, linewidth=0.8)
    ax.bar(x[~up], (o - c)[~up], bottom=c[~up], width=width, color=DOWN_FILL, edgecolor=DOWN_FILL, linewidth=0.8)
    step = max(len(ohlc) // n_ticks, 1)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([d.strftime("%d %b %y") for d in ohlc.index[::step]], rotation=0, fontsize=8)
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return x


def mark_patterns(ax, ohlc: pd.DataFrame, flags: pd.DataFrame, patterns: dict):
    """Mark pattern candles: bullish below the low, bearish / neutral above the high."""
    pad = (ohlc["high"] - ohlc["low"]).median() * 0.8
    x = np.arange(len(ohlc))
    for direction in ("bullish", "bearish", "neutral"):
        names = [n for n, (d, _) in patterns.items() if d == direction]
        hit = flags[names].fillna(0).sum(axis=1).to_numpy() > 0
        if not hit.any():
            continue
        y = (ohlc["low"] - pad) if direction == "bullish" else (ohlc["high"] + pad)
        ax.scatter(x[hit], y.to_numpy()[hit], marker=PAT_MARKERS[direction], s=28,
                   color=PAT_COLORS[direction], label=f"{direction} pattern", zorder=3)


def band_vs_actual(ax, g: pd.DataFrame, color: str, label: str):
    """Plot a 10-90% quantile band, the median and the actual value (all in % moves)."""
    g = g.sort_values("date")
    ax.fill_between(g["date"], g["q10"] * 100, g["q90"] * 100, color=color, alpha=0.18, lw=0,
                    label=f"80% band ({label})")
    ax.plot(g["date"], g["q50"] * 100, color=color, lw=1.2, label=f"median forecast ({label})")
    ax.plot(g["date"], g["actual"] * 100, color="#191919", lw=0.7, alpha=0.8, label=f"actual ({label})")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_ylabel("% move vs. close")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
