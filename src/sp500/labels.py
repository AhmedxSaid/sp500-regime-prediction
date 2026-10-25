"""Regime labels (bull/bear, calm/stressed) and forward-shifted prediction targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BEAR_MERGE_MAX_GAP,
    BEAR_THRESHOLD,
    STRESS_MIN_HISTORY,
    STRESS_QUANTILE,
    STRESS_WINDOW,
    TRADING_DAYS,
)


def bear_market_labels(
    close: pd.Series, threshold: float = BEAR_THRESHOLD, method: str = "peak_to_trough"
) -> pd.Series:
    """Label each day 1 (bear) or 0 (bull).

    method="peak_to_trough": classic dating. A bear runs from the peak before a fall of at
        least `threshold` to the trough; it is confirmed over once price rises `threshold`
        above the trough. Uses hindsight, so it is a LABEL only, never a feature.
    method="recovery": a bear starts on the day price closes `threshold` below its peak and
        ends when price gets back to that peak. Causal (known at day t), so it can also be
        used as a current-state feature or baseline.
    """
    if method not in {"peak_to_trough", "recovery"}:
        raise ValueError("method must be 'peak_to_trough' or 'recovery'")

    s = close.dropna()
    v = s.to_numpy(dtype=float)
    n = len(v)
    bear = np.zeros(n, dtype=bool)
    in_bear = False
    peak_i = trough_i = start_i = 0

    for i in range(1, n):
        p = v[i]
        if not in_bear:
            if p > v[peak_i]:
                peak_i = i
            elif p <= v[peak_i] * (1 - threshold):
                in_bear = True
                trough_i = i
                start_i = peak_i if method == "peak_to_trough" else i
        else:
            if p < v[trough_i]:
                trough_i = i
            if method == "peak_to_trough":
                if p >= v[trough_i] * (1 + threshold):
                    bear[start_i : trough_i + 1] = True
                    in_bear = False
                    peak_i = i
            else:
                if p >= v[peak_i]:
                    bear[start_i:i] = True
                    in_bear = False
                    peak_i = i

    if in_bear:  # bear still ongoing at the end of the sample
        bear[start_i:] = True

    out = pd.Series(bear.astype("int8"), index=s.index, name=f"bear_{method}")
    return out.reindex(close.index)


def merge_bear_episodes(
    bear: pd.Series, close: pd.Series, max_gap: int = BEAR_MERGE_MAX_GAP
) -> pd.Series:
    """Merge bear markets separated by a short, failed rally (robustness label).

    Two consecutive bear episodes are merged (the bull days between them become bear) when
      - the bull gap between them is at most `max_gap` trading days, and
      - price never closes above the peak where the (merged) bear started.
    Uses hindsight, like `bear_market_labels(method="peak_to_trough")`: LABEL only.
    """
    s = bear.fillna(0).astype(int).to_numpy().copy()
    price = close.reindex(bear.index).ffill().to_numpy(dtype=float)
    n = len(s)

    # (start, end) positions of each run of 1s
    runs, i = [], 0
    while i < n:
        if s[i] == 1:
            j = i
            while j + 1 < n and s[j + 1] == 1:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1

    merged = []
    for start, end in runs:
        if merged:
            prev_start, prev_end = merged[-1]
            gap = start - prev_end - 1
            peak = price[prev_start]
            gap_max = np.nanmax(price[prev_end + 1 : start]) if gap > 0 else -np.inf
            if gap <= max_gap and gap_max <= peak:
                merged[-1][1] = end
                continue
        merged.append([start, end])

    out = np.zeros(n, dtype="int8")
    for start, end in merged:
        out[start : end + 1] = 1
    result = pd.Series(out, index=bear.index, name="bear_merged")
    return result.where(bear.notna())


def realized_vol(close: pd.Series, window: int = STRESS_WINDOW) -> pd.Series:
    """Annualized close-to-close realized volatility."""
    r = np.log(close).diff()
    return r.rolling(window).std() * np.sqrt(TRADING_DAYS)


def stress_labels(
    close: pd.Series,
    window: int = STRESS_WINDOW,
    quantile: float = STRESS_QUANTILE,
    min_history: int = STRESS_MIN_HISTORY,
) -> pd.DataFrame:
    """Stressed = 1 when realized vol is above the expanding `quantile` of PAST realized vol.

    Returns a DataFrame with the realized vol, the threshold and the label (NaN until enough history).
    """
    rv = realized_vol(close, window)
    threshold = rv.expanding(min_periods=min_history).quantile(quantile).shift(1)
    label = (rv > threshold).astype(float)
    label[rv.isna() | threshold.isna()] = np.nan
    return pd.DataFrame({"rv": rv, "stress_threshold": threshold, "stressed": label})


def make_target(label: pd.Series, horizon: int) -> pd.Series:
    """Regime `horizon` days ahead, aligned to day t. Only the label is shifted, never the features."""
    return label.shift(-horizon).rename(f"target_{label.name}_h{horizon}")


def episodes(label: pd.Series) -> pd.DataFrame:
    """Start, end and length (trading days) of each run of 1s in a 0/1 label."""
    s = label.fillna(0).astype(int)
    change = s.diff().fillna(s.iloc[0] if len(s) else 0)
    starts = s.index[change == 1]
    ends = s.index[(change == -1).shift(-1, fill_value=False)]
    if len(ends) < len(starts):  # ongoing at the end
        ends = ends.append(pd.DatetimeIndex([s.index[-1]]))
    rows = []
    for a, b in zip(starts, ends):
        rows.append({"start": a, "end": b, "days": int(s.loc[a:b].sum())})
    return pd.DataFrame(rows)
