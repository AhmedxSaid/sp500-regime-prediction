"""Regime labels (bull/bear, calm/stressed) and forward-shifted prediction targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
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
