"""Time-ordered validation splits."""
from __future__ import annotations

import re
from typing import Iterator

import numpy as np
import pandas as pd

from .config import HOLDOUT_START


def walk_forward_splits(
    index: pd.DatetimeIndex,
    first_test_year: int,
    gap: int,
    last_test_year: int | None = None,
    holdout_start: str = HOLDOUT_START,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (test_year, train_idx, test_idx) as integer positions.

    Train = everything before the test year minus a `gap` of trading days (use gap >= horizon h,
    so overlapping targets cannot leak). Test = one calendar year. Years at or after
    `holdout_start` are never yielded.
    """
    index = pd.DatetimeIndex(index)
    holdout_year = pd.Timestamp(holdout_start).year
    last = min(last_test_year or index.max().year, holdout_year - 1)
    years = index.year
    for year in range(first_test_year, last + 1):
        test_idx = np.flatnonzero(years == year)
        if len(test_idx) == 0:
            continue
        train_end = test_idx[0] - gap
        if train_end <= 0:
            continue
        yield year, np.arange(train_end), test_idx


def truncate_at(table: pd.DataFrame, end: str) -> pd.DataFrame:
    """The model table as it was when data ended on `end`.

    Drops later rows and blanks forward targets (`target_*_h{h}`) whose answer lies after `end`,
    as `process_data.py` would have produced on data ending that day.
    """
    out = table.loc[:end].copy()
    for col in out.columns:
        m = re.fullmatch(r"target_.+_h(\d+)", col)
        if m:
            h = int(m.group(1))
            out.iloc[len(out) - h:, out.columns.get_loc(col)] = np.nan
    return out


def holdout_mask(index: pd.DatetimeIndex, holdout_start: str = HOLDOUT_START) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(index) >= pd.Timestamp(holdout_start))
