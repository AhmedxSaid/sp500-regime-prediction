"""Clean raw series, run quality checks, and align everything on one trading calendar.

Usage:
    python -m sp500.data_prep
Outputs:
    data/processed/sp500_daily.parquet
    reports/data_quality.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, FFILL_LIMIT, REPORTS, TICKERS

PRICE_COLS = ["open", "high", "low", "close", "adj_close"]
KEEP_COLS = PRICE_COLS + ["volume"]


def raw_path(name: str):
    return DATA_RAW / f"{name}.csv"


def load_raw(name: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path(name), parse_dates=["date"], index_col="date")
    df.columns = [c.lower() for c in df.columns]
    return df


def clean_ohlcv(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    """Basic cleaning plus a dict of quality-check counts."""
    df = df[~df.index.duplicated(keep="last")].sort_index().copy()
    report: dict = {
        "series": name,
        "rows": len(df),
        "start": df.index.min().date(),
        "end": df.index.max().date(),
    }

    price_cols = [c for c in PRICE_COLS if c in df.columns]
    non_pos = df[price_cols] <= 0
    report["non_positive_prices"] = int(non_pos.to_numpy().sum())
    df[price_cols] = df[price_cols].mask(non_pos)

    if {"open", "high", "low", "close"}.issubset(df.columns):
        # Before ~1962 Yahoo stores ^GSPC open/high/low equal to the close
        flat = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
        report["flat_ohlc_rows"] = int(flat.sum())
        df.loc[flat, ["open", "high", "low"]] = np.nan

        bad_range = (df["high"] < df["low"]) | (df["close"] > df["high"] * 1.0001) | (
            df["close"] < df["low"] * 0.9999
        )
        report["inconsistent_hlc_rows"] = int(bad_range.sum())

    if "volume" in df.columns:
        zero_vol = df["volume"] <= 0
        report["zero_volume_rows"] = int(zero_vol.sum())
        df["volume"] = df["volume"].mask(zero_vol)

    log_ret = np.log(df["close"]).diff()
    report["missing_close"] = int(df["close"].isna().sum())
    report["moves_gt_15pct"] = int((log_ret.abs() > 0.15).sum())
    report["largest_up_move"] = float(log_ret.max())
    report["largest_down_move"] = float(log_ret.min())
    report["largest_down_date"] = log_ret.idxmin().date() if log_ret.notna().any() else None

    return df[[c for c in KEEP_COLS if c in df.columns]], report


def es_roll_window(index: pd.DatetimeIndex, days_before_expiry: int = 8, half_width: int = 2) -> pd.Series:
    """Flag trading days around the quarterly E-mini roll.

    ES contracts expire on the 3rd Friday of Mar/Jun/Sep/Dec, and volume moves to the
    next contract about 8 calendar days earlier. The continuous ES=F series can jump
    then, so returns in a +/- `half_width` trading-day window are unreliable.
    """
    flag = pd.Series(0, index=index, name="es_roll_window", dtype="int8")
    if len(index) == 0:
        return flag
    for year in range(index.min().year, index.max().year + 1):
        for month in (3, 6, 9, 12):
            days = pd.date_range(f"{year}-{month:02d}-01", periods=21, freq="D")
            third_friday = days[days.weekday == 4][2]
            roll = third_friday - pd.Timedelta(days=days_before_expiry)
            pos = index.searchsorted(roll)
            lo, hi = max(pos - half_width, 0), min(pos + half_width + 1, len(index))
            if lo < len(index):
                flag.iloc[lo:hi] = 1
    return flag


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    reports = []
    for name in TICKERS:
        path = raw_path(name)
        if not path.exists():
            print(f"[skip] {path.name} not found (run python -m sp500.data_download)")
            continue
        df, rep = clean_ohlcv(load_raw(name), name)
        frames[name] = df
        reports.append(rep)

    if "gspc" not in frames:
        raise FileNotFoundError("data/raw/gspc.csv is required as the master calendar")

    calendar = frames["gspc"].index
    aligned = []
    for name, df in frames.items():
        a = df.reindex(calendar)
        price_cols = [c for c in PRICE_COLS if c in a.columns]
        # only fill short gaps, and only inside the series' own history
        first, last = df.index.min(), df.index.max()
        inside = (a.index >= first) & (a.index <= last)
        a.loc[inside, price_cols] = a.loc[inside, price_cols].ffill(limit=FFILL_LIMIT)
        a.columns = [f"{name}_{c}" for c in a.columns]
        aligned.append(a)

    data = pd.concat(aligned, axis=1)
    if "es" in frames:
        data["es_roll_window"] = es_roll_window(calendar)
    data.index.name = "date"
    return data, pd.DataFrame(reports)


def main() -> None:
    data, quality = build_dataset()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    data.to_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    quality.to_csv(REPORTS / "data_quality.csv", index=False)
    print(quality.to_string(index=False))
    print(f"\nSaved {data.shape[0]} rows x {data.shape[1]} cols -> {DATA_PROCESSED / 'sp500_daily.parquet'}")


if __name__ == "__main__":
    main()
