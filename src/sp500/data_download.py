"""Download raw daily data for the core S&P 500 series and save it as CSV in data/raw/.

The repo already ships a snapshot of these files. Run this only to refresh them;
it overwrites data/raw/*.csv.

Usage:
    python -m sp500.data_download            # Yahoo via yfinance, Stooq as fallback
    python -m sp500.data_download --source stooq
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from .config import DATA_RAW, START_DATE, STOOQ_SYMBOLS, TICKERS


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names, a tz-naive DatetimeIndex named 'date', sorted."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (Price, Ticker) columns even for a single ticker
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def download_yahoo(symbol: str, start: str = START_DATE) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, start=start, auto_adjust=False, actions=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol}")
    return _standardize(df)


def download_stooq(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"Stooq returned no data for {symbol} (try opening {url} in a browser)")
    return _standardize(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["yahoo", "stooq"], default="yahoo")
    parser.add_argument("--start", default=START_DATE)
    args = parser.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    meta = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "series": {}}

    for name, yahoo_symbol in TICKERS.items():
        source = args.source
        try:
            if source == "yahoo":
                df = download_yahoo(yahoo_symbol, args.start)
            else:
                if name not in STOOQ_SYMBOLS:
                    print(f"[skip] {name}: not available on Stooq")
                    continue
                df = download_stooq(STOOQ_SYMBOLS[name])
        except Exception as exc:  # fall back to Stooq if Yahoo fails
            if source == "yahoo" and name in STOOQ_SYMBOLS:
                print(f"[warn] Yahoo failed for {yahoo_symbol} ({exc}); trying Stooq")
                source = "stooq"
                df = download_stooq(STOOQ_SYMBOLS[name])
            else:
                print(f"[error] {name}: {exc}")
                continue

        path = DATA_RAW / f"{name}.csv"
        price_cols = [c for c in ("open", "high", "low", "close", "adj_close") if c in df.columns]
        df[price_cols] = df[price_cols].round(4)  # Yahoo stores float32; drop the noise digits
        df.to_csv(path, date_format="%Y-%m-%d")
        meta["series"][name] = {
            "source": source,
            "symbol": yahoo_symbol if source == "yahoo" else STOOQ_SYMBOLS[name],
            "rows": len(df),
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "columns": list(df.columns),
        }
        print(f"[ok] {name:8s} {len(df):6d} rows  {df.index.min().date()} -> {df.index.max().date()}  ({source})")

    (DATA_RAW / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
