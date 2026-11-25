# Raw data snapshot

Complete daily history of the five core series, downloaded from Yahoo Finance on **2026-11-25**. The last trading day included is **2026-11-24**.

| File | Ticker | Description | First date | Last date | Rows | Size |
|---|---|---|---|---|---|---|
| `gspc.csv` | `^GSPC` | S&P 500 price index | 1927-12-30 | 2026-11-24 | 24,843 | 1.1 MB |
| `sp500tr.csv` | `^SP500TR` | S&P 500 total return index (dividends reinvested) | 1988-01-04 | 2026-11-24 | 9,798 | 0.4 MB |
| `spy.csv` | `SPY` | SPDR S&P 500 ETF Trust | 1993-01-29 | 2026-11-24 | 8,514 | 0.5 MB |
| `es.csv` | `ES=F` | E-mini S&P 500 futures, continuous front month | 2000-09-18 | 2026-11-24 | 6,612 | 0.4 MB |
| `vix.csv` | `^VIX` | CBOE Volatility Index (30-day implied volatility, in %) | 1990-01-02 | 2026-11-24 | 9,295 | 0.4 MB |

## Columns

`date, open, high, low, close, adj_close, volume`

- `date`: trading day (exchange local date), `YYYY-MM-DD`.
- Prices are rounded to 4 decimals. Yahoo stores them as 32-bit floats, so the extra digits were noise.
- `adj_close`: adjusted for splits and dividends. For SPY it differs from `close`; for the indexes it equals `close`.
- `volume`: shares (SPY), contracts (ES), or Yahoo's reported index volume (`^GSPC`).

## Known issues (handled in `src/sp500/data_prep.py`)

- **Close-only rows:** Yahoo returns open = high = low = close where only a close exists: `^GSPC` before 1962-01-02 (plus a few later days, last 1983-06-30), `^VIX` 1990-01-02 to 1991-12-31 (plus 2 days in 1997) and `^SP500TR` before 2009-03-30. The download script empties open/high/low on those rows (8,547 / 508 / 5,379 rows), so a missing O/H/L means "no intraday data". `^GSPC` volume is also 0 before 1950.
- **`^SP500TR`:** volume is always 0.
- **`ES=F`:** a continuous contract that jumps at the quarterly roll (around 8 days before the 3rd Friday of Mar/Jun/Sep/Dec). 11 rows have zero volume.
- **`^GSPC` suspected bad prints:** 1935-04-16, 1935-08-16, 1936-09-16 and 1961-04-17 are one-day spikes that reverse the next day. They're left as-is in this raw file; cleaning replaces and flags them.
- **`^VIX`:** volume is always 0. Values before 2003 are back-calculated by CBOE with the current method. Checks: record closes 80.86 (2008-11-20) and 82.69 (2020-03-16), record low 9.14 (2017-11-03).
- No duplicate dates in any file.

The full quality analysis is in [`notebooks/01_data_understanding.ipynb`](../../notebooks/01_data_understanding.ipynb).

Spot checks against known history: `^GSPC` closed at 224.84 on 1987-10-19 (Black Monday) and at 2,237.40 on 2020-03-23 (the COVID low).

## Refreshing

```bash
python -m sp500.data_download
```

This overwrites these files with a fresh download. Today's row is dropped until 17:15 New York time, so an unfinished session never gets in. Yahoo can revise history, so commit refreshed data as a separate, clearly labeled commit.

## Terms of use

The data comes from Yahoo Finance. S&P 500 index values are the property of S&P Dow Jones Indices. This snapshot is included for **non-commercial academic research and reproducibility only**. Check the source terms before any other use.
