# Raw data snapshot

Complete daily history of the four core series, downloaded from Yahoo Finance on **2026-10-25**. The last trading day included is **2026-10-23**.

| File | Ticker | Description | First date | Last date | Rows | Size |
|---|---|---|---|---|---|---|
| `gspc.csv` | `^GSPC` | S&P 500 price index | 1927-12-30 | 2026-10-23 | 24,821 | 1.3 MB |
| `sp500tr.csv` | `^SP500TR` | S&P 500 total return index (dividends reinvested) | 1988-01-04 | 2026-10-23 | 9,776 | 0.5 MB |
| `spy.csv` | `SPY` | SPDR S&P 500 ETF Trust | 1993-01-29 | 2026-10-23 | 8,492 | 0.5 MB |
| `es.csv` | `ES=F` | E-mini S&P 500 futures, continuous front month | 2000-09-18 | 2026-10-23 | 6,590 | 0.3 MB |

## Columns

`date, open, high, low, close, adj_close, volume`

- `date`: trading day (exchange local date), `YYYY-MM-DD`.
- Prices are rounded to 4 decimals. Yahoo stores them as 32-bit floats, so the extra digits were noise.
- `adj_close`: adjusted for splits and dividends. For SPY it differs from `close`; for the indexes it equals `close`.
- `volume`: shares (SPY), contracts (ES), or Yahoo's reported index volume (`^GSPC`).

## Known issues (handled in `src/sp500/data_prep.py`)

- **`^GSPC` before 1962-01-02:** open = high = low = close (only the close is real), and volume is 0.
- **`^SP500TR`:** volume is always 0.
- **`ES=F`:** a continuous contract that jumps at the quarterly roll (around 8 days before the 3rd Friday of Mar/Jun/Sep/Dec). 11 rows have zero volume.
- **`^GSPC` suspected bad prints:** 1935-04-16, 1935-08-16, 1936-09-16 and 1961-04-17 are one-day spikes that reverse the next day. They're left as-is in this raw file; cleaning replaces and flags them.
- No duplicate dates in any file.

The full quality analysis is in [`notebooks/00_data_understanding.ipynb`](../../notebooks/00_data_understanding.ipynb).

Spot checks against known history: `^GSPC` closed at 224.84 on 1987-10-19 (Black Monday) and at 2,237.40 on 2020-03-23 (the COVID low).

## Refreshing

```bash
python -m sp500.data_download
```

This overwrites these files with a fresh download. Yahoo can revise history, so commit refreshed data as a separate, clearly labeled commit.

## Terms of use

The data comes from Yahoo Finance. S&P 500 index values are the property of S&P Dow Jones Indices. This snapshot is included for **non-commercial academic research and reproducibility only**. Check the source terms before any other use.
