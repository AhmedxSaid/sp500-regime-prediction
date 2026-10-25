# 2. Data Understanding

## Sources

### Core S&P 500 series

| Data | Ticker / source | Range | Use |
|---|---|---|---|
| S&P 500 price index, daily OHLCV | `^GSPC` via yfinance or Stooq | 1927 onward (open/high/low from about 1962) | Main series. Labels: drawdown for bull/bear, realized volatility for calm/stressed. Features: returns, momentum, moving averages, drawdown |
| S&P 500 total return index | `^SP500TR` (Yahoo) | 1988 onward | Returns including dividends; more accurate for bull/bear |
| SPY ETF | `SPY` | 1993 onward | Tradable version, with real trading volume |
| E-mini S&P 500 futures | `ES=F` (Yahoo) | about 2000 onward | Overnight moves and gaps |

### Links

| Data | Yahoo Finance | Stooq (free CSV) | Official / other |
|---|---|---|---|
| `^GSPC` | [history](https://finance.yahoo.com/quote/%5EGSPC/history) | [stooq ^spx](https://stooq.com/q/d/?s=%5Espx) | [S&P DJI](https://www.spglobal.com/spdji/en/indices/equity/sp-500/) · [FRED SP500](https://fred.stlouisfed.org/series/SP500) (last 10 years only) |
| `^SP500TR` | [history](https://finance.yahoo.com/quote/%5ESP500TR/history) | – | S&P DJI page (total-return version) |
| `SPY` | [history](https://finance.yahoo.com/quote/SPY/history) | [stooq spy.us](https://stooq.com/q/d/?s=spy.us) | [State Street SPY](https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy) |
| `ES=F` | [history](https://finance.yahoo.com/quote/ES%3DF/history) | [stooq es.f](https://stooq.com/q/d/?s=es.f) | [CME E-mini S&P 500](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html) |

Tools: [yfinance](https://pypi.org/project/yfinance/) ([GitHub](https://github.com/ranaroussi/yfinance)) · [pandas-datareader](https://pandas-datareader.readthedocs.io/) (Stooq reader).

**The data is included in the repo:** [`data/raw/`](../data/raw/README.md) holds the complete daily history of all four series as CSV (Yahoo Finance, downloaded 2026-10-25, through 2026-10-23):

| File | Range | Rows |
|---|---|---|
| `gspc.csv` | 1927-12-30 → 2026-10-23 | 24,821 |
| `sp500tr.csv` | 1988-01-04 → 2026-10-23 | 9,776 |
| `spy.csv` | 1993-01-29 → 2026-10-23 | 8,492 |
| `es.csv` | 2000-09-18 → 2026-10-23 | 6,590 |

To refresh it with the latest prices:

```bash
python -m sp500.data_download
```

### Candidate extra sources (added later, one group at a time)

- **Volatility:** VIX (`^VIX`), VIX3M (`^VIX3M`), VIX9D, VVIX → term structure (VIX/VIX3M).
- **Sector ETFs:** XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB → dispersion and defensive vs. cyclical rotation.
- **Breadth:** share of constituents above their 200-day MA, advance/decline line.
- **Macro / rates:** 10Y–2Y spread, credit spreads (FRED).
- **News sentiment:** for the SVM + sentiment extension.

## Data quality notes

| Issue | Series | How we handle it |
|---|---|---|
| Open/high/low missing or equal to close before ~1962 | `^GSPC` | Rows with O=H=L=C get O/H/L set to NaN; range-based volatility only from 1962 |
| Zero or missing volume in early history | `^GSPC` | Zero volume → NaN; use SPY volume as the real volume feature |
| Total return index only from 1988 | `^SP500TR` | Bull/bear labels on `^SP500TR` from 1988, `^GSPC` before (or for the full-history version) |
| Quarterly contract roll jumps | `ES=F` | Roll windows flagged (~8 days before the 3rd Friday of Mar/Jun/Sep/Dec); ES returns set to NaN on those days |
| Different trading calendars (futures trade on more days) | `ES=F` vs. index | Everything aligned to the `^GSPC` trading calendar |
| Small gaps / holidays | all | Forward-fill prices for at most 2 days; volume is never filled |
| Yahoo history can change or break | all | A dated snapshot is committed in `data/raw/`; refreshes go in separate commits |
| `^SP500TR` volume is always 0 | `^SP500TR` | Ignored (use SPY volume) |
| Prices stored as 32-bit floats (e.g. `20.430000305175781`) | all | Rounded to 4 decimals |
| FRED SP500 covers only 10 years (licensing) | FRED | Use Yahoo or Stooq for full history |
| Yahoo "Download" button may require sign-in | Yahoo | Use yfinance (no sign-in) or Stooq CSVs |

`python -m sp500.data_prep` writes a quality report to `reports/data_quality.csv`. It counts non-positive prices, flat OHLC rows, high < low inconsistencies, zero volume, missing closes and daily moves above 15%.

**Note on large moves:** some extreme days are genuine. For example, 19 Oct 1987 was −20.5%, and March 2020 had several ±9% days. They are flagged for review, not removed.
