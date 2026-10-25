"""Process the raw S&P 500 data into a model-ready table, in one command.

Steps:
    1. Load the raw CSVs in data/raw/ and clean them (bad prices, pre-1962 OHL, zero volume).
    2. Align all series on the ^GSPC trading calendar and flag ES futures roll days.
    3. Build regime labels (bull/bear, calm/stressed) and forward targets (h days ahead).
    4. Build features known at the close of day t.
    5. Run checks (no look-ahead, known bear markets captured, value ranges).
    6. Save the outputs and a summary report.

Usage (from the repo root):
    python process_data.py
    python process_data.py --horizons 5 10 21 --format both

Outputs:
    data/processed/sp500_daily.{csv,parquet}   cleaned + aligned prices
    data/processed/model_table.{csv,parquet}   features + labels + targets
    reports/data_quality.csv                   quality counts per raw series
    reports/processing_summary.md              what was built, regime stats, check results
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # works without `pip install -e .`

from sp500.config import DATA_PROCESSED, HOLDOUT_START, HORIZONS, REPORTS  # noqa: E402
from sp500.data_prep import build_dataset  # noqa: E402
from sp500.features import TARGET_LABELS, build_model_table  # noqa: E402

BEAR_LABELS = ("bear", "bear_merged")          # hindsight labels (main + robustness)
STRESS_LABELS = ("stressed", "stressed_fixed")  # causal labels (main + robustness)
ALL_LABELS = ("bear", "bear_merged", "bear_recovery", "stressed", "stressed_fixed")

# Known stress periods (should be mostly stressed) and a very calm year (should not be)
KNOWN_STRESS = {
    "Black Monday 1987": ("1987-10-15", "1987-11-30"),
    "GFC Oct-Nov 2008": ("2008-10-01", "2008-11-30"),
    "COVID Mar 2020": ("2020-03-01", "2020-04-15"),
}
KNOWN_MEDIUM_STRESS = {
    "1962 flash crash": ("1962-05-01", "1962-07-31"),
    "1966 sell-off": ("1966-08-01", "1966-10-31"),
    "1970 bear": ("1970-04-15", "1970-06-30"),
}
CALM_YEAR = ("2017-01-01", "2017-12-31")
from sp500.labels import episodes  # noqa: E402

KNOWN_BEAR_MARKETS = {
    "Great Depression 1929-32": ("1929-09-16", "1932-06-01"),
    "1973-74": ("1973-01-11", "1974-10-03"),
    "Dot-com 2000-02": ("2000-03-24", "2002-10-09"),
    "GFC 2007-09": ("2007-10-09", "2009-03-09"),
    "COVID 2020": ("2020-02-19", "2020-03-23"),
    "Inflation 2022": ("2022-01-03", "2022-10-12"),
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_checks(table: pd.DataFrame, daily: pd.DataFrame, horizons) -> list[tuple[str, bool, str]]:
    """Return (check name, passed, detail) for each check."""
    checks = []

    # 1. Targets are exactly the label shifted forward by h (no look-ahead in features)
    for h in horizons:
        for label in TARGET_LABELS:
            col = f"target_{label}_h{h}"
            expected = table[label].shift(-h)
            ok = table[col].equals(expected) or (
                (table[col].isna() == expected.isna()).all()
                and np.allclose(table[col].dropna(), expected.dropna())
            )
            checks.append((f"{col} == {label}.shift(-{h})", bool(ok), ""))
            tail_nan = table[col].iloc[-h:].isna().all()
            checks.append((f"{col} is NaN in the last {h} rows", bool(tail_nan), ""))

    # 2. No look-ahead: features and the stress label must be identical when all data after a
    #    cut-off date is removed (a feature that peeks at the future would change)
    cut = pd.Timestamp("2008-06-30")
    truncated = build_model_table(daily.loc[:cut], horizons=horizons)
    point_in_time = [c for c in table.columns
                     if not c.startswith("target_") and c not in BEAR_LABELS]  # these use hindsight by design
    a = table.loc[:cut, point_in_time]
    b = truncated[point_in_time]
    diff_cols = [c for c in point_in_time
                 if not np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True)]
    checks.append((f"features + stress label unchanged when data after {cut.date()} is removed",
                   not diff_cols, ", ".join(diff_cols)))

    # 3. Known bear markets are captured (the bear label covers most of each window).
    #    Reported as WARN, not FAIL: the 20% rule ends a bear after any 20% rally, so long bears
    #    with big bear-market rallies (1929-32, 2000-02) get split. That is a labeling choice.
    for label in BEAR_LABELS:
        for name, (a, b) in KNOWN_BEAR_MARKETS.items():
            window = table.loc[a:b, label].dropna()
            share = window.mean() if len(window) else 0.0
            checks.append((f"`{label}` covers {name}", "PASS" if share >= 0.8 else "WARN",
                           f"{share:.0%} of days labeled bear"))

    # 3a. Stress labels: big crises fully caught (FAIL if not), calm 2017 not stressed,
    #     medium sell-offs reported as PASS/WARN (the fixed-25% label is expected to miss them)
    for label in STRESS_LABELS:
        for name, (a, b) in KNOWN_STRESS.items():
            share = table.loc[a:b, label].mean()
            checks.append((f"`{label}` catches {name}", bool(share >= 0.8), f"{share:.0%} of days stressed"))
        for name, (a, b) in KNOWN_MEDIUM_STRESS.items():
            share = table.loc[a:b, label].mean()
            checks.append((f"`{label}` catches {name}", "PASS" if share >= 0.5 else "WARN",
                           f"{share:.0%} of days stressed"))
        share = table.loc[CALM_YEAR[0]:CALM_YEAR[1], label].mean()
        checks.append((f"`{label}` is calm in 2017", bool(share <= 0.05), f"{share:.0%} of days stressed"))

    # 3b. Merging only ever adds bear days, never removes them
    checks.append(("`bear_merged` >= `bear` on every day",
                   bool((table["bear_merged"].fillna(0) >= table["bear"].fillna(0)).all()), ""))

    # 4. Value ranges
    checks.append(("drawdown is between -1 and 0", bool(table["drawdown"].dropna().between(-1, 0).all()), ""))
    checks.append(("realized vol is non-negative", bool((table["rv_21d"].dropna() >= 0).all()), ""))
    for col in ALL_LABELS:
        vals = {int(v) for v in table[col].dropna().unique()}
        checks.append((f"{col} only takes values 0/1", vals <= {0, 1}, str(sorted(vals))))
    checks.append(("dates are unique and sorted", table.index.is_unique and table.index.is_monotonic_increasing, ""))

    # normalize to PASS / WARN / FAIL
    return [(n, s if isinstance(s, str) else ("PASS" if s else "FAIL"), d) for n, s, d in checks]


def feature_availability(table: pd.DataFrame, feature_cols) -> pd.DataFrame:
    return pd.DataFrame({
        "first_valid": [table[c].first_valid_index() for c in feature_cols],
        "non_missing": [int(table[c].notna().sum()) for c in feature_cols],
    }, index=feature_cols)


def write_summary(path: Path, daily, table, quality, checks, horizons, feature_cols) -> None:
    def fmt_date(d):
        return "" if pd.isna(d) else pd.Timestamp(d).strftime("%Y-%m-%d")

    lines = [
        "# Processing summary",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `process_data.py`.",
        "",
        "## Outputs",
        "",
        "| Table | Rows | Columns | From | To |",
        "|---|---|---|---|---|",
        f"| `sp500_daily` | {len(daily):,} | {daily.shape[1]} | {fmt_date(daily.index.min())} | {fmt_date(daily.index.max())} |",
        f"| `model_table` | {len(table):,} | {table.shape[1]} | {fmt_date(table.index.min())} | {fmt_date(table.index.max())} |",
        "",
        f"Final holdout (do not use for tuning): {HOLDOUT_START} onward, "
        f"{int((table.index >= HOLDOUT_START).sum()):,} rows.",
        "",
        "## Raw data quality",
        "",
        "| " + " | ".join(quality.columns) + " |",
        "|" + "---|" * len(quality.columns),
        *["| " + " | ".join(str(v) for v in row) + " |" for row in quality.itertuples(index=False)],
        "",
        "## Regimes",
        "",
        "| Label | Share of days | Episodes | Median length (days) | Longest (days) |",
        "|---|---|---|---|---|",
    ]
    for col in ALL_LABELS:
        ep = episodes(table[col])
        lines.append(
            f"| `{col}` | {table[col].mean():.1%} | {len(ep)} | "
            f"{ep['days'].median():.0f} | {ep['days'].max():.0f} |"
        )

    lines += ["", "- `bear`: **main label**. Standard 20% rule, peak to trough.",
              "- `bear_merged`: **robustness label**. `bear`, with bear markets merged when the rally "
              "between them lasts at most 126 trading days and stays below the starting peak.",
              "- `bear_recovery`: causal \"20% down and not yet back to the old high\". Used as the "
              "feature `below_peak_regime`, not as a target.",
              "- `stressed`: **main label**. 21-day realized vol above the 80th percentile of the "
              "**previous 5 years**' realized vol (unusually volatile for the recent market).",
              "- `stressed_fixed`: **robustness label**. 21-day realized vol above a fixed 25% a year.",
              "",
              "### Stressed share by decade",
              "",
              "| Decade | `stressed` (5-year window) | `stressed_fixed` (> 25% vol) |",
              "|---|---|---|"]
    decade = (table.index.year // 10) * 10
    by_dec = table[["stressed", "stressed_fixed"]].groupby(decade).mean()
    for dec, row in by_dec.iterrows():
        cells = ["–" if pd.isna(v) else f"{v:.0%}" for v in row]
        lines.append(f"| {dec}s | {cells[0]} | {cells[1]} |")

    merged_eps = episodes(table["bear_merged"])
    lines += ["", "### Bear markets found", "",
              "`bear` (main) episodes. The last column shows which `bear_merged` episode each belongs to; "
              "rows sharing a number were merged.", "",
              "| Start | End | Trading days | Merged episode |", "|---|---|---|---|"]
    for _, row in episodes(table["bear"]).iterrows():
        k = merged_eps.index[(merged_eps["start"] <= row["start"]) & (merged_eps["end"] >= row["end"])]
        lines.append(f"| {fmt_date(row['start'])} | {fmt_date(row['end'])} | {row['days']} | "
                     f"{int(k[0]) + 1 if len(k) else ''} |")

    lines += ["", "## Targets", "", "| Target | Non-missing | Share = 1 |", "|---|---|---|"]
    for h in horizons:
        for label in TARGET_LABELS:
            col = f"target_{label}_h{h}"
            lines.append(f"| `{col}` | {table[col].notna().sum():,} | {table[col].mean():.1%} |")

    lines += ["", "## Feature availability", "", "| Feature | First valid date | Non-missing rows |", "|---|---|---|"]
    for name, row in feature_availability(table, feature_cols).iterrows():
        lines.append(f"| `{name}` | {fmt_date(row['first_valid'])} | {row['non_missing']:,} |")

    counts = {s: sum(st == s for _, st, _ in checks) for s in ("PASS", "WARN", "FAIL")}
    lines += ["", f"## Checks ({counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail)", "",
              "WARN on a known-bear-market check means that label splits the bear market at a "
              "20%+ rally. This is expected for `bear` (the main label) and is the reason `bear_merged` exists. "
              "WARN on a medium sell-off for `stressed_fixed` is also expected: a fixed 25% level misses "
              "stress in calm eras, which is why the main stress label is relative.", "",
              "| Check | Result | Detail |", "|---|---|---|"]
    for name, status, detail in checks:
        lines.append(f"| {name} | {status} | {detail} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(df: pd.DataFrame, stem: str, fmt: str) -> list[Path]:
    paths = []
    if fmt in ("csv", "both"):
        p = DATA_PROCESSED / f"{stem}.csv"
        df.to_csv(p, date_format="%Y-%m-%d", float_format="%.6g")
        paths.append(p)
    if fmt in ("parquet", "both"):
        p = DATA_PROCESSED / f"{stem}.parquet"
        df.to_parquet(p)
        paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS),
                        help="prediction horizons in trading days (default: %(default)s)")
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="both",
                        help="output file format (default: %(default)s)")
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    log("1-2/6 Loading, cleaning and aligning raw data ...")
    daily, quality = build_dataset()
    quality = quality.round(4)
    log(f"      sp500_daily: {daily.shape[0]:,} rows x {daily.shape[1]} columns "
        f"({daily.index.min().date()} -> {daily.index.max().date()})")

    log(f"3-4/6 Building labels, targets (h = {args.horizons}) and features ...")
    table = build_model_table(daily, horizons=args.horizons)
    label_cols = set(ALL_LABELS) | {"stress_threshold"}
    feature_cols = [c for c in table.columns if c not in label_cols and not c.startswith("target_")]
    log(f"      model_table: {table.shape[0]:,} rows x {table.shape[1]} columns "
        f"({len(feature_cols)} features)")

    log("5/6   Running checks ...")
    checks = run_checks(table, daily, args.horizons)
    for name, status, detail in checks:
        if status != "PASS":
            log(f"      {status}: {name} ({detail})")
    failed = [c for c in checks if c[1] == "FAIL"]
    n_pass = sum(c[1] == "PASS" for c in checks)
    log(f"      {n_pass} pass, {len(checks) - n_pass - len(failed)} warn, {len(failed)} fail")

    log("6/6   Saving outputs ...")
    written = save(daily, "sp500_daily", args.format) + save(table, "model_table", args.format)
    quality.to_csv(REPORTS / "data_quality.csv", index=False)
    summary_path = REPORTS / "processing_summary.md"
    write_summary(summary_path, daily, table, quality, checks, args.horizons, feature_cols)
    for p in written + [REPORTS / "data_quality.csv", summary_path]:
        log(f"      wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)")

    print("\nShare of days in each regime:")
    print(table[list(ALL_LABELS)].mean().map("{:.1%}".format).to_string())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
