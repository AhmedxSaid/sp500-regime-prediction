"""Hypothesis tests and the validation-vs-holdout comparison (report sections 4.6 and 4.7).

Tests (Test 1 on the full history; Tests 2-4 on 1962-2021, the modeling period):
    1. Ljung-Box (20 lags) on daily returns and on absolute returns: volatility clustering.
    2. Chi-square: stressed today vs. stressed in 21 days: persistence (Cramer's V).
    3. Mann-Whitney U: 21-day realized vol, by stressed in 21 days (rank-biserial r).
    4. Mann-Whitney U: distance from the 200-day average, by bear in 21 days.
    +  Jarque-Bera normality test of daily returns.
Comparison: validation period (1962-2021) vs. holdout (2022+), plus a Kolmogorov-Smirnov
test on 21-day realized volatility.

Daily observations are autocorrelated and samples are large, so p-values overstate the
evidence: read the effect sizes.

Usage (from the repo root, after `python process_data.py`):
    python run_statistical_tests.py
Output: reports/statistical_tests.md
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sp500.config import DATA_PROCESSED, HOLDOUT_START, MODEL_START, REPORTS  # noqa: E402

VALIDATION = (MODEL_START[:4], str(pd.Timestamp(HOLDOUT_START).year - 1))  # 1962-2021


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def mann_whitney(x: pd.Series, group: pd.Series) -> dict:
    a, b = x[group == 1], x[group == 0]
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"n1": len(a), "n0": len(b), "median1": a.median(), "median0": b.median(),
            "U": u.statistic, "p": u.pvalue, "r": 2 * u.statistic / (len(a) * len(b)) - 1}


def main() -> int:
    table = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    daily = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    r = np.log(daily["gspc_close"]).diff().dropna()
    val = table.loc[VALIDATION[0]:VALIDATION[1]]

    # Test 1: Ljung-Box
    lb = {name: acorr_ljungbox(x, lags=[20], return_df=True).iloc[0]
          for name, x in [("returns", r), ("absolute returns", r.abs())]}
    ac1 = {"returns": r.autocorr(1), "absolute returns": r.abs().autocorr(1)}
    jb = stats.jarque_bera(r)

    # Test 2: chi-square, stressed today vs. in 21 days
    c = val[["stressed", "target_stressed_h21"]].dropna()
    ct = pd.crosstab(c["stressed"], c["target_stressed_h21"])
    chi2, p_chi, dof, expected = stats.chi2_contingency(ct, correction=False)
    cramer_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))

    # Tests 3 and 4: Mann-Whitney
    s = val[["rv_21d", "target_stressed_h21"]].dropna()
    t3 = mann_whitney(s["rv_21d"], s["target_stressed_h21"])
    s = val[["dist_ma200", "target_bear_h21"]].dropna()
    t4 = mann_whitney(s["dist_ma200"], s["target_bear_h21"])

    # Validation vs. holdout
    periods = {f"Validation {VALIDATION[0]}–{VALIDATION[1]}": val,
               f"Holdout {HOLDOUT_START[:4]}–{table.index.max().year}": table.loc[HOLDOUT_START:]}
    comp = {}
    for name, x in periods.items():
        rr = x["ret_1d"].dropna()
        comp[name] = {
            "Trading days": f"{len(x):,}",
            "Annual return / volatility": f"{rr.mean() * 252:.1%} / {rr.std() * np.sqrt(252):.1%}",
            "Share of days in a bear market": f"{x['bear'].mean():.1%}",
            "Share of days stressed (main label)": f"{x['stressed'].mean():.1%}",
            "Share of days stressed (fixed 25% label)": f"{x['stressed_fixed'].mean():.1%}",
            "Median 21-day realized volatility": f"{x['rv_21d'].median():.1%}",
            "Deepest drawdown from a peak": f"{x['drawdown'].min():.1%}",
            "Worst day (log return)": f"{rr.min():.1%}",
        }
    ks = stats.ks_2samp(*[x["rv_21d"].dropna() for x in periods.values()])

    # ---------------- report
    L = ["# Statistical tests and validation vs. holdout", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `run_statistical_tests.py`. Test 1 uses the full history; "
         f"Tests 2–4 use {VALIDATION[0]}–{VALIDATION[1]}, the modeling period. Daily observations are autocorrelated "
         "and samples are large, so p-values overstate the evidence: read the effect sizes.", "",
         "## Hypothesis tests", "",
         "| Test | H0 | n | Statistic | p-value | Effect size | Decision |",
         "|---|---|---|---|---|---|---|"]
    for name in ("returns", "absolute returns"):
        L.append(f"| Ljung–Box, {name} | No autocorrelation up to lag 20 | {len(r):,} | Q(20) = {lb[name].lb_stat:,.1f} | "
                 f"{fmt_p(lb[name].lb_pvalue)} | lag-1 autocorrelation {ac1[name]:+.3f} | "
                 f"{'Reject' if lb[name].lb_pvalue < 0.05 else 'Keep'} H0 |")
    L.append(f"| Chi-square, stressed today vs. in 21 days | Independent | {int(ct.values.sum()):,} | "
             f"χ²({dof}) = {chi2:,.1f} | {fmt_p(p_chi)} | Cramér's V = {cramer_v:.2f} | "
             f"{'Reject' if p_chi < 0.05 else 'Keep'} H0 |")
    for label, t, var in [("21-day realized vol, by stressed in 21 days", t3, "volatility"),
                          ("distance from 200-day average, by bear in 21 days", t4, "distance")]:
        L.append(f"| Mann–Whitney, {label} | Same distribution | {t['n1']:,} vs. {t['n0']:,} | U = {t['U']:.3g} | "
                 f"{fmt_p(t['p'])} | r = {t['r']:+.2f} (medians {t['median1']:+.1%} vs. {t['median0']:+.1%}) | "
                 f"{'Reject' if t['p'] < 0.05 else 'Keep'} H0 |")
    L += ["", f"Smallest expected count in the chi-square test: {expected.min():,.0f} (condition: ≥ 5).", "",
          f"Jarque–Bera normality test of daily returns: JB = {jb.statistic:,.0f}, p {fmt_p(jb.pvalue)} "
          f"(skew {stats.skew(r):+.2f}, excess kurtosis {stats.kurtosis(r):.1f}).", "",
          "Contingency table (rows: stressed today; columns: stressed in 21 days):", "",
          "| | Not stressed in 21 days | Stressed in 21 days |", "|---|---|---|",
          f"| Not stressed today | {ct.iloc[0, 0]:,} | {ct.iloc[0, 1]:,} |",
          f"| Stressed today | {ct.iloc[1, 0]:,} | {ct.iloc[1, 1]:,} |", "",
          "## Validation vs. holdout", "",
          "| Measure | " + " | ".join(comp) + " |", "|---|" + "---|" * len(comp)]
    for key in next(iter(comp.values())):
        L.append(f"| {key} | " + " | ".join(v[key] for v in comp.values()) + " |")
    L += ["", f"Kolmogorov–Smirnov test on 21-day realized volatility, validation vs. holdout: "
              f"D = {ks.statistic:.3f}, p {fmt_p(ks.pvalue)}."]
    out = REPORTS / "statistical_tests.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
