"""Test candidate feature groups one at a time on top of the main feature set.

Every feature set is evaluated on exactly the same rows: training from 1993-03-01 (when all
candidate features exist), walk-forward test years 2000-2021, same models (logistic regression,
LightGBM), same gap. A group is kept only if it passes the rule fixed in config.py beforehand:

    mean PR-AUC over the 4 main targets improves by >= GROUP_MIN_GAIN for BOTH models, and
    mean onset PR-AUC drops by no more than GROUP_MAX_ONSET_LOSS for either.

Usage (from the repo root, after `python process_data.py`):
    python run_feature_groups.py

Outputs: reports/feature_groups.md, reports/feature_groups.csv
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sp500.config import (  # noqa: E402
    DATA_PROCESSED, FEATURE_GROUPS, FEATURES_MAIN, GROUP_MAX_ONSET_LOSS, GROUP_MIN_GAIN, HOLDOUT_START,
    HORIZONS, REPORTS, WALK_FORWARD_GAP,
)
from sp500.features import bear_price_series  # noqa: E402
from sp500.models import run_walk_forward, score  # noqa: E402

START = "1993-03-01"
FIRST_TEST_YEAR = 2000
MODELS = ("logreg", "lightgbm")
MAIN_LABELS = ("bear", "stressed")
ROBUST_LABELS = ("bear_merged", "stressed_fixed")
MODEL_NAMES = {"logreg": "Logistic regression", "lightgbm": "LightGBM"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def evaluate(table, bear_price, features, name):
    t0 = time.time()
    preds = run_walk_forward(table, bear_price, features, MAIN_LABELS + ROBUST_LABELS, HORIZONS,
                             FIRST_TEST_YEAR, WALK_FORWARD_GAP, START, HOLDOUT_START, MODELS,
                             log=lambda m: None)
    preds = preds[preds.method.isin(MODELS)]
    m = preds.groupby(["label", "horizon", "method"]).apply(score, include_groups=False).reset_index()
    m["feature_set"] = name
    # per-year PR-AUC on the main stress target (for "how many years did it help")
    yearly = (preds[(preds.label == "stressed") & (preds.horizon == 21)]
              .groupby(["method", "test_year"]).apply(score, include_groups=False)["pr_auc"].rename(name))
    log(f"  {name:<22} ({len(features)} features) done in {time.time() - t0:.0f}s")
    return m, yearly


def summarize(metrics: pd.DataFrame, labels) -> pd.DataFrame:
    sub = metrics[metrics.label.isin(labels)]
    return sub.groupby(["feature_set", "method"])[["pr_auc", "onset_pr_auc"]].mean()


def main() -> int:
    table = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    daily = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    bear_price = bear_price_series(daily)

    every = FEATURES_MAIN + [f for g in FEATURE_GROUPS.values() for f in g]
    common = table[table[every].notna().all(axis=1)]  # identical rows for every feature set
    log(f"Common sample: {common.index.min().date()} -> {pd.Timestamp(HOLDOUT_START).year - 1}, "
        f"test years {FIRST_TEST_YEAR}-{pd.Timestamp(HOLDOUT_START).year - 1}, models: {', '.join(MODELS)}")

    sets = {"base": FEATURES_MAIN}
    sets.update({f"base + {g}": FEATURES_MAIN + cols for g, cols in FEATURE_GROUPS.items()})
    results, yearly = [], []
    for name, feats in sets.items():
        m, y = evaluate(common, bear_price, feats, name)
        results.append(m)
        yearly.append(y)

    # decision per group
    def decide(all_metrics):
        main = summarize(all_metrics, MAIN_LABELS)
        decisions = {}
        for g in FEATURE_GROUPS:
            name = f"base + {g}"
            gain = main.loc[name, "pr_auc"] - main.loc["base", "pr_auc"]
            onset = main.loc[name, "onset_pr_auc"] - main.loc["base", "onset_pr_auc"]
            keep = bool((gain >= GROUP_MIN_GAIN).all() and (onset >= -GROUP_MAX_ONSET_LOSS).all())
            decisions[g] = {"keep": keep, "gain": gain, "onset": onset}
        return decisions

    metrics = pd.concat(results, ignore_index=True)
    decisions = decide(metrics)
    kept = [g for g, d in decisions.items() if d["keep"]]
    if len(kept) > 1:  # confirm the combination
        name = "base + " + " + ".join(kept)
        m, y = evaluate(common, bear_price, FEATURES_MAIN + [f for g in kept for f in FEATURE_GROUPS[g]], name)
        metrics = pd.concat([metrics, m], ignore_index=True)
        yearly.append(y)
    metrics.to_csv(REPORTS / "feature_groups.csv", index=False)
    yearly = pd.concat(yearly, axis=1)

    # ---------------- report
    set_order = list(dict.fromkeys(metrics.feature_set))
    L = [
        "# Feature group tests",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `run_feature_groups.py`. Same rows for every feature set: "
        f"training from {common.index.min().date()}, walk-forward test years **{FIRST_TEST_YEAR}–"
        f"{pd.Timestamp(HOLDOUT_START).year - 1}**, {WALK_FORWARD_GAP}-day gap. Models: logistic regression and LightGBM "
        "(the two best in the main run). The 2022+ holdout is not used.",
        "",
        "**Decision rule (fixed before testing):** keep a group only if mean PR-AUC over the 4 main targets "
        f"(bear and stressed, 5 and 21 days) improves by **≥ {GROUP_MIN_GAIN}** for **both** models, and mean "
        f"onset PR-AUC drops by no more than {GROUP_MAX_ONSET_LOSS} for either.",
        "",
        "| Group | Features | Logistic regression: Δ PR-AUC | LightGBM: Δ PR-AUC | Δ onset PR-AUC (LR / LGBM) | Decision |",
        "|---|---|---|---|---|---|",
    ]
    for g, d in decisions.items():
        L.append(f"| {g} | " + ", ".join(f"`{c}`" for c in FEATURE_GROUPS[g]) + " | "
                 f"{d['gain']['logreg']:+.3f} | {d['gain']['lightgbm']:+.3f} | "
                 f"{d['onset']['logreg']:+.3f} / {d['onset']['lightgbm']:+.3f} | "
                 f"{'✅ keep' if d['keep'] else '❌ drop'} |")

    for title, labels in [("Main targets", MAIN_LABELS), ("Robustness targets", ROBUST_LABELS)]:
        L += ["", f"## {title}: PR-AUC by target", ""]
        piv = metrics[metrics.label.isin(labels)].pivot_table(
            index=["feature_set", "method"], columns=["label", "horizon"], values="pr_auc")
        cols = list(piv.columns)
        L += ["| Feature set | Model | " + " | ".join(f"{l} {h}d" for l, h in cols) + " | Mean |",
              "|---|---|" + "---|" * (len(cols) + 1)]
        for s in set_order:
            for mth in MODELS:
                row = piv.loc[(s, mth)]
                L.append(f"| {s} | {MODEL_NAMES[mth]} | " + " | ".join(f"{v:.3f}" for v in row) +
                         f" | **{row.mean():.3f}** |")
        L += ["", f"Onset PR-AUC ({title.lower()}):", ""]
        piv = metrics[metrics.label.isin(labels)].pivot_table(
            index=["feature_set", "method"], columns=["label", "horizon"], values="onset_pr_auc")
        L += ["| Feature set | Model | " + " | ".join(f"{l} {h}d" for l, h in cols) + " | Mean |",
              "|---|---|" + "---|" * (len(cols) + 1)]
        for s in set_order:
            for mth in MODELS:
                row = piv.loc[(s, mth)]
                L.append(f"| {s} | {MODEL_NAMES[mth]} | " + " | ".join(f"{v:.3f}" for v in row) +
                         f" | **{row.mean():.3f}** |")

    L += ["", "## Year by year (`stressed` in 21 days)", "",
          "Number of test years in which each set beat the base set on PR-AUC (years with both classes):", ""]
    for mth in MODELS:
        y = yearly.loc[mth].dropna()
        parts = [f"{c}: {int((y[c] > y['base']).sum())}/{len(y)}" for c in y.columns if c != "base"]
        L.append(f"- {MODEL_NAMES[mth]}: " + "; ".join(parts))
    (REPORTS / "feature_groups.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("wrote reports/feature_groups.md and .csv")

    print("\nMean over main targets:\n" + summarize(metrics, MAIN_LABELS).round(3).to_string())
    for g, d in decisions.items():
        print(f"{g}: {'KEEP' if d['keep'] else 'DROP'}  gain={d['gain'].round(3).to_dict()}  "
              f"onset={d['onset'].round(3).to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
