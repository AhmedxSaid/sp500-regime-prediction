"""Train and evaluate baselines and models walk-forward (1980-2021), then write a report.

Run `python process_data.py` first.

Usage (from the repo root):
    python run_models.py                       # main feature set (from 1962), test 1980-2021
    python run_models.py --features modern     # main + VIX (from 1990), test 2000-2021

Outputs:
    data/processed/predictions_<features>.parquet   every out-of-sample prediction
    reports/model_results_<features>.csv            metrics per target and method
    reports/model_results_<features>.md             the report
    docs/figures/model_timeline_<features>.png      predicted probability vs. actual regimes

The final holdout (2022 onward) is NOT used here.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sp500.config import (  # noqa: E402
    DATA_PROCESSED, FEATURES_MAIN, FEATURES_MODERN, FIRST_TEST_YEAR, HOLDOUT_START, HORIZONS,
    MODEL_START, MODERN_FIRST_TEST_YEAR, MODERN_START, REPORTS, WALK_FORWARD_GAP,
)
from sp500.features import bear_price_series  # noqa: E402
from sp500.labels import episodes  # noqa: E402
from sp500.models import BASELINES, MODELS, lead_times, run_walk_forward, score  # noqa: E402

MAIN_LABELS = ("bear", "stressed")
ROBUST_LABELS = ("bear_merged", "stressed_fixed")
METHOD_NAMES = {
    "persistence": "Baseline: no change",
    "ma200_rule": "Baseline: price < 200-day MA",
    "hmm": "Baseline: 2-state HMM",
    "logreg": "Logistic regression",
    "lightgbm": "LightGBM",
    "svm": "SVM (RBF)",
}
FIG_DIR = ROOT / "docs" / "figures"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def md_table(df: pd.DataFrame, fmt: dict[str, str], bold_max: list[str], bold_min: list[str]) -> list[str]:
    """Markdown table; best value per column in bold (ties included)."""
    cols = list(df.columns)
    lines = ["| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("–")
                continue
            s = fmt.get(c, "{:.3f}").format(v)
            col = df[c].dropna()
            if (c in bold_max and np.isclose(v, col.max())) or (c in bold_min and np.isclose(v, col.min())):
                s = f"**{s}**"
            cells.append(s)
        lines.append(f"| {METHOD_NAMES.get(idx, idx)} | " + " | ".join(cells) + " |")
    return lines


def timeline_figure(preds: pd.DataFrame, table: pd.DataFrame, method: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, label, color in [(axes[0], "stressed", "C4"), (axes[1], "bear", "C3")]:
        g = preds[(preds.label == label) & (preds.horizon == 21) & (preds.method == method)].set_index("date")
        g = g.loc["2000":]
        for _, row in episodes(table.loc[g.index.min():g.index.max(), label]).iterrows():
            ax.axvspan(row["start"], row["end"], color=color, alpha=0.2, lw=0)
        ax.plot(g.index, g["prob"], color="k", lw=0.7, label=f"{METHOD_NAMES[method]}: P({label} in 21 days)")
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("probability")
        ax.set_title(f"{label.capitalize()} in 21 days: predicted probability (line) vs. actual {label} periods (shaded)")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", choices=["main", "modern"], default="main")
    parser.add_argument("--reuse", action="store_true",
                        help="reuse saved predictions and only rebuild the metrics and report")
    args = parser.parse_args()

    features = FEATURES_MAIN if args.features == "main" else FEATURES_MODERN
    start = MODEL_START if args.features == "main" else MODERN_START
    first_test = FIRST_TEST_YEAR if args.features == "main" else MODERN_FIRST_TEST_YEAR
    tag = args.features

    table = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    daily = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    bear_price = bear_price_series(daily)

    pred_path = DATA_PROCESSED / f"predictions_{tag}.parquet"
    if args.reuse and pred_path.exists():
        preds = pd.read_parquet(pred_path)
        log(f"Reusing {len(preds):,} saved predictions from {pred_path.name}")
    else:
        log(f"Walk-forward: features={tag} ({len(features)}), train from {start}, "
            f"test {first_test}-{pd.Timestamp(HOLDOUT_START).year - 1}, gap {WALK_FORWARD_GAP} days")
        t0 = time.time()
        preds = run_walk_forward(table, bear_price, features, MAIN_LABELS + ROBUST_LABELS, HORIZONS,
                                 first_test, WALK_FORWARD_GAP, start, HOLDOUT_START, MODELS, log=log)
        log(f"Done in {time.time() - t0:.0f}s; {len(preds):,} predictions")
        preds.to_parquet(pred_path)

    # metrics
    metrics = preds.groupby(["label", "horizon", "method"]).apply(score, include_groups=False).reset_index()
    metrics.to_csv(REPORTS / f"model_results_{tag}.csv", index=False)
    leads = lead_times(preds)

    # pick the best model (not baseline) by mean PR-AUC over the main targets
    main_m = metrics[metrics.label.isin(MAIN_LABELS) & metrics.method.isin(MODELS)]
    best = main_m.groupby("method")["pr_auc"].mean().idxmax()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / f"model_timeline_{tag}.png"
    timeline_figure(preds, table, best, fig_path)

    # ---------------- report
    order = list(BASELINES) + list(MODELS)
    years = f"{first_test}–{pd.Timestamp(HOLDOUT_START).year - 1}"
    L = [
        f"# Model results ({tag} features)",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `run_models.py`. Walk-forward out-of-sample "
        f"predictions for **{years}** (one test year at a time; models retrained each year on all earlier "
        f"data from {start[:4]}, with a {WALK_FORWARD_GAP}-day gap). The 2022+ holdout is not used.",
        "",
        f"Features ({len(features)}): " + ", ".join(f"`{f}`" for f in features),
        "",
        "**How to read the metrics**",
        "",
        "- **PR-AUC:** how well the method ranks future regime days above normal days (1 = perfect; "
        "a random guess scores the base rate).",
        "- **Onset PR-AUC:** the same, but only on days when the regime is **not active yet**: can it "
        "predict a regime *starting*? This is where a model must beat \"no change\".",
        "- **Brier:** mean squared error of the probability (lower is better; 0 = perfect).",
        "- **Flips/yr:** how often the 50% signal switches on/off per year (compare with the true number).",
        "- Bold = best in the column.",
        "",
    ]
    for title, labels in [("Main targets", MAIN_LABELS), ("Robustness targets", ROBUST_LABELS)]:
        L += [f"## {title}", ""]
        for label in labels:
            for h in HORIZONS:
                m = metrics[(metrics.label == label) & (metrics.horizon == h)].set_index("method").reindex(order)
                base, onset_base = m["base_rate"].dropna().iloc[0], m["onset_base_rate"].dropna().iloc[0]
                true_flips = m["true_flips_per_year"].dropna().iloc[0]
                L += [f"### `{label}` in {h} days",
                      "",
                      f"Base rate {base:.1%} of days; on not-yet-{label} days, {onset_base:.1%} turn {label} "
                      f"within {h} days. True regime switches: {true_flips:.1f} per year.", ""]
                show = m[["pr_auc", "onset_pr_auc", "brier", "signal_flips_per_year"]].rename(columns={
                    "pr_auc": "PR-AUC", "onset_pr_auc": "Onset PR-AUC", "brier": "Brier",
                    "signal_flips_per_year": "Flips/yr"})
                L += md_table(show, {"Flips/yr": "{:.1f}"}, ["PR-AUC", "Onset PR-AUC"], ["Brier"]) + [""]

    L += ["## Early warning (21-day horizon)", "",
          "For each regime start in the test years: consecutive trading days **before** the start on which "
          "the method's probability was already ≥ 50% (0 = no warning). Median over all starts, and the "
          "share of starts with at least 5 days of warning.", "",
          "| Method | Stress starts: median days | ≥ 5 days warned | Bear starts: median days | ≥ 5 days warned |",
          "|---|---|---|---|---|"]
    for method in order:
        cells = []
        for label in ("stressed", "bear"):  # same order as the column headers
            g = leads[(leads.label == label) & (leads.horizon == 21) & (leads.method == method)]
            if g.empty or (method == "ma200_rule" and label == "stressed"):
                cells += ["–", "–"]
            else:
                cells += [f"{g.warning_days.median():.0f}", f"{(g.warning_days >= 5).mean():.0%}"]
        L.append(f"| {METHOD_NAMES[method]} | " + " | ".join(cells) + " |")
    n_stress = leads[(leads.label == "stressed") & (leads.horizon == 21) & (leads.method == "logreg")].shape[0]
    n_bear = leads[(leads.label == "bear") & (leads.horizon == 21) & (leads.method == "logreg")].shape[0]
    L += ["", f"({n_stress} stress starts and {n_bear} bear starts in the test years. A bear start is "
          "dated at the market peak, which is only known in hindsight, so \"warning\" before it means "
          "the signal was already on at the top, e.g. because the market was still below an older high.)", ""]

    L += ["## Timeline", "",
          f"Best model by mean PR-AUC on the main targets: **{METHOD_NAMES[best]}**.", "",
          f"![timeline](../docs/figures/{fig_path.name})", ""]
    (REPORTS / f"model_results_{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote reports/model_results_{tag}.md, .csv and docs/figures/{fig_path.name}")

    # console summary
    summ = metrics[metrics.label.isin(MAIN_LABELS)].pivot_table(
        index="method", columns=["label", "horizon"], values="pr_auc").reindex(order)
    print("\nPR-AUC (main targets):\n" + summ.round(3).to_string())
    summ = metrics[metrics.label.isin(MAIN_LABELS)].pivot_table(
        index="method", columns=["label", "horizon"], values="onset_pr_auc").reindex(order)
    print("\nOnset PR-AUC (main targets):\n" + summ.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
