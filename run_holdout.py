"""ONE-TIME final test on the 2022+ holdout, with the setup fixed before it was ever used.

Setup (unchanged from validation): FEATURES_MAIN (13), training from 1962, yearly walk-forward
(refit each January on all earlier data, 21-day gap), all baselines and models, all 4 labels,
horizons 5 and 21 days. Test years 2022 through the last date with a known target.

The models are fitted only once. To rebuild the report from the saved predictions without
refitting, use --report-only.

Usage (from the repo root, after `python process_data.py`):
    python run_holdout.py
Outputs: data/processed/predictions_holdout.parquet, reports/holdout_results.md/.csv,
         docs/figures/holdout_timeline.png
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
from sklearn.metrics import average_precision_score  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from interpret_models import signal_offset  # noqa: E402
from sp500.config import (  # noqa: E402
    DATA_PROCESSED, FEATURES_MAIN, HOLDOUT_START, HORIZONS, MODEL_START, REPORTS, WALK_FORWARD_GAP,
)
from sp500.features import bear_price_series  # noqa: E402
from sp500.labels import episodes  # noqa: E402
from sp500.models import BASELINES, MODELS, run_walk_forward, score  # noqa: E402

LABELS = ("bear", "stressed", "bear_merged", "stressed_fixed")
NAMES = {"persistence": "Baseline: no change", "ma200_rule": "Baseline: price < 200-day MA",
         "hmm": "Baseline: 2-state HMM", "logreg": "Logistic regression", "lightgbm": "LightGBM",
         "svm": "SVM (RBF)"}
FIG = ROOT / "docs" / "figures"


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def block_bootstrap_diff(g_model, g_base, n_rep=1000, block=63, seed=0):
    """95% CI for PR-AUC(model) - PR-AUC(baseline), resampling blocks of consecutive days."""
    a = g_model.dropna(subset=["prob", "truth"]).set_index("date")
    b = g_base.dropna(subset=["prob", "truth"]).set_index("date")
    idx = a.index.intersection(b.index)
    y, pa, pb = a.loc[idx, "truth"].to_numpy(int), a.loc[idx, "prob"].to_numpy(), b.loc[idx, "prob"].to_numpy()
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_rep):
        starts = rng.integers(0, n - block, size=int(np.ceil(n / block)))
        s = np.concatenate([np.arange(k, k + block) for k in starts])[:n]
        if y[s].min() == y[s].max():
            continue
        diffs.append(average_precision_score(y[s], pa[s]) - average_precision_score(y[s], pb[s]))
    point = average_precision_score(y, pa) - average_precision_score(y, pb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from saved predictions")
    args = parser.parse_args()

    table = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    daily = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    first_year = pd.Timestamp(HOLDOUT_START).year
    pred_path = DATA_PROCESSED / "predictions_holdout.parquet"

    if args.report_only:
        preds = pd.read_parquet(pred_path)
        log(f"Report only: {len(preds):,} saved holdout predictions")
    else:
        log(f"ONE-TIME HOLDOUT: features=main ({len(FEATURES_MAIN)}), train from {MODEL_START}, "
            f"test {first_year} -> {table.index.max().date()}, yearly refit, gap {WALK_FORWARD_GAP}")
        t0 = time.time()
        preds = run_walk_forward(table, bear_price_series(daily), FEATURES_MAIN, LABELS, HORIZONS,
                                 first_year, WALK_FORWARD_GAP, MODEL_START, holdout_start="2100-01-01",
                                 models=MODELS, log=log)
        preds = preds[preds.date >= HOLDOUT_START]
        preds.to_parquet(pred_path)
        log(f"Done in {time.time() - t0:.0f}s; {len(preds):,} predictions")

    metrics = preds.groupby(["label", "horizon", "method"]).apply(score, include_groups=False).reset_index()
    metrics.to_csv(REPORTS / "holdout_results.csv", index=False)
    val = pd.read_csv(REPORTS / "model_results_main.csv")
    last_known = preds.dropna(subset=["truth"]).date.max().date()
    order = list(BASELINES) + list(MODELS)

    # ---------------- figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for ax, label, color in [(axes[0], "stressed", "C4"), (axes[1], "bear", "C3")]:
        for _, ep in episodes(table.loc[HOLDOUT_START:, label]).iterrows():
            ax.axvspan(ep["start"], ep["end"], color=color, alpha=0.2, lw=0)
        for m, c in [("logreg", "k"), ("lightgbm", "C1")]:
            g = preds[(preds.label == label) & (preds.horizon == 21) & (preds.method == m)]
            ax.plot(g.date, g.prob, color=c, lw=0.8, label=NAMES[m])
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("probability")
        ax.set_title(f"HOLDOUT: P({label} in 21 days), lines; actual {label} periods, shaded")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "holdout_timeline.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------------- report
    L = ["# Final holdout results (one-time test)", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `run_holdout.py`. Test period **{HOLDOUT_START} → "
         f"{last_known}** (last day whose target is already known). Models refit each January on all data "
         f"from {MODEL_START[:4]}, {WALK_FORWARD_GAP}-day gap. Setup identical to validation; nothing was tuned on this period.", ""]

    for label in LABELS:
        for h in HORIZONS:
            m = metrics[(metrics.label == label) & (metrics.horizon == h)].set_index("method").reindex(order)
            v = val[(val.label == label) & (val.horizon == h)].set_index("method").reindex(order)
            base = m["base_rate"].dropna().iloc[0]
            L += [f"## `{label}` in {h} days", "",
                  f"Holdout base rate {base:.1%} ({int(m['n_days'].dropna().iloc[0]):,} days).", "",
                  "| Method | PR-AUC (holdout) | PR-AUC (validation 1980–2021) | Onset PR-AUC (holdout) | Brier (holdout) |",
                  "|---|---|---|---|---|"]
            best = m["pr_auc"].max()
            for meth in order:
                r, rv = m.loc[meth], v.loc[meth]
                if pd.isna(r["pr_auc"]):
                    if not pd.isna(r["base_rate"]):
                        L.append(f"| {NAMES[meth]} | n/a | | | |")
                    continue
                pr = f"**{r.pr_auc:.3f}**" if np.isclose(r.pr_auc, best) else f"{r.pr_auc:.3f}"
                onset = "–" if pd.isna(r.onset_pr_auc) else f"{r.onset_pr_auc:.3f}"
                L.append(f"| {NAMES[meth]} | {pr} | {rv.pr_auc:.3f} | {onset} | {r.brier:.3f} |")
            L.append("")

    # bootstrap: best model vs best baseline, main targets
    L += ["## Is the improvement over the baselines real? (block bootstrap)", "",
          "95% confidence interval for PR-AUC(model) − PR-AUC(best baseline), resampling 63-day blocks "
          "(1,000 repetitions). If the interval stays above 0, the model is better than the baseline in the holdout.", "",
          "| Target | Model | Best baseline | Difference | 95% CI |", "|---|---|---|---|---|"]
    boot_rows = []
    for label in ("stressed", "bear"):
        for h in HORIZONS:
            sub = metrics[(metrics.label == label) & (metrics.horizon == h)].set_index("method")
            bl = sub.loc[list(BASELINES), "pr_auc"].dropna().idxmax()
            for model in ("logreg", "lightgbm"):
                g = preds[(preds.label == label) & (preds.horizon == h)]
                d, lo, hi = block_bootstrap_diff(g[g.method == model], g[g.method == bl])
                boot_rows.append((label, h, model, bl, d, lo, hi))
                L.append(f"| `{label}` {h}d | {NAMES[model]} | {NAMES[bl]} | {d:+.3f} | {lo:+.3f} to {hi:+.3f} |")
    L.append("")

    # stress episodes in the holdout
    lab = table.loc[HOLDOUT_START:last_known.isoformat(), "stressed"]
    eps = episodes(lab)
    eps = eps[eps.days >= 10]
    L += ["## Stress episodes in the holdout (lasting ≥ 10 trading days)", "",
          "Signal start relative to the first stressed day (21-day predictions): negative = warned in advance.", "",
          "| Stress start | Days | HMM | Logistic | LightGBM |", "|---|---|---|---|---|"]
    for _, ep in eps.iterrows():
        cells = []
        for m in ("hmm", "logreg", "lightgbm"):
            g = preds[(preds.label == "stressed") & (preds.horizon == 21) & (preds.method == m)] \
                .sort_values("date").reset_index(drop=True)
            pos = g.index[g.date == ep["start"]]
            cells.append(signal_offset(g, int(pos[0])) if len(pos) else "n/a")
        L.append(f"| {ep['start'].date()} | {ep['days']} | " + " | ".join(cells) + " |")
    L.append("")

    # bear markets in the holdout
    bl = table.loc[HOLDOUT_START:, "bear"]
    L += ["## Bear markets in the holdout", ""]
    px = daily["gspc_close"]
    for _, ep in episodes(bl).iterrows():
        peak = px.loc[ep["start"]]
        L += [f"Bear market {ep['start'].date()} → {ep['end'].date()} "
              f"({px.loc[ep['start']:ep['end']].min() / peak - 1:.0%}). First ≥ 50% \"bear within 21 days\" signal:", "",
              "| Method | Signal date | Days after peak | Market fall at signal |", "|---|---|---|---|"]
        for m in ("ma200_rule", "hmm", "logreg", "lightgbm"):
            g = preds[(preds.label == "bear") & (preds.horizon == 21) & (preds.method == m)].set_index("date").sort_index()
            on = g.loc[ep["start"]:ep["end"], "prob"]
            on = on[on >= 0.5]
            if len(on) == 0:
                L.append(f"| {NAMES[m]} | never | – | – |")
            else:
                d = on.index[0]
                L.append(f"| {NAMES[m]} | {d.date()} | {g.index.get_loc(d) - g.index.get_loc(ep['start'])} | "
                         f"{px.loc[d] / peak - 1:.0%} |")
        L.append("")
    L += ["![Holdout timeline](../docs/figures/holdout_timeline.png)", ""]
    (REPORTS / "holdout_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("wrote reports/holdout_results.md/.csv and docs/figures/holdout_timeline.png")

    for metric in ("pr_auc", "onset_pr_auc"):
        piv = metrics.pivot_table(index="method", columns=["label", "horizon"], values=metric).reindex(order)
        print(f"\n{metric} (holdout):\n" + piv.round(3).to_string())
    print("\nbootstrap (model - best baseline):")
    for r in boot_rows:
        print(f"  {r[0]} h{r[1]} {r[2]} vs {r[3]}: {r[4]:+.3f} [{r[5]:+.3f}, {r[6]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
