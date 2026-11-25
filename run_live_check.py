"""Live check: score the models on trading days that no model or report has ever seen.

Every result in the repo (validation and the one-time 2022+ holdout) used data up to the
2026-10-23 snapshot. This script trains on data known at that cut-off only, exactly like one
walk-forward fold, and scores the forecasts made from the cut-off onward against what happened.

  * Regime models (FEATURES_MAIN, all baselines and models, all 4 labels, h = 5 and 21):
    trained on rows up to WALK_FORWARD_GAP days before the first live day; bear labels are
    recomputed from prices known at the cut-off (as in `run_walk_forward`).
  * Range models (XGBoost, CatBoost, LSTM, baselines): trained on rows whose min/max targets
    were known before the first live day (gap + h, as in `run_range_models.run_validation`).

The live period starts with the cut-off day itself: its forecast is made at that close.
Nothing is tuned here and no existing output is changed.

Usage (from the repo root, after `python process_data.py` on refreshed data):
    python run_live_check.py
    python run_live_check.py --cutoff 2026-10-23 --no-lstm
Outputs: reports/live_check.md, data/processed/live_predictions.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from run_range_models import TARGETS, fit_predict_block  # noqa: E402
from sp500.config import (  # noqa: E402
    DATA_PROCESSED, FEATURES_MAIN, HORIZONS, MODEL_START, REPORTS, SVM_TRAIN_EVERY, WALK_FORWARD_GAP,
)
from sp500.features import bear_price_series  # noqa: E402
from sp500.labels import episodes  # noqa: E402
from sp500.models import (  # noqa: E402
    CURRENT_STATE, MODELS, fit_hmm, fold_bear_target, hmm_filtered_probs, make_model,
)
from sp500.range_data import RANGE_HORIZONS, build_range_table, feature_columns  # noqa: E402
from sp500.range_models import MODEL_NAMES, MODELS as RANGE_MODELS  # noqa: E402

DEFAULT_CUTOFF = "2026-10-23"  # last trading day of the snapshot all earlier results used
LABELS = ("stressed", "stressed_fixed", "bear", "bear_merged")
NAMES = {"persistence": "Baseline: no change", "ma200_rule": "Baseline: price < 200-day MA",
         "hmm": "Baseline: 2-state HMM", "logreg": "Logistic regression", "lightgbm": "LightGBM",
         "svm": "SVM (RBF)"}
ORDER = ["persistence", "ma200_rule", "hmm", "logreg", "lightgbm", "svm"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- regime models
def regime_predictions(table: pd.DataFrame, bear_price: pd.Series, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One walk-forward fold: train before the live period, predict every live day."""
    data = table.loc[MODEL_START:]
    data = data[data[FEATURES_MAIN].notna().all(axis=1)]
    test_start = int(data.index.searchsorted(cutoff))           # first live day = the cut-off day
    known_until = data.index[test_start - 1]                      # last close before it
    train = data.iloc[: test_start - WALK_FORWARD_GAP]
    test = data.iloc[test_start:]

    hmm = fit_hmm(train["ret_1d"].to_numpy())
    filt, trans = hmm_filtered_probs(hmm, data["ret_1d"].to_numpy())
    filt_test = filt[test_start:]

    rows = []
    for label in LABELS:
        for h in HORIZONS:
            if label.startswith("bear"):
                y_train = fold_bear_target(bear_price, label, h, known_until).reindex(train.index)
            else:
                y_train = train[f"target_{label}_h{h}"]
            target_date_ok = pd.Series(data.index, index=data.index).shift(-h).reindex(train.index) <= known_until
            ok = y_train.notna() & target_date_ok
            X_tr, y_tr = train.loc[ok, FEATURES_MAIN], y_train[ok].astype(int)

            preds = {
                "persistence": test[CURRENT_STATE[label]].astype(float).to_numpy(),
                "ma200_rule": (test["dist_ma200"] < 0).astype(float).to_numpy()
                if label.startswith("bear") else np.full(len(test), np.nan),
                "hmm": (filt_test @ np.linalg.matrix_power(trans, h))[:, 1],
            }
            for name in MODELS:
                Xf, yf = (X_tr.iloc[::SVM_TRAIN_EVERY], y_tr.iloc[::SVM_TRAIN_EVERY]) if name == "svm" else (X_tr, y_tr)
                model = make_model(name)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(Xf, yf)
                    preds[name] = model.predict_proba(test[FEATURES_MAIN])[:, 1]
            for method, p in preds.items():
                rows.append(pd.DataFrame({
                    "date": test.index, "label": label, "horizon": h, "method": method, "prob": p,
                    "truth": test[f"target_{label}_h{h}"].to_numpy(), "current": test[label].to_numpy(),
                }))
            log(f"  regime {label:<14} h={h:<2} trained on {len(y_tr):,} rows up to {known_until.date()}")
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- range models
def range_predictions(daily: pd.DataFrame, cutoff: pd.Timestamp, models) -> pd.DataFrame:
    table = build_range_table(daily)
    feats = feature_columns(table)
    test_start = int(table.index.searchsorted(cutoff))
    frames = []
    for h in RANGE_HORIZONS:
        sigma = table[f"sigma_{h}"].to_numpy()
        train_pos = np.arange(max(test_start - WALK_FORWARD_GAP - h, 0))
        train_pos = train_pos[np.isfinite(sigma[train_pos])]
        test_pos = np.arange(test_start, len(table))
        test_pos = test_pos[np.isfinite(sigma[test_pos])]
        t1 = time.time()
        preds, _ = fit_predict_block(table, feats, h, train_pos, test_pos, models)
        for model, d in preds.items():
            for t in TARGETS:
                q = d[t]
                frames.append(pd.DataFrame({
                    "date": table.index[test_pos], "model": model, "h": h, "target": t,
                    "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2],
                    "actual": table[f"{t}_{h}"].to_numpy()[test_pos],
                    "close": table["close"].to_numpy()[test_pos],
                }))
        log(f"  range h={h:<2} trained on {len(train_pos):,} rows, {len(test_pos)} live days ({time.time()-t1:.0f}s)")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- report helpers
def first_signal(g: pd.DataFrame, start: pd.Timestamp, threshold: float = 0.5) -> str:
    """Trading days between the first >= threshold signal and the episode start (- = in advance)."""
    g = g.sort_values("date").reset_index(drop=True)
    hit = g.index[g["prob"] >= threshold]
    if len(hit) == 0:
        return "never"
    s = g.index[g["date"] >= start][0] if (g["date"] >= start).any() else len(g)
    off = int(hit[0]) - int(s)
    return f"{off:+d} ({g.loc[hit[0], 'date'].date()})"


def fmt_prob(p: float) -> str:
    return "–" if pd.isna(p) else f"{p:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="first live day (its close is the first forecast)")
    ap.add_argument("--no-lstm", action="store_true")
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    t0 = time.time()

    table = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    daily = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    last = table.index.max()
    live_days = table.loc[cutoff:].index
    if len(live_days) < 2:
        raise SystemExit(f"No live days after {cutoff.date()}: refresh the data first (last day {last.date()}).")
    log(f"LIVE CHECK: forecasts from {cutoff.date()} to {last.date()} ({len(live_days)} days)")

    reg = regime_predictions(table, bear_price_series(daily), cutoff)
    models = [m for m in RANGE_MODELS if not (args.no_lstm and m == "lstm")]
    rng = range_predictions(daily, cutoff, models)
    pd.concat([reg.assign(task="regime"), rng.assign(task="range")], ignore_index=True) \
        .to_parquet(DATA_PROCESSED / "live_predictions.parquet")

    px = daily["gspc_close"]
    L = ["# Live check", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `run_live_check.py` ({(time.time()-t0)/60:.1f} min). "
         f"Models trained only on data known before **{cutoff.date()}** (the last day of the snapshot every "
         f"other result used); forecasts made at each close from {cutoff.date()} to {last.date()} "
         f"({len(live_days)} trading days). Same features, settings and gaps as validation; nothing tuned.", "",
         f"S&P 500: {px.loc[cutoff]:,.0f} at the cut-off → {px.loc[last]:,.0f} on {last.date()} "
         f"({px.loc[last] / px.loc[cutoff] - 1:+.1%}); lowest close {px.loc[cutoff:].min():,.0f} "
         f"on {px.loc[cutoff:].idxmin().date()}.", "",
         "With this few days, the numbers are a sanity check of the reported results, not a new estimate of "
         "accuracy: one stress onset is a single event.", ""]

    # ---- what happened
    L += ["## What happened", ""]
    for label in LABELS:
        lab = table.loc[cutoff:, label]
        eps = episodes(lab)
        if len(eps):
            parts = [f"{e['start'].date()} → {e['end'].date()} ({e['days']} days)" for _, e in eps.iterrows()]
            L.append(f"- `{label}`: " + "; ".join(parts) + ("" if lab.iloc[0] == 0 else " (already on at the cut-off)"))
        else:
            L.append(f"- `{label}`: not on at any time in the live period")
    L.append("")

    # ---- regime: forecast made at the cut-off
    L += ["## Regime forecasts made at the cut-off close", "",
          f"Probability, at the {cutoff.date()} close, that the regime is on within *h* days, and whether it was.", "",
          "| Target | Happened? | " + " | ".join(NAMES[m] for m in ORDER) + " |",
          "|---|---|" + "---|" * len(ORDER)]
    at_cut = reg[reg.date == cutoff]
    for label in LABELS:
        for h in HORIZONS:
            g = at_cut[(at_cut.label == label) & (at_cut.horizon == h)].set_index("method")
            truth = g["truth"].dropna()
            happened = "not known yet" if truth.empty else ("**yes**" if truth.iloc[0] == 1 else "no")
            cells = [fmt_prob(g.loc[m, "prob"]) if m in g.index else "–" for m in ORDER]
            L.append(f"| `{label}` in {h}d | {happened} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- regime: warning before each onset in the live period
    onsets = []
    for label in LABELS:
        for _, e in episodes(table.loc[cutoff:, label]).iterrows():
            if e["start"] > cutoff:
                onsets.append((label, e["start"]))
    if onsets:
        L += ["## Warning before each regime start in the live period", "",
              "First day the probability reached 0.5, in trading days relative to the start "
              "(negative = warned in advance; the forecast must come before the start to count as a warning).", "",
              "| Regime start | Horizon | " + " | ".join(NAMES[m] for m in ORDER if m != "ma200_rule") + " |",
              "|---|---|" + "---|" * (len(ORDER) - 1)]
        for label, start in onsets:
            for h in HORIZONS:
                cells = []
                for m in ORDER:
                    if m == "ma200_rule":
                        continue
                    g = reg[(reg.label == label) & (reg.horizon == h) & (reg.method == m)]
                    cells.append(first_signal(g, start))
                L.append(f"| `{label}` {start.date()} | {h}d | " + " | ".join(cells) + " |")
        L.append("")

    # ---- regime: daily probabilities, main stress label
    L += ["## Daily `stressed` probabilities", "",
          "| Date | Close | Stressed | Within 5d? | LogReg 5d | LightGBM 5d | HMM 5d | Within 21d? | LogReg 21d | LightGBM 21d | HMM 21d |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    s = reg[reg.label == "stressed"]
    piv = s.pivot_table(index="date", columns=["horizon", "method"], values="prob")
    tru = s[s.method == "logreg"].pivot_table(index="date", columns="horizon", values="truth", dropna=False)
    cur = s[(s.method == "logreg") & (s.horizon == HORIZONS[0])].set_index("date")["current"]
    yn = lambda v: "–" if pd.isna(v) else ("yes" if v == 1 else "no")  # noqa: E731
    for d in piv.index:
        L.append(f"| {d.date()} | {px.loc[d]:,.0f} | {yn(cur.get(d))} | {yn(tru.loc[d].get(5))} | "
                 f"{fmt_prob(piv.loc[d, (5, 'logreg')])} | {fmt_prob(piv.loc[d, (5, 'lightgbm')])} | {fmt_prob(piv.loc[d, (5, 'hmm')])} | "
                 f"{yn(tru.loc[d].get(21))} | {fmt_prob(piv.loc[d, (21, 'logreg')])} | {fmt_prob(piv.loc[d, (21, 'lightgbm')])} | "
                 f"{fmt_prob(piv.loc[d, (21, 'hmm')])} |")
    L += ["", "\"–\" in the *Within* columns: the answer needs days after the last date in the data.", ""]

    # ---- range: forecast made at the cut-off
    range_order = [m for m in list(RANGE_MODELS) + ["ensemble"] if m in set(rng.model)]
    L += ["## Min / max forecasts made at the cut-off close", "",
          f"10–90% band and median for the lowest and highest close-to-extreme move over the next *h* days, "
          f"in S&P 500 points (close {px.loc[cutoff]:,.0f}).", "",
          "| Horizon | Target | Actual | " + " | ".join(MODEL_NAMES.get(m, m) for m in range_order) + " |",
          "|---|---|---|" + "---|" * len(range_order)]
    rc = rng[rng.date == cutoff]
    for h in RANGE_HORIZONS:
        for t in TARGETS:
            g = rc[(rc.h == h) & (rc.target == t)].set_index("model").reindex(range_order)
            if g.empty:
                continue
            c = g["close"].iloc[0]
            act = g["actual"].iloc[0]
            act_s = "not known yet" if pd.isna(act) else f"{c * np.exp(act):,.0f}"
            cells = []
            for m, r in g.iterrows():
                lo, mid, hi = (c * np.exp(r[q]) for q in ("q10", "q50", "q90"))
                inside = "" if pd.isna(act) else (" ✓" if r.q10 <= act <= r.q90 else " ✗")
                cells.append(f"{lo:,.0f} / {mid:,.0f} / {hi:,.0f}{inside}")
            name = "max high" if t == "up" else "min low"
            L.append(f"| {h}d | {name} | {act_s} | " + " | ".join(cells) + " |")
    L += ["", "Each cell: 10% / 50% / 90% quantile. ✓ = actual inside the 10–90% band, ✗ = outside.", ""]

    # ---- range: all live days with a known outcome
    known = rng.dropna(subset=["actual"])
    if len(known):
        L += ["## Min / max: every live day with a known outcome", "",
              "Coverage should be near 80% for a well-calibrated 10–90% band (with this few, overlapping days, "
              "expect a lot of noise). Error = mean absolute error of the median, in % of the close.", "",
              "| Horizon | Target | Days | Model | Inside band | Median error | Validation coverage |", "|---|---|---|---|---|---|---|"]
        val = pd.read_csv(REPORTS / "range_results.csv") if (REPORTS / "range_results.csv").exists() else None
        for (h, t), g in sorted(known.groupby(["h", "target"]), key=lambda kv: (kv[0][0], kv[0][1] != "up")):
            for m in range_order:
                gm = g[g.model == m]
                cov = np.mean((gm.actual >= gm.q10) & (gm.actual <= gm.q90))
                err = np.mean(np.abs(gm.q50 - gm.actual)) * 100
                vcov = "–"
                if val is not None:
                    v = val[(val.split == "walk_forward") & (val.model == m) & (val.h == h) & (val.target == t)]
                    vcov = f"{v.coverage_80.mean():.0%}" if len(v) else "–"
                L.append(f"| {h}d | {'max high' if t == 'up' else 'min low'} | {len(gm)} | {MODEL_NAMES.get(m, m)} | "
                         f"{cov:.0%} | {err:.2f}% | {vcov} |")
        L.append("")

    (REPORTS / "live_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"wrote reports/live_check.md and data/processed/live_predictions.parquet ({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
