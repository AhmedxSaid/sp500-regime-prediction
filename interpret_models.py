"""Interpret the main models and break results down by crisis.

1. Logistic regression coefficients (fit on 1962-2021) + sign stability across the 42
   walk-forward folds.
2. LightGBM SHAP values (LightGBM's built-in `pred_contrib`), fit on 1962-2021.
3. Per-crisis results from the saved out-of-sample walk-forward predictions
   (run `python run_models.py` first).

Nothing here uses the 2022+ holdout: final fits stop at 2021-12-31, and bear labels are
recomputed from prices known up to that date.

Usage (from the repo root):
    python interpret_models.py
Outputs: reports/interpretation.md, docs/figures/interp_*.png
"""
from __future__ import annotations

import sys
import warnings
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
    DATA_PROCESSED, FEATURES_MAIN, FIRST_TEST_YEAR, HOLDOUT_START, MODEL_START, REPORTS, WALK_FORWARD_GAP,
)
from sp500.features import bear_price_series  # noqa: E402
from sp500.labels import episodes  # noqa: E402
from sp500.models import fold_bear_target, make_model  # noqa: E402
from sp500.validation import walk_forward_splits  # noqa: E402

FIG = ROOT / "docs" / "figures"
H = 21
TARGETS = ("stressed", "bear")
NICE = {
    "ret_1d": "1-day return", "mom_1m": "1-month momentum", "mom_3m": "3-month momentum",
    "mom_12m": "12-month momentum", "dist_ma50": "distance from 50-day MA",
    "dist_ma200": "distance from 200-day MA", "drawdown": "drawdown from peak",
    "below_peak_regime": "20% below old peak (flag)", "rv_21d": "21-day realized vol",
    "rv_63d": "63-day realized vol", "parkinson_21d": "21-day high-low vol",
    "rv_vs_5y_p80": "vol vs. stress threshold", "gap_open": "overnight gap",
}
STRESS_CRISES = {
    "1987 crash": ("1987-08-01", "1987-12-31"),
    "1990 Gulf War": ("1990-07-01", "1990-12-31"),
    "1997 Asian crisis": ("1997-10-01", "1997-12-31"),
    "1998 LTCM / Russia": ("1998-07-15", "1998-11-30"),
    "2000 dot-com top": ("2000-03-01", "2000-06-30"),
    "2001 9/11": ("2001-09-01", "2001-10-31"),
    "2002 accounting scandals": ("2002-06-01", "2002-10-31"),
    "2007 subprime start": ("2007-07-15", "2007-09-30"),
    "2008 Lehman": ("2008-09-01", "2008-12-31"),
    "2010 flash crash / euro": ("2010-04-20", "2010-07-31"),
    "2011 US downgrade": ("2011-07-20", "2011-10-31"),
    "2015 China devaluation": ("2015-08-01", "2015-10-31"),
    "2018 Volmageddon": ("2018-01-25", "2018-04-30"),
    "2018 Q4 sell-off": ("2018-10-01", "2018-12-31"),
    "2020 COVID": ("2020-02-15", "2020-05-31"),
}
# Correlated features share credit, so importance is also summed per group
FEATURE_GROUPS_INTERP = {
    "Volatility": ["rv_vs_5y_p80", "rv_21d", "rv_63d", "parkinson_21d"],
    "Trend / momentum": ["dist_ma50", "dist_ma200", "mom_1m", "mom_3m", "mom_12m"],
    "Drawdown": ["drawdown", "below_peak_regime"],
    "Daily moves": ["ret_1d", "gap_open"],
}
METHODS = {"persistence": "No change", "hmm": "HMM", "ma200_rule": "MA200 rule",
           "logreg": "Logistic", "lightgbm": "LightGBM"}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def training_set(table, bear_price, target, cutoff):
    data = table.loc[MODEL_START:cutoff]
    data = data[data[FEATURES_MAIN].notna().all(axis=1)]
    if target == "bear":
        y = fold_bear_target(bear_price, "bear", H, cutoff).reindex(data.index)
    else:
        y = data[f"target_{target}_h{H}"]
    target_date = pd.Series(data.index, index=data.index).shift(-H)
    ok = y.notna() & (target_date <= cutoff)
    return data.loc[ok, FEATURES_MAIN], y[ok].astype(int)


# --------------------------------------------------------------------------- 1. logistic regression
def logreg_coefficients(table, bear_price):
    cutoff = pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)
    out = {}
    for target in TARGETS:
        X, y = training_set(table, bear_price, target, cutoff)
        model = make_model("logreg").fit(X, y)
        coef = pd.Series(model[-1].coef_[0], index=FEATURES_MAIN)
        # stability: refit in every walk-forward fold
        data = table.loc[MODEL_START:cutoff]
        data = data[data[FEATURES_MAIN].notna().all(axis=1)]
        fold_coefs = {}
        for year, tr, te in walk_forward_splits(data.index, FIRST_TEST_YEAR, WALK_FORWARD_GAP,
                                                holdout_start=HOLDOUT_START):
            c = data.index[te[0] - 1]
            Xf, yf = training_set(table, bear_price, target, c)
            if yf.nunique() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_coefs[year] = make_model("logreg").fit(Xf, yf)[-1].coef_[0]
        folds = pd.DataFrame(fold_coefs, index=FEATURES_MAIN).T
        same_sign = (np.sign(folds) == np.sign(coef)).mean()
        out[target] = pd.DataFrame({
            "coef": coef, "odds_ratio": np.exp(coef), "fold_min": folds.min(), "fold_max": folds.max(),
            "same_sign_share": same_sign, "n_folds": len(folds),
        }).sort_values("coef", key=np.abs, ascending=False)
        log(f"  logistic regression {target}: fitted on {len(X):,} rows, {len(folds)} folds")
    return out


# --------------------------------------------------------------------------- 2. LightGBM SHAP
def lightgbm_shap(table, bear_price):
    cutoff = pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)
    out = {}
    for target in TARGETS:
        X, y = training_set(table, bear_price, target, cutoff)
        model = make_model("lightgbm").fit(X, y)
        contrib = model.booster_.predict(X, pred_contrib=True)  # last column = bias
        shap = pd.DataFrame(contrib[:, :-1], index=X.index, columns=FEATURES_MAIN)
        direction = {f: np.corrcoef(X[f].rank(), shap[f])[0, 1] for f in FEATURES_MAIN}
        out[target] = (shap, X, pd.DataFrame({
            "mean_abs_shap": shap.abs().mean(), "direction": pd.Series(direction),
        }).sort_values("mean_abs_shap", ascending=False))
        log(f"  LightGBM {target}: SHAP on {len(X):,} rows")
    return out


def shap_figure(shap_out, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, target in zip(axes, TARGETS):
        shap, X, summary = shap_out[target]
        top = summary.index[:10][::-1]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), size=min(4000, len(X)), replace=False)
        for i, f in enumerate(top):
            vals = X[f].iloc[idx]
            q = vals.rank(pct=True)
            ax.scatter(shap[f].iloc[idx], i + rng.uniform(-0.3, 0.3, len(idx)), c=q, cmap="coolwarm",
                       s=3, alpha=0.5, vmin=0, vmax=1)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([NICE[f] for f in top])
        ax.set_xlabel("SHAP value (effect on log-odds)")
        ax.set_title(f"LightGBM: {target} in {H} days")
    fig.subplots_adjust(wspace=0.55)
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("feature value (low → high, percentile)")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def coef_figure(coefs, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, target in zip(axes, TARGETS):
        c = coefs[target].sort_values("coef")
        colors = ["C3" if v > 0 else "C0" for v in c["coef"]]
        ax.barh([NICE[f] for f in c.index], c["coef"], color=colors)
        ax.hlines(range(len(c)), c["fold_min"], c["fold_max"], color="k", alpha=0.5, lw=1.5)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(f"Logistic regression: {target} in {H} days\n(bar = 1962–2021 fit; line = range over 42 folds)")
        ax.set_xlabel("coefficient per interquartile range of the feature")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- 3. per-crisis
def signal_offset(g: pd.DataFrame, start_pos: int, look: int = 42) -> str:
    """Trading days from the regime start to the start of the method's >= 50% signal.

    Negative = warned in advance. '≤ -42' = signal already on 42 days before. 'missed' = no
    signal within 42 days after the start.
    """
    p = g["prob"].to_numpy()
    lo, hi = max(0, start_pos - look), min(len(p), start_pos + look + 1)
    window = p[lo:hi] >= 0.5
    rel = np.arange(lo, hi) - start_pos
    if not window.any():
        return "missed"
    # the signal "episode" that covers or follows the start
    on = np.where(window)[0]
    after = on[rel[on] >= 0]
    k = after[0] if len(after) else on[-1]
    while k > 0 and window[k - 1]:
        k -= 1
    if k == 0 and lo == start_pos - look and window[0]:
        return f"≤ {-look}"
    return f"{rel[k]:+d}"


def stress_crisis_table(preds, table):
    rows = []
    for name, (a, b) in STRESS_CRISES.items():
        lab = table["stressed"]
        w = lab.loc[a:b]
        prev = lab.shift(1).loc[a:b]
        starts = w.index[(w == 1) & (prev == 0)]
        if len(starts) == 0:
            starts = w.index[w == 1][:1]
        if len(starts) == 0:
            continue
        start = starts[0]
        dd = table.loc[a:b, "drawdown"].min()
        row = {"crisis": name, "stress start": start.date(), "max drawdown": dd}
        for m in ("persistence", "hmm", "logreg", "lightgbm"):
            g = preds[(preds.label == "stressed") & (preds.horizon == H) & (preds.method == m)] \
                .sort_values("date").reset_index(drop=True)
            pos = g.index[g.date == start]
            if len(pos) == 0:
                row[m] = "n/a"
                continue
            row[m] = signal_offset(g, int(pos[0]))
            if m in ("logreg", "lightgbm"):
                before = g.iloc[max(0, pos[0] - 21): pos[0]]["prob"].mean()
                row[f"{m}_prob_before"] = before
        rows.append(row)
    return pd.DataFrame(rows)


def bear_crisis_table(preds, table, daily):
    rows = []
    bear = table["bear"].loc[f"{FIRST_TEST_YEAR}":f"{pd.Timestamp(HOLDOUT_START).year - 1}"]
    px = daily["gspc_close"]
    for _, ep in episodes(bear).iterrows():
        start, end = ep["start"], ep["end"]
        peak = px.loc[start]
        trough_dd = px.loc[start:end].min() / peak - 1
        row = {"bear start (peak)": start.date(), "end (trough)": end.date(), "peak→trough": trough_dd}
        for m in ("persistence", "ma200_rule", "hmm", "logreg", "lightgbm"):
            g = preds[(preds.label == "bear") & (preds.horizon == H) & (preds.method == m)] \
                .sort_values("date").set_index("date")
            seg = g.loc[start: start + pd.Timedelta(days=400), "prob"]
            on = seg[seg >= 0.5]
            # ignore a signal that was simply still on from before the peak
            if len(on) and on.index[0] == seg.index[0]:
                still = g.loc[:start, "prob"].iloc[-6:-1]
                if (still >= 0.5).all():
                    off = seg[seg < 0.5]
                    on = on[on.index > off.index[0]] if len(off) else on.iloc[:0]
                    row[m] = "on at peak"
                    row[f"{m}_dd"] = np.nan
                    continue
            if len(on) == 0:
                row[m] = "never"
                row[f"{m}_dd"] = np.nan
            else:
                d = on.index[0]
                row[m] = f"+{g.index.get_loc(d) - g.index.get_loc(start)}d"
                row[f"{m}_dd"] = px.loc[d] / peak - 1
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def main() -> int:
    table = pd.read_parquet(DATA_PROCESSED / "model_table.parquet")
    daily = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    preds = pd.read_parquet(DATA_PROCESSED / "predictions_main.parquet")
    bear_price = bear_price_series(daily)
    FIG.mkdir(parents=True, exist_ok=True)

    log("1/3 Logistic regression coefficients")
    coefs = logreg_coefficients(table, bear_price)
    coef_figure(coefs, FIG / "interp_logreg_coefficients.png")
    log("2/3 LightGBM SHAP")
    shap_out = lightgbm_shap(table, bear_price)
    shap_figure(shap_out, FIG / "interp_lightgbm_shap.png")
    log("3/3 Per-crisis results")
    stress_tab = stress_crisis_table(preds, table)
    bear_tab = bear_crisis_table(preds, table, daily)

    # ---------------- report
    L = ["# Model interpretation and per-crisis results", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `interpret_models.py`. Horizon: {H} trading days. "
         "Final fits use 1962–2021 (bear labels recomputed with prices up to 2021-12-31); per-crisis "
         "results use the out-of-sample walk-forward predictions from `run_models.py`. The 2022+ holdout "
         "is not used.", ""]
    L += ["## 1. Logistic regression coefficients", "",
          "Features are robust-scaled, so a coefficient is the change in log-odds when the feature moves "
          "by one interquartile range (IQR). Odds ratio = exp(coef). *Same sign* = share of the walk-forward "
          "refits (one per test year) with the same sign as the final fit.", "",
          "![Logistic regression coefficients](../docs/figures/interp_logreg_coefficients.png)", ""]
    for target in TARGETS:
        c = coefs[target]
        L += [f"### `{target}` in {H} days ({int(c.n_folds.iloc[0])} folds)", "",
              "| Feature | Coef | Odds ratio per IQR | Range over folds | Same sign |", "|---|---|---|---|---|"]
        for f, r in c.iterrows():
            L.append(f"| {NICE[f]} (`{f}`) | {r.coef:+.2f} | {r.odds_ratio:.2f} | "
                     f"{r.fold_min:+.2f} to {r.fold_max:+.2f} | {r.same_sign_share:.0%} |")
        L.append("")
    L += ["## 2. LightGBM SHAP", "",
          "Mean |SHAP| = average size of the feature's effect on the log-odds. *Direction* = rank "
          "correlation between the feature value and its SHAP value (+ means higher value → higher "
          "probability).", "", "![LightGBM SHAP](../docs/figures/interp_lightgbm_shap.png)", "",
          "**By feature group** (share of total mean |SHAP|; correlated features within a group share credit):", "",
          "| Group | " + " | ".join(f"`{t}` in {H} days" for t in TARGETS) + " |", "|---|---|---|"]
    for grp, feats in FEATURE_GROUPS_INTERP.items():
        cells = []
        for target in TARGETS:
            s = shap_out[target][2]["mean_abs_shap"]
            cells.append(f"{s[feats].sum() / s.sum():.0%}")
        L.append(f"| {grp} | " + " | ".join(cells) + " |")
    L.append("")
    for target in TARGETS:
        s = shap_out[target][2]
        L += [f"### `{target}` in {H} days", "", "| Rank | Feature | Mean \\|SHAP\\| | Share | Direction |",
              "|---|---|---|---|---|"]
        tot = s.mean_abs_shap.sum()
        for i, (f, r) in enumerate(s.iterrows(), 1):
            L.append(f"| {i} | {NICE[f]} (`{f}`) | {r.mean_abs_shap:.3f} | {r.mean_abs_shap / tot:.0%} | "
                     f"{r.direction:+.2f} |")
        L.append("")

    L += ["## 3. Per-crisis results (out-of-sample)", "",
          f"### Stress episodes ({H}-day predictions)", "",
          "Signal start relative to the first stressed day, in trading days: **negative = warned in "
          "advance**, +k = recognized k days after stress began, *missed* = no ≥ 50% signal within 42 days. "
          "*P before* = mean predicted probability over the 21 trading days before stress began.", "",
          "| Crisis | Stress start | Max drawdown | No change | HMM | Logistic | LightGBM | P before (Logistic / LightGBM) |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in stress_tab.iterrows():
        L.append(f"| {r.crisis} | {r['stress start']} | {r['max drawdown']:.0%} | {r.persistence} | {r.hmm} | "
                 f"{r.logreg} | {r.lightgbm} | {r.get('logreg_prob_before', np.nan):.2f} / "
                 f"{r.get('lightgbm_prob_before', np.nan):.2f} |")
    L += ["", f"### Bear markets ({H}-day predictions)", "",
          "When did each method first say ≥ 50% \"bear within 21 days\", counted from the market peak, and how "
          "far had the market already fallen by then? *on at peak* = the signal was still on from before.", "",
          "| Bear (peak → trough) | Fall | No change | MA200 rule | HMM | Logistic | LightGBM |",
          "|---|---|---|---|---|---|---|"]
    for _, r in bear_tab.iterrows():
        cells = []
        for m in ("persistence", "ma200_rule", "hmm", "logreg", "lightgbm"):
            dd = r.get(f"{m}_dd", np.nan)
            cells.append(f"{r[m]}" + (f" ({dd:.0%})" if pd.notna(dd) else ""))
        L.append(f"| {r['bear start (peak)']} → {r['end (trough)']} | {r['peak→trough']:.0%} | " + " | ".join(cells) + " |")
    (REPORTS / "interpretation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("wrote reports/interpretation.md, docs/figures/interp_logreg_coefficients.png, interp_lightgbm_shap.png")

    for target in TARGETS:
        print(f"\n== {target}: logistic coefficients\n" + coefs[target][["coef", "same_sign_share"]].round(2).to_string())
        print(f"== {target}: LightGBM SHAP\n" + shap_out[target][2].round(3).to_string())
    print("\n" + stress_tab.to_string())
    print("\n" + bear_tab.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
