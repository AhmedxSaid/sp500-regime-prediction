"""Predict the MIN and MAX S&P 500 price over the next 5 and 21 trading days.

Models: XGBoost, CatBoost and an LSTM (quantile regression: 10%, 50%, 90%), compared with two
baselines (historical quantiles, volatility band). Run `python process_data.py` first.

Validation follows the rest of the project:
  * walk-forward 2000-2021: retrain every `--block` years on everything before the block,
    with a 21-day gap so no training target overlaps the test period;
  * holdout 2022-01-01 onward: one model trained on data before 2022 (new target, so this
    holdout is fresh for the range task);
  * final models trained on all data -> the forecast shown in the dashboard.

Usage:
    python run_range_models.py              # everything (~15-25 min on 2 cores)
    python run_range_models.py --fast       # 4-year blocks, quicker check
    python run_range_models.py --no-lstm

Outputs:
    data/processed/range_table.parquet          features + targets
    data/processed/range_predictions.parquet    every out-of-sample prediction
    data/processed/range_forecast.json          latest forecast (dashboard)
    reports/range_results.csv / .md             metrics
    reports/range_feature_importance.csv
    reports/range_pattern_stats.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sp500.config import DATA_PROCESSED, HOLDOUT_START, REPORTS, WALK_FORWARD_GAP  # noqa: E402
from sp500.patterns import PATTERNS, pattern_stats  # noqa: E402
from sp500.range_data import RANGE_HORIZONS, build_range_table, feature_columns  # noqa: E402
from sp500.range_models import (  # noqa: E402
    ML_MODELS, MODEL_NAMES, MODELS, QUANTILES, LSTMQuantile, pinball,
)

FIRST_TEST_YEAR = 2000
TARGETS = ("up", "down")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def make_blocks(index: pd.DatetimeIndex, block: int):
    years = index.year
    hold = pd.Timestamp(HOLDOUT_START)
    for y in range(FIRST_TEST_YEAR, hold.year, block):
        test = np.flatnonzero((years >= y) & (years < min(y + block, hold.year)))
        yield f"wf_{y}", "walk_forward", test
    yield "holdout", "holdout", np.flatnonzero(index >= hold)


def fit_predict_block(table, feats, h, train_pos, test_pos, models, keep=False):
    """Train every model for horizon h on train_pos and predict test_pos.

    Returns {model: {target: (n_test, n_q) raw log-return quantiles}} and fitted models.
    """
    X = table[feats]
    sigma = table[f"sigma_{h}"].to_numpy()
    out, fitted = {}, {}
    for name in models:
        out[name] = {}
        if name == "lstm":
            Y = np.column_stack([table[f"{t}_{h}"].to_numpy() / sigma for t in TARGETS])
            m = LSTMQuantile().fit_frame(X, Y, train_pos)
            q = m.predict_frame(X, test_pos)
            for k, t in enumerate(TARGETS):
                out[name][t] = q[:, k, :] * sigma[test_pos, None]
            fitted[(name, "both")] = m if keep else None
            continue
        for t in TARGETS:
            m = MODELS[name]()
            y_raw = table[f"{t}_{h}"].to_numpy()
            y = y_raw / sigma if m.scaled else y_raw
            ok = train_pos[np.isfinite(y[train_pos])]
            m.fit(X.iloc[ok], y[ok])
            q = m.predict(X.iloc[test_pos])
            out[name][t] = q * sigma[test_pos, None] if m.scaled else q
            if keep:
                fitted[(name, t)] = m
    ml = [m for m in ML_MODELS if m in out]
    if len(ml) >= 2:
        out["ensemble"] = {t: np.mean([out[m][t] for m in ml], axis=0) for t in TARGETS}
    return out, fitted


def to_rows(table, h, test_pos, preds, block, split):
    idx = table.index[test_pos]
    frames = []
    for model, d in preds.items():
        for t, q in d.items():
            frames.append(pd.DataFrame({
                "date": idx, "block": block, "split": split, "model": model, "h": h, "target": t,
                "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2],
                "actual": table[f"{t}_{h}"].to_numpy()[test_pos],
                "close": table["close"].to_numpy()[test_pos],
            }))
    return pd.concat(frames, ignore_index=True)


def run_validation(table, feats, models, block: int = 2, log=log) -> pd.DataFrame:
    """Walk-forward 2000-2021 (`block`-year blocks) + holdout 2022+ for the given models.

    Returns one row per (date, model, horizon, target) with the 10/50/90% quantiles and the actual
    value (log move vs. today's close). Used by `main` and by the per-model notebooks.
    """
    all_rows = []
    for block_name, split, test_pos in make_blocks(table.index, block):
        test_start = test_pos[0]
        for h in RANGE_HORIZONS:
            # training rows: targets fully known before the test period starts (+ gap)
            train_pos = np.arange(max(test_start - WALK_FORWARD_GAP - h, 0))
            train_pos = train_pos[np.isfinite(table[f"sigma_{h}"].to_numpy()[train_pos])]
            tp = test_pos[np.isfinite(table[f"sigma_{h}"].to_numpy()[test_pos])]
            t1 = time.time()
            preds, _ = fit_predict_block(table, feats, h, train_pos, tp, models)
            all_rows.append(to_rows(table, h, tp, preds, block_name, split))
            log(f"  {block_name:<9} h={h:<2} train {len(train_pos):>5} test {len(tp):>4}  ({time.time()-t1:.0f}s)")
    return pd.concat(all_rows, ignore_index=True)


def add_ensemble(pred: pd.DataFrame, members=ML_MODELS) -> pd.DataFrame:
    """Average the quantiles of the ML models (rows aligned on date / horizon / target)."""
    keys = ["date", "block", "split", "h", "target", "actual", "close"]
    sub = pred[pred["model"].isin(members)]
    if sub["model"].nunique() < 2:
        return pred
    ens = sub.groupby(keys, as_index=False)[["q10", "q50", "q90"]].mean().assign(model="ensemble")
    return pd.concat([pred[pred["model"] != "ensemble"], ens], ignore_index=True)


def metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, model, h, t), g in pred.dropna(subset=["actual"]).groupby(["split", "model", "h", "target"]):
        y = g["actual"].to_numpy()
        q = g[["q10", "q50", "q90"]].to_numpy()
        rows.append({
            "split": split, "model": model, "h": h, "target": t, "n": len(g),
            "mae_pct": np.mean(np.abs(q[:, 1] - y)) * 100,
            "pinball_x100": pinball(y, q) * 100,
            "coverage_80": np.mean((y >= q[:, 0]) & (y <= q[:, 2])),
            "width_pct": np.mean(q[:, 2] - q[:, 0]) * 100,
            "corr": np.corrcoef(q[:, 1], y)[0, 1] if np.std(q[:, 1]) > 0 else np.nan,
        })
    m = pd.DataFrame(rows)
    ref = m[m["model"] == "vol_band"].set_index(["split", "h", "target"])["mae_pct"]
    m["mae_skill_vs_volband"] = 1 - m["mae_pct"] / m.set_index(["split", "h", "target"]).index.map(ref)
    return m


def write_report(m: pd.DataFrame, path: Path, runtime: float) -> None:
    lines = [
        "# Min / max price prediction: results", "",
        f"Generated on {datetime.now():%Y-%m-%d %H:%M}"
        + (f" by `run_range_models.py` ({runtime/60:.1f} min)." if runtime else " by `notebooks/15_range_model_comparison.ipynb`."), "",
        "Target: highest high (`up`) and lowest low (`down`) over the next *h* trading days, as a log",
        "move from today's close. Models give the 10%, 50% and 90% quantiles.", "",
        "* **MAE %**: mean absolute error of the median forecast, in % of price (lower is better).",
        "* **Pinball**: average quantile loss x100 (lower is better; scores the whole band).",
        "* **Coverage 80%**: share of days where the actual value fell inside the 10-90% band (target 0.80).",
        "* **Skill**: 1 - MAE / MAE of the volatility-band baseline (> 0 beats it).",
        "* **Corr**: correlation of the median forecast with the actual value.", "",
    ]
    for split, title in (("walk_forward", "Walk-forward 2000-2021"), ("holdout", "Holdout 2022+")):
        lines += [f"## {title}", ""]
        for h in sorted(m["h"].unique()):
            for t in TARGETS:
                g = m[(m["split"] == split) & (m["h"] == h) & (m["target"] == t)]
                if g.empty:
                    continue
                label = "max high" if t == "up" else "min low"
                lines += [f"### {label}, next {h} days (n = {int(g['n'].iloc[0])})", "",
                          "| Model | MAE % | Pinball | Coverage 80% | Width % | Corr | Skill |",
                          "|---|---|---|---|---|---|---|"]
                best = g["pinball_x100"].min()
                for _, r in g.sort_values("pinball_x100").iterrows():
                    name = MODEL_NAMES.get(r["model"], r["model"])
                    pb = f"**{r['pinball_x100']:.3f}**" if r["pinball_x100"] == best else f"{r['pinball_x100']:.3f}"
                    lines.append(f"| {name} | {r['mae_pct']:.3f} | {pb} | {r['coverage_80']:.2f} | "
                                 f"{r['width_pct']:.2f} | {r['corr']:.2f} | {r['mae_skill_vs_volband']:+.3f} |")
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def final_forecast(table, feats, models):
    """Train every model on all rows with a known target and forecast from the last day.

    Returns (forecast dict for the dashboard, feature-importance frame of the tree models).
    """
    forecast = {"as_of": str(table.index[-1].date()), "close": float(table["close"].iloc[-1]),
                "quantiles": list(QUANTILES), "horizons": {}}
    imps = []
    last = np.array([len(table) - 1])
    for h in RANGE_HORIZONS:
        sig = table[f"sigma_{h}"].to_numpy()
        train_pos = np.flatnonzero(np.isfinite(sig) & np.isfinite(table[f"up_{h}"].to_numpy()))
        preds, fitted = fit_predict_block(table, feats, h, train_pos, last, models, keep=True)
        forecast["horizons"][str(h)] = {
            name: {t: [float(v) for v in d[t][0]] for t in TARGETS} for name, d in preds.items()
        }
        forecast["horizons"][str(h)]["sigma"] = float(sig[-1])
        for (name, t), fm in fitted.items():
            if hasattr(fm, "importance"):
                imp = fm.importance(feats)
                imps.append(pd.DataFrame({"feature": feats, "importance": (imp / imp.sum()).values,
                                          "model": name, "h": h, "target": t}))
    return forecast, (pd.concat(imps, ignore_index=True) if imps else pd.DataFrame())


def pattern_stats_table(table) -> pd.DataFrame:
    """Market behaviour after each daily candlestick pattern, for both horizons."""
    ps = []
    flags = table[[f"pat_{k}" for k in PATTERNS]].rename(columns=lambda c: c[4:])
    for h in RANGE_HORIZONS:
        s = pattern_stats(flags, table[f"up_{h}"], table[f"down_{h}"], table[f"ret_{h}"])
        s["h"] = h
        ps.append(s)
    return pd.concat(ps, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=2, help="years per walk-forward block")
    ap.add_argument("--fast", action="store_true", help="4-year blocks")
    ap.add_argument("--no-lstm", action="store_true")
    args = ap.parse_args()
    block = 4 if args.fast else args.block
    models = [m for m in MODELS if not (args.no_lstm and m == "lstm")]
    t0 = time.time()

    log("Building range table ...")
    df = pd.read_parquet(DATA_PROCESSED / "sp500_daily.parquet")
    table = build_range_table(df)
    table.to_parquet(DATA_PROCESSED / "range_table.parquet")
    feats = feature_columns(table)
    log(f"  {len(table)} rows, {len(feats)} features, {table.index.min().date()} -> {table.index.max().date()}")

    pred = run_validation(table, feats, models, block)
    pred.to_parquet(DATA_PROCESSED / "range_predictions.parquet")
    m = metrics(pred)
    m.to_csv(REPORTS / "range_results.csv", index=False)

    log("Final models on all data -> forecast ...")
    forecast, imps = final_forecast(table, feats, models)
    (DATA_PROCESSED / "range_forecast.json").write_text(json.dumps(forecast, indent=2))
    imps.to_csv(REPORTS / "range_feature_importance.csv", index=False)
    pattern_stats_table(table).to_csv(REPORTS / "range_pattern_stats.csv", index=False)

    runtime = time.time() - t0
    write_report(m, REPORTS / "range_results.md", runtime)
    log(f"Done in {runtime/60:.1f} min. Summary (pinball x100, lower is better):")
    print(m.pivot_table(index=["split", "h", "target"], columns="model", values="pinball_x100").round(3).to_string())


if __name__ == "__main__":
    main()
