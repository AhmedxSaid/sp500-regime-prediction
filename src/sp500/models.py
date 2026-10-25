"""Baselines and models for regime prediction, evaluated walk-forward.

Every prediction for test year N uses only information available before year N starts:
  - models are trained on rows up to WALK_FORWARD_GAP trading days before the test year,
  - bear labels (which need hindsight to date the peak) are RECOMPUTED for each fold from
    prices known at the fold's cut-off, so training never sees how a later bear was dated,
  - the HMM baseline uses a forward (filtering) pass only, never smoothing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from .config import SVM_TRAIN_EVERY, TRADING_DAYS
from .labels import bear_market_labels, merge_bear_episodes
from .validation import walk_forward_splits

BASELINES = ("persistence", "ma200_rule", "hmm")
MODELS = ("logreg", "lightgbm", "svm")

# Causal "current state" used by the persistence baseline (the hindsight `bear` label at day t
# is not known at day t, so bear persistence uses "20% below the old peak and not recovered").
CURRENT_STATE = {
    "bear": "below_peak_regime",
    "bear_merged": "below_peak_regime",
    "stressed": "stressed",
    "stressed_fixed": "stressed_fixed",
}


# --------------------------------------------------------------------------- models
def make_model(name: str):
    """Fixed hyperparameters (no tuning, to avoid fitting the test years)."""
    if name == "logreg":
        return make_pipeline(RobustScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=200,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=0, verbose=-1,
        )
    if name == "svm":
        # RBF SVM (as in the literature review), probabilities via Platt scaling fitted on
        # time-ordered splits of the training data
        svc = make_pipeline(RobustScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
        return CalibratedClassifierCV(svc, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))
    raise ValueError(name)


# --------------------------------------------------------------------------- HMM baseline
def fit_hmm(returns: np.ndarray):
    from hmmlearn.hmm import GaussianHMM

    m = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(returns.reshape(-1, 1))
    return m


def hmm_filtered_probs(model, returns: np.ndarray) -> np.ndarray:
    """Causal forward filter: P(state_t | returns up to t). Columns ordered [calm, volatile]."""
    means = model.means_.ravel()
    sd = np.sqrt(np.asarray(model.covars_).reshape(len(means), -1)[:, 0])
    lik = norm.pdf(returns[:, None], means[None, :], sd[None, :]) + 1e-300
    a = model.transmat_
    p = model.startprob_.copy()
    out = np.empty_like(lik)
    for t in range(len(returns)):
        if t:
            p = p @ a
        p = p * lik[t]
        p /= p.sum()
        out[t] = p
    order = np.argsort(sd)  # calm first
    return out[:, order], a[np.ix_(order, order)]


# --------------------------------------------------------------------------- labels per fold
def fold_bear_target(bear_price: pd.Series, label: str, horizon: int, cutoff: pd.Timestamp) -> pd.Series:
    """Bear target as it could have been computed with prices up to `cutoff` only."""
    known = bear_price.loc[:cutoff]
    bear = bear_market_labels(known, method="peak_to_trough")
    lab = merge_bear_episodes(bear, known) if label == "bear_merged" else bear
    target = lab.shift(-horizon)
    return target


# --------------------------------------------------------------------------- walk-forward
def run_walk_forward(
    table: pd.DataFrame,
    bear_price: pd.Series,
    features: list[str],
    labels: tuple[str, ...],
    horizons: tuple[int, ...],
    first_test_year: int,
    gap: int,
    start: str,
    holdout_start: str,
    models: tuple[str, ...] = MODELS,
    log=print,
    allow_missing: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return long-format predictions: date, label, horizon, method, prob, truth, current.

    `allow_missing`: features that may be NaN (e.g. VIX before 1990). Rows are only required to
    have the other features. Use with LightGBM only, which handles missing values natively.
    """
    end = pd.Timestamp(holdout_start) - pd.Timedelta(days=1)
    data = table.loc[start:end]
    required = [f for f in features if f not in allow_missing]
    data = data[data[required].notna().all(axis=1)]
    rows = []

    splits = list(walk_forward_splits(data.index, first_test_year, gap, holdout_start=holdout_start))
    for k, (year, train_idx, test_idx) in enumerate(splits, 1):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        cutoff = data.index[test_idx[0] - 1]  # last day known when the test year starts

        # HMM fitted once per fold on training returns; filtered over the contiguous history
        hist = data.iloc[: test_idx[-1] + 1]
        hmm = fit_hmm(train["ret_1d"].to_numpy())
        filt, trans = hmm_filtered_probs(hmm, hist["ret_1d"].to_numpy())
        filt_test = filt[test_idx[0]: test_idx[-1] + 1]

        for label in labels:
            for h in horizons:
                truth = test[f"target_{label}_h{h}"]
                current = test[label]
                if label.startswith("bear"):
                    y_all = fold_bear_target(bear_price, label, h, cutoff)
                    y_train = y_all.reindex(train.index)
                else:
                    y_train = train[f"target_{label}_h{h}"]
                # a training row's target must be known at the cut-off
                target_date_ok = pd.Series(data.index, index=data.index).shift(-h).reindex(train.index) <= cutoff
                ok = y_train.notna() & target_date_ok
                X_tr, y_tr = train.loc[ok, features], y_train[ok].astype(int)

                preds = {
                    "persistence": test[CURRENT_STATE[label]].astype(float).to_numpy(),
                    "ma200_rule": (test["dist_ma200"] < 0).astype(float).to_numpy()
                    if label.startswith("bear") else np.full(len(test), np.nan),
                    "hmm": (filt_test @ np.linalg.matrix_power(trans, h))[:, 1],
                }
                for name in models:
                    if y_tr.nunique() < 2:
                        preds[name] = np.full(len(test), y_tr.mean() if len(y_tr) else np.nan)
                        continue
                    Xf, yf = (X_tr.iloc[::SVM_TRAIN_EVERY], y_tr.iloc[::SVM_TRAIN_EVERY]) \
                        if name == "svm" else (X_tr, y_tr)
                    model = make_model(name)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            model.fit(Xf, yf)
                        except ValueError:
                            # e.g. an early calibration split with only one class: fall back to
                            # the SVM's built-in Platt scaling
                            model = make_pipeline(RobustScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                                      probability=True, random_state=0))
                            model.fit(Xf, yf)
                        preds[name] = model.predict_proba(test[features])[:, 1]

                for method, p in preds.items():
                    rows.append(pd.DataFrame({
                        "date": test.index, "label": label, "horizon": h, "method": method,
                        "prob": p, "truth": truth.to_numpy(), "current": current.to_numpy(),
                        "test_year": year,
                    }))
        log(f"      fold {k}/{len(splits)}: test {year} (train {len(train):,} rows up to {cutoff.date()})")
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- metrics
SCORE_KEYS = ["n_days", "base_rate", "pr_auc", "roc_auc", "brier", "onset_base_rate", "onset_pr_auc",
              "signal_flips_per_year", "true_flips_per_year"]


def score(group: pd.DataFrame) -> pd.Series:
    """Metrics for one (label, horizon, method). Always returns the same keys (NaN if n/a)."""
    out = dict.fromkeys(SCORE_KEYS, np.nan)
    g = group.dropna(subset=["prob", "truth"])
    if g.empty:  # e.g. the 200-day-MA rule, which only exists for bear targets
        return pd.Series(out)
    y, p = g["truth"].astype(int), g["prob"].clip(0, 1)
    years = g["date"].nunique() / TRADING_DAYS
    out.update({"n_days": len(g), "base_rate": y.mean()})
    if y.nunique() == 2:
        out["pr_auc"] = average_precision_score(y, p)
        out["roc_auc"] = roc_auc_score(y, p)
        out["brier"] = brier_score_loss(y, p)
    # onset: days not yet in the regime -> will it start within h days?
    o = g[g["current"] == 0]
    yo = o["truth"].astype(int)
    out["onset_base_rate"] = yo.mean()
    out["onset_pr_auc"] = average_precision_score(yo, o["prob"].clip(0, 1)) if yo.nunique() == 2 else np.nan
    # stability: switches per year of the 0.5-thresholded signal vs. of the truth
    sig = (g.sort_values("date")["prob"] >= 0.5).astype(int)
    out["signal_flips_per_year"] = sig.diff().abs().sum() / years
    out["true_flips_per_year"] = g.sort_values("date")["truth"].diff().abs().sum() / years
    return pd.Series(out)[SCORE_KEYS]


def lead_times(preds: pd.DataFrame, threshold: float = 0.5, window: int = 63) -> pd.DataFrame:
    """For each regime start in the test years: trading days of warning before it began.

    Warning = the number of consecutive days, ending the day before the start, on which the
    predicted probability was >= threshold (0 = no warning; capped at `window`).
    """
    rows = []
    for (label, h, method), g in preds.groupby(["label", "horizon", "method"]):
        g = g.sort_values("date").reset_index(drop=True)
        cur = g["current"].fillna(0).astype(int)
        starts = g.index[(cur == 1) & (cur.shift(1) == 0)]
        for s in starts:
            before = g["prob"].iloc[max(0, s - window): s] >= threshold
            run = 0
            for v in before[::-1]:
                if not v:
                    break
                run += 1
            rows.append({"label": label, "horizon": h, "method": method,
                         "start": g.loc[s, "date"], "warning_days": run})
    return pd.DataFrame(rows)
