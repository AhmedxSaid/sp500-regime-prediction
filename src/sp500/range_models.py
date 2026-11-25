"""Quantile models for the min / max price over the next h days.

Every model has the same interface:
    m = make_model(name, h)
    m.fit(X_train, y_train)      # y = target / sigma (volatility-scaled), one target at a time
    q = m.predict(X_test)        # array (n, len(QUANTILES)), sorted along axis 1

The LSTM is the exception: it needs the full time-ordered feature frame because it looks at the
last `SEQ_LEN` days, so it gets `fit_frame` / `predict_frame` instead (handled by `run_block`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)
SEED = 42


# --------------------------------------------------------------------------- baselines
class HistQuantile:
    """Baseline 1: unconditional quantiles of the RAW target in the training data."""
    scaled = False

    def fit(self, X, y):
        self.q = np.nanquantile(y, QUANTILES)
        return self

    def predict(self, X):
        return np.tile(self.q, (len(X), 1))


class VolBand:
    """Baseline 2: quantiles of target / sigma in training, times today's sigma.

    Equivalent to "today's volatility x a historical multiplier": a strong, standard benchmark.
    """
    scaled = True

    def fit(self, X, y):
        self.q = np.nanquantile(y, QUANTILES)
        return self

    def predict(self, X):
        return np.tile(self.q, (len(X), 1))


# --------------------------------------------------------------------------- boosting
class XGBQuantile:
    scaled = True

    def __init__(self):
        import xgboost as xgb
        self.model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES),
            n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist",
            n_jobs=2, random_state=SEED,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.sort(self.model.predict(X), axis=1)

    def importance(self, columns) -> pd.Series:
        g = self.model.get_booster().get_score(importance_type="gain")
        return pd.Series({c: g.get(c, 0.0) for c in columns})


class CatQuantile:
    scaled = True

    def __init__(self):
        from catboost import CatBoostRegressor
        alphas = ",".join(str(q) for q in QUANTILES)
        self.model = CatBoostRegressor(
            # depth 4 / lr 0.03 / l2 10 chosen on walk-forward 2000-2021 only (depth 5 overfit)
            loss_function=f"MultiQuantile:alpha={alphas}", iterations=400, learning_rate=0.03,
            depth=4, l2_leaf_reg=10.0, random_seed=SEED, verbose=0, thread_count=2,
            allow_writing_files=False,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.sort(np.asarray(self.model.predict(X)), axis=1)

    def importance(self, columns) -> pd.Series:
        return pd.Series(self.model.get_feature_importance(), index=columns)


# --------------------------------------------------------------------------- LSTM
SEQ_LEN = 30


class LSTMQuantile:
    """Small LSTM over the last SEQ_LEN days of (standardized) features.

    Predicts both targets (up and down) and all quantiles at once with the pinball loss.
    """
    scaled = True

    def __init__(self, hidden=48, epochs=25, batch=256, lr=2e-3, patience=4):
        self.hidden, self.epochs, self.batch, self.lr, self.patience = hidden, epochs, batch, lr, patience

    # sequences ending at each row position in `pos`
    @staticmethod
    def _sequences(Z: np.ndarray, pos: np.ndarray) -> np.ndarray:
        idx = pos[:, None] - np.arange(SEQ_LEN - 1, -1, -1)[None, :]
        return Z[np.clip(idx, 0, None)]

    def _net(self, n_feat, n_out):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(n_feat, self.hidden, batch_first=True)
                s.drop = nn.Dropout(0.2)
                s.head = nn.Linear(self.hidden, n_out)

            def forward(s, x):
                out, _ = s.lstm(x)
                return s.head(s.drop(out[:, -1]))
        return Net()

    def fit_frame(self, X: pd.DataFrame, Y: np.ndarray, train_pos: np.ndarray):
        """X: full feature frame (all dates), Y: (n_rows, n_targets) scaled targets,
        train_pos: integer positions of training rows (time ordered)."""
        import torch
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        tr = X.iloc[train_pos]
        self.mu = tr.mean()
        self.sd = tr.std().replace(0, 1)
        Z = ((X - self.mu) / self.sd).clip(-6, 6).fillna(0).to_numpy(np.float32)
        ok = train_pos[(train_pos >= SEQ_LEN - 1) & np.isfinite(Y[train_pos]).all(axis=1)]
        n_val = max(int(0.1 * len(ok)), 200)          # last 10% of training for early stopping
        fit_pos, val_pos = ok[:-n_val], ok[-n_val:]
        n_t = Y.shape[1]
        self.n_t = n_t
        q = torch.tensor(QUANTILES, dtype=torch.float32).repeat(n_t)  # (n_t*n_q,)
        net = self._net(Z.shape[1], n_t * len(QUANTILES))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)

        def tensors(pos):
            xs = torch.tensor(self._sequences(Z, pos))
            ys = torch.tensor(np.repeat(Y[pos], len(QUANTILES), axis=1), dtype=torch.float32)
            return xs, ys

        def loss_fn(pred, y):
            e = y - pred
            return torch.maximum(q * e, (q - 1) * e).mean()

        Xf, Yf = tensors(fit_pos)
        Xv, Yv = tensors(val_pos)
        best, best_state, bad = np.inf, None, 0
        self.history = []  # validation pinball loss per epoch
        g = torch.Generator().manual_seed(SEED)
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xf), generator=g)
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(net(Xf[b]), Yf[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xv), Yv).item()
            self.history.append(v)
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net, self.Z = net, Z
        return self

    def predict_frame(self, X: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
        """Returns (len(pos), n_targets, n_quantiles)."""
        import torch
        Z = ((X - self.mu) / self.sd).clip(-6, 6).fillna(0).to_numpy(np.float32)
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(self._sequences(Z, pos))).numpy()
        return np.sort(out.reshape(len(pos), self.n_t, len(QUANTILES)), axis=2)


MODELS = {
    "hist_quantile": HistQuantile,
    "vol_band": VolBand,
    "xgboost": XGBQuantile,
    "catboost": CatQuantile,
    "lstm": LSTMQuantile,
}
MODEL_NAMES = {
    "hist_quantile": "Baseline: historical quantiles",
    "vol_band": "Baseline: volatility band",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lstm": "LSTM",
    "ensemble": "Ensemble (XGB + Cat + LSTM)",
}
ML_MODELS = ("xgboost", "catboost", "lstm")


def pinball(y: np.ndarray, q: np.ndarray) -> float:
    a = np.asarray(QUANTILES)[None, :]
    e = y[:, None] - q
    return float(np.mean(np.maximum(a * e, (a - 1) * e)))
