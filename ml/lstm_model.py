"""
Level 3: LSTM / GRU sequence classifier (Phase M6), PyTorch.

Input to the network is a window of the last `lookback` days of features:
shape (samples, lookback, n_features). The window for sample t ends AT row t,
so it can never see the future.

Exposes a scikit-learn style API (fit / predict_proba) plus `context`, so it plugs
into `walk_forward_predict` exactly like the tree models.

Guard rails built in:
  * feature scaling is fitted on the training rows only
  * validation for early stopping is the LAST slice of the training windows
    (time-ordered, with a small embargo gap), never a random split
  * optional averaging over several random seeds (neural nets are noisy)
  * the network is deliberately tiny: bigger models overfit faster than they learn
"""

import copy

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


def make_sequences(X: np.ndarray, y: np.ndarray | None, lookback: int):
    """
    Slide a window of `lookback` rows over X.

    Returns (windows, targets, end_idx):
        windows  (m, lookback, n_features) float32
        targets  (m,) = y at each window's LAST row  (None if y is None)
        end_idx  (m,) row position of each window's last row
    Windows containing NaN (or with a NaN target) are dropped.
    """
    n, f = X.shape
    if n < lookback:
        return (np.empty((0, lookback, f), np.float32),
                None if y is None else np.empty(0), np.empty(0, int))
    win = np.lib.stride_tricks.sliding_window_view(X, lookback, axis=0)  # (n-L+1, f, L)
    win = win.transpose(0, 2, 1)                                          # (n-L+1, L, f)
    end_idx = np.arange(lookback - 1, n)
    ok = ~np.isnan(win).any(axis=(1, 2))
    if y is not None:
        ok &= ~np.isnan(y[end_idx])
    targets = None if y is None else y[end_idx][ok]
    return win[ok].astype(np.float32), targets, end_idx[ok]


class _SeqNet(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_layers: int, dropout: float, cell: str):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            n_features, hidden, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)  # logit from the last time step


class SequenceClassifier:
    def __init__(
        self,
        lookback: int = 30,
        hidden: int = 32,
        n_layers: int = 1,
        dropout: float = 0.2,
        cell: str = "lstm",          # "lstm" or "gru"
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.15,
        val_gap: int = 5,            # windows dropped between train and validation
        patience: int = 6,
        n_seeds: int = 1,
        seed: int = 0,
        clip: float = 5.0,           # clip scaled features to +/- clip std-devs (outlier guard)
        verbose: bool = False,
    ):
        self.lookback = lookback
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.cell = cell
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.val_gap = val_gap
        self.patience = patience
        self.n_seeds = n_seeds
        self.seed = seed
        self.clip = clip
        self.verbose = verbose

    @property
    def context(self) -> int:
        """Prior rows needed to score a row (used by walk_forward_predict)."""
        return self.lookback - 1

    # ------------------------------------------------------------------ helpers
    def _scale(self, X) -> np.ndarray:
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        return np.clip(Z, -self.clip, self.clip)  # NaN stays NaN

    def _train_one(self, Wtr, ttr, Wva, tva, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = _SeqNet(Wtr.shape[2], self.hidden, self.n_layers, self.dropout, self.cell)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        p = float(ttr.mean())
        pos_weight = torch.tensor((1 - p) / p if 0 < p < 1 else 1.0, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # balances up/down classes

        Xtr, ytr = torch.from_numpy(Wtr), torch.from_numpy(ttr.astype(np.float32))
        Xva = torch.from_numpy(Wva) if len(Wva) else None
        yva = torch.from_numpy(tva.astype(np.float32)) if len(Wva) else None

        best_loss, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
        for epoch in range(self.epochs):
            net.train()
            perm = torch.from_numpy(rng.permutation(len(Xtr)))
            for i in range(0, len(perm), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xtr[b]), ytr[b])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

            if Xva is None:
                continue
            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(Xva), yva))
            if self.verbose:
                print(f"  epoch {epoch + 1:3d}  val_loss {val_loss:.4f}")
            if val_loss < best_loss - 1e-5:
                best_loss, best_state, bad = val_loss, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if Xva is not None:
            net.load_state_dict(best_state)  # roll back to the best validation epoch
        net.eval()
        return net

    # ---------------------------------------------------------------------- API
    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=float)
        self.scaler_ = StandardScaler().fit(Xa)  # training rows only; ignores NaN
        W, t, _ = make_sequences(self._scale(Xa), ya, self.lookback)
        if len(W) < 50:
            raise ValueError(f"Only {len(W)} usable training windows; need at least 50")

        self.nets_, self.constant_ = [], None
        if len(np.unique(t)) < 2:
            self.constant_ = float(t.mean())
            return self

        n_val = int(len(W) * self.val_fraction)
        split = len(W) - n_val
        tr_stop = max(1, split - self.val_gap) if n_val else len(W)  # embargo before validation
        Wtr, ttr = W[:tr_stop], t[:tr_stop]
        Wva, tva = W[split:], t[split:]

        for s in range(self.n_seeds):
            self.nets_.append(self._train_one(Wtr, ttr, Wva, tva, self.seed + s))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """(n, 2) array. The first `lookback - 1` rows (no full window) are NaN."""
        Xa = np.asarray(X, dtype=float)
        p = np.full(len(Xa), np.nan)
        W, _, end_idx = make_sequences(self._scale(Xa), None, self.lookback)
        if len(W):
            if self.constant_ is not None:
                p[end_idx] = self.constant_
            else:
                Wt = torch.from_numpy(W)
                with torch.no_grad():
                    probs = np.mean(
                        [torch.sigmoid(net(Wt)).numpy() for net in self.nets_], axis=0
                    )
                p[end_idx] = probs
        return np.column_stack([1 - p, p])


def make_sequence_model(**params) -> SequenceClassifier:
    """Factory: `functools.partial(make_sequence_model, lookback=30, hidden=32)` for walk-forward."""
    return SequenceClassifier(**params)
