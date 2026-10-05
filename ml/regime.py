"""
Level 2: market-regime detection (Phase M5).

Unsupervised: a Gaussian mixture groups days that look alike in (volatility,
momentum, trend). Two traps handled here:

1. LABEL SWITCHING. Cluster ids are arbitrary and change between refits, so after
   every fit clusters are ranked by mean volatility: regime 0 = calmest,
   regime n-1 = most turbulent, always.
2. LOOKAHEAD. A Gaussian mixture scores each day independently, so it is causal
   once fitted on the past. (An HMM's predict / predict_proba use the whole
   sequence you pass in, including days after t, which would leak the future.)
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REGIME_FEATURES = ["vol_20", "ret_20", "trend_50"]


def regime_features(df: pd.DataFrame, price_col: str = "Close", periods: int = 252) -> pd.DataFrame:
    """A few regime-descriptive, causal features (annualised volatility, 20d return, trend)."""
    close = df[price_col]
    ret1 = close.pct_change()
    f = pd.DataFrame(index=df.index)
    f["vol_20"] = ret1.rolling(20).std() * np.sqrt(periods)
    f["ret_20"] = close.pct_change(20)
    f["trend_50"] = close / close.rolling(50).mean() - 1
    return f


class RegimeModel:
    def __init__(self, n_regimes: int = 3, n_init: int = 5, random_state: int = 0,
                 order_by: str = "vol_20"):
        self.n_regimes = n_regimes
        self.n_init = n_init
        self.random_state = random_state
        self.order_by = order_by

    def fit(self, feats: pd.DataFrame) -> "RegimeModel":
        X = feats[REGIME_FEATURES].dropna()
        if len(X) < 20 * self.n_regimes:
            raise ValueError("Not enough rows to fit the regime model")
        self.scaler_ = StandardScaler().fit(X)
        self.gmm_ = GaussianMixture(
            n_components=self.n_regimes, covariance_type="full", n_init=self.n_init,
            reg_covar=1e-4, random_state=self.random_state,
        ).fit(self.scaler_.transform(X))

        # rank clusters by mean volatility in ORIGINAL units -> stable meaning across refits
        means = self.scaler_.inverse_transform(self.gmm_.means_)
        order = np.argsort(means[:, REGIME_FEATURES.index(self.order_by)])
        self.rank_of_ = np.empty(self.n_regimes, dtype=int)
        self.rank_of_[order] = np.arange(self.n_regimes)
        return self

    def _valid(self, feats: pd.DataFrame):
        X = feats[REGIME_FEATURES]
        return X.notna().all(axis=1).to_numpy(), X

    def predict(self, feats: pd.DataFrame) -> pd.Series:
        ok, X = self._valid(feats)
        out = np.full(len(feats), np.nan)
        if ok.any():
            raw = self.gmm_.predict(self.scaler_.transform(X[ok]))
            out[ok] = self.rank_of_[raw]
        return pd.Series(out, index=feats.index, name="regime")

    def predict_proba(self, feats: pd.DataFrame) -> pd.DataFrame:
        ok, X = self._valid(feats)
        out = np.full((len(feats), self.n_regimes), np.nan)
        if ok.any():
            raw_p = self.gmm_.predict_proba(self.scaler_.transform(X[ok]))
            ranked = np.empty_like(raw_p)
            ranked[:, self.rank_of_] = raw_p  # column for raw cluster k moves to column rank_of_[k]
            out[ok] = ranked
        return pd.DataFrame(out, index=feats.index,
                            columns=[f"regime_{i}" for i in range(self.n_regimes)])

    def describe(self) -> pd.DataFrame:
        """Mean feature values and weight per regime (ranked), for interpretation."""
        means = self.scaler_.inverse_transform(self.gmm_.means_)
        table = pd.DataFrame(means, columns=REGIME_FEATURES)
        table["weight"] = self.gmm_.weights_
        table["regime"] = self.rank_of_
        return table.set_index("regime").sort_index()


def walk_forward_regimes(
    feats: pd.DataFrame,
    n_regimes: int = 3,
    initial_train: int = 750,
    step: int = 63,
    train_window: int | None = None,
    **model_kwargs,
) -> pd.Series:
    """Refit on the past every `step` rows, label the next block. Out-of-sample regimes."""
    regime = pd.Series(np.nan, index=feats.index, name="regime")
    n = len(feats)
    train_end = initial_train
    while train_end < n:
        test_end = min(train_end + step, n)
        start = 0 if train_window is None else max(0, train_end - train_window)
        train = feats.iloc[start:train_end]
        if train[REGIME_FEATURES].dropna().shape[0] >= 20 * n_regimes:
            model = RegimeModel(n_regimes, **model_kwargs).fit(train)
            regime.iloc[train_end:test_end] = model.predict(feats.iloc[train_end:test_end]).to_numpy()
        train_end = test_end
    return regime


def performance_by_regime(returns: pd.Series, regime: pd.Series, periods: int = 252) -> pd.DataFrame:
    """Daily strategy returns grouped by regime -> days, share, annualised return / vol, Sharpe, hit rate."""
    d = pd.DataFrame({"r": returns, "regime": regime}).dropna()
    rows = {}
    for k, g in d.groupby("regime"):
        vol = g["r"].std() * np.sqrt(periods)
        rows[int(k)] = {
            "days": len(g),
            "share": len(g) / len(d),
            "ann_return": g["r"].mean() * periods,
            "ann_vol": vol,
            "sharpe": (g["r"].mean() * periods) / vol if vol > 0 else np.nan,
            "hit_rate": (g["r"] > 0).mean(),
        }
    return pd.DataFrame(rows).T.rename_axis("regime")
