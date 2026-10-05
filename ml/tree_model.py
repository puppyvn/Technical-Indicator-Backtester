"""
Level 1: tree-based (and linear baseline) classifiers (Phase M4).

Regularisation matters more than tuning on noisy financial data: shallow trees,
few estimators, subsampling. Start with `logistic` as the honest baseline; if
XGBoost does not beat it out-of-sample, that is a finding, not a failure.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_DEFAULTS = {
    "logistic": dict(C=0.5, max_iter=1000),
    "random_forest": dict(
        n_estimators=300, max_depth=5, min_samples_leaf=30,
        max_features="sqrt", n_jobs=-1, random_state=0,
    ),
    "xgboost": dict(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
        eval_metric="logloss", tree_method="hist", random_state=0, n_jobs=1,
    ),
    "hist_gbm": dict(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
}


def make_tree_model(kind: str = "xgboost", **params):
    """Return a FRESH, untrained model. Pass `functools.partial(make_tree_model, "xgboost")`
    to `walk_forward_predict` as the factory."""
    if kind not in _DEFAULTS:
        raise ValueError(f"Unknown model kind '{kind}'. Choose from {sorted(_DEFAULTS)}")
    cfg = {**_DEFAULTS[kind], **params}

    if kind == "logistic":
        # linear models need scaled inputs; the scaler is fitted inside each fold
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(**cfg))])
    if kind == "random_forest":
        return RandomForestClassifier(**cfg)
    if kind == "hist_gbm":
        return HistGradientBoostingClassifier(**cfg)
    from xgboost import XGBClassifier  # imported lazily so the package works without xgboost
    return XGBClassifier(**cfg)


def probabilities_to_signal(
    p: pd.Series,
    threshold: float = 0.55,
    exit_threshold: float | None = None,
) -> pd.Series:
    """
    P(up) -> 0/1 signal (decision at the close of day t; shift by 1 before trading).

    exit_threshold=None : long whenever p > threshold.
    exit_threshold=x    : hysteresis. Enter above `threshold`, stay in until p < x.
                          Cuts churn (and therefore fees).
    NaN probabilities stay NaN (they mark "no prediction available").
    """
    if exit_threshold is None:
        sig = (p > threshold).astype(float)
        sig[p.isna()] = np.nan
        return sig.rename("signal")

    out = np.full(len(p), np.nan)
    holding = 0.0
    for i, v in enumerate(p.to_numpy(dtype=float)):
        if np.isnan(v):
            continue
        if v > threshold:
            holding = 1.0
        elif v < exit_threshold:
            holding = 0.0
        out[i] = holding
    return pd.Series(out, index=p.index, name="signal")


def average_feature_importance(models: list, columns: list[str]) -> pd.Series:
    """Mean importance across walk-forward folds. `models` = [(train_end, model), ...]."""
    rows = []
    for _, m in models:
        if hasattr(m, "feature_importances_"):
            imp = np.asarray(m.feature_importances_, dtype=float)
        elif hasattr(m, "named_steps") and hasattr(m.named_steps.get("clf"), "coef_"):
            imp = np.abs(m.named_steps["clf"].coef_.ravel())  # |coefficient| on scaled inputs
        else:
            continue
        rows.append(imp)
    if not rows:
        raise ValueError("None of the models expose feature importances")
    return pd.Series(np.mean(rows, axis=0), index=columns).sort_values(ascending=False)
