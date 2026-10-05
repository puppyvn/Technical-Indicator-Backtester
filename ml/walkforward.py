"""
Walk-forward prediction (Phase M3): the backbone every model level goes through.

For each block of `step` rows:
    1. train a FRESH model on the past only (minus a purge of `horizon` rows)
    2. predict the next block
    3. slide forward
Only these out-of-sample predictions are ever used for trading.

Model API (scikit-learn style):
    model.fit(X: DataFrame, y: ndarray)
    model.predict_proba(X: DataFrame) -> ndarray of shape (n, 2)
Optional attribute:
    model.context: int   number of PRIOR rows the model needs to score a row
                         (sequence models: lookback - 1). The loop supplies them.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    make_model: Callable[[], object],
    initial_train: int,
    step: int,
    horizon: int,
    min_train_rows: int = 100,
    train_window: int | None = None,
    return_models: bool = False,
):
    """
    Returns a Series `p_up` (same index as X). NaN before the first test block
    and wherever the features are missing.

    Parameters
    ----------
    make_model     factory returning a fresh untrained model (NOT an instance:
                   reusing one instance would leak knowledge between folds)
    initial_train  rows in the first training window
    step           rows predicted before the model is retrained
    horizon        label horizon; the last `horizon` rows before each test block
                   are excluded from training because their labels look into it
    train_window   None = expanding window; int = rolling window of that many rows
    return_models  also return [(train_end, model), ...] (e.g. for feature importance)
    """
    if not X.index.equals(y.index):
        y = y.reindex(X.index)
    n = len(X)
    if initial_train >= n:
        raise ValueError(f"initial_train ({initial_train}) must be smaller than the data length ({n})")

    preds = pd.Series(np.nan, index=X.index, name="p_up")
    valid_x = X.notna().all(axis=1).to_numpy()
    y_arr = y.to_numpy(dtype=float)
    models = []

    train_end = initial_train
    while train_end < n:
        test_end = min(train_end + step, n)
        fit_stop = train_end - horizon  # exclusive: last training row = train_end - horizon - 1
        fit_start = 0 if train_window is None else max(0, fit_stop - train_window)

        model = make_model()
        context = int(getattr(model, "context", 0))

        if context == 0:
            idx = np.arange(fit_start, fit_stop)
            idx = idx[valid_x[idx] & ~np.isnan(y_arr[idx])]
        else:
            # sequence models need contiguous rows; they skip windows containing NaN themselves
            seg = valid_x[fit_start:fit_stop]
            first_valid = fit_start + int(np.argmax(seg)) if seg.any() else fit_stop
            idx = np.arange(first_valid, fit_stop)

        if len(idx) < min_train_rows:
            logger.info("Skipping fold ending %d: only %d training rows", train_end, len(idx))
            train_end = test_end
            continue

        y_train = y_arr[idx]
        labelled = y_train[~np.isnan(y_train)]

        if len(np.unique(labelled)) < 2:
            # only one class seen in training: fall back to that class frequency
            const = float(labelled.mean())
            block = np.arange(train_end, test_end)
            preds.iloc[block] = np.where(valid_x[block], const, np.nan)
        else:
            model.fit(X.iloc[idx], y_train)
            if context == 0:
                block = np.arange(train_end, test_end)
                block = block[valid_x[block]]
                if len(block):
                    preds.iloc[block] = model.predict_proba(X.iloc[block])[:, 1]
            else:
                lo = max(0, train_end - context)
                proba = model.predict_proba(X.iloc[lo:test_end])[:, 1]
                preds.iloc[train_end:test_end] = proba[-(test_end - train_end):]

        if return_models:
            models.append((train_end, model))
        train_end = test_end

    return (preds, models) if return_models else preds
