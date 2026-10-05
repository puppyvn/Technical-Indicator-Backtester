"""
Tests for the ML extension. The important ones are the LEAKAGE tests:

* features are causal              (truncating the data does not change earlier rows)
* purge / ordering in walk-forward (training never touches the test block or its labels)
* future perturbation invariance   (mangling prices after day t0 cannot change any prediction at or before t0)
* noise / shuffled-label models show no skill
* a planted signal IS found        (so "no skill" results are meaningful)
"""

from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch

from ml.ensemble import (apply_regime_gate, combine_probabilities, learn_allowed_regimes,
                         size_position, smooth_position)
from ml.evaluate import (ablation_table, attach_ml_position, backtest_signal,
                         classification_metrics)
from ml.features import build_features
from ml.io import load_saved_signal, save_predictions
from ml.labels import build_labels
from ml.lstm_model import SequenceClassifier, make_sequence_model, make_sequences
from ml.regime import (RegimeModel, performance_by_regime, regime_features,
                       walk_forward_regimes)
from ml.synthetic import make_synthetic_ohlcv
from ml.tree_model import (average_feature_importance, make_tree_model,
                           probabilities_to_signal)
from ml.walkforward import walk_forward_predict

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def df():
    return make_synthetic_ohlcv(1500, seed=1)


def mangle_future(d: pd.DataFrame, t0: int, seed: int = 0) -> pd.DataFrame:
    """Scramble all prices strictly after row t0."""
    d2 = d.copy()
    rng = np.random.default_rng(seed)
    n_future = len(d2) - t0 - 1
    factor = rng.uniform(0.5, 2.0, (n_future, 1))
    d2.loc[t0 + 1:, ["Open", "High", "Low", "Close"]] = d2.loc[t0 + 1:, ["Open", "High", "Low", "Close"]].to_numpy() * factor
    return d2


# --------------------------------------------------------------- features / labels
def test_features_are_causal(df):
    full = build_features(df)
    cut = 900
    part = build_features(df.iloc[:cut].reset_index(drop=True))
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, rtol=1e-9, atol=1e-12)


def test_features_have_no_infinities_and_warmup_is_nan(df):
    f = build_features(df)
    assert not np.isinf(f.to_numpy(dtype=float)).any()
    assert f.iloc[:40].isna().any(axis=1).all()  # SMA_50 etc. need warm-up
    assert f.iloc[100:].notna().all(axis=None)


def test_labels_alignment(df):
    y = build_labels(df, horizon=5)
    assert y.iloc[-5:].isna().all() and y.iloc[:-5].notna().all()
    for t in (0, 100, 777):
        assert y.iloc[t] == float(df["Close"].iloc[t + 5] > df["Close"].iloc[t])


# ------------------------------------------------------------------ walk-forward
class Recorder:
    trained = []

    def fit(self, X, y):
        Recorder.trained.append((int(X.index.min()), int(X.index.max())))
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


def test_walkforward_purge_and_ordering(df):
    X, y = build_features(df), build_labels(df, 5)
    Recorder.trained = []
    p = walk_forward_predict(X, y, Recorder, initial_train=500, step=100, horizon=5)
    assert p.iloc[:500].isna().all()
    assert p.iloc[500:].notna().all()
    for k, (_, last_train_row) in enumerate(Recorder.trained):
        train_end = 500 + 100 * k
        assert last_train_row <= train_end - 5 - 1, "training window overlaps the purge zone"


def test_walkforward_hands_context_rows_to_sequence_models():
    class Last3Mean:
        context = 2

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            v = X.iloc[:, 0].rolling(3).mean().to_numpy()
            return np.column_stack([1 - v, v])

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 1, (600, 2)), columns=["a", "b"])
    y = pd.Series(rng.integers(0, 2, 600).astype(float))
    p = walk_forward_predict(X, y, Last3Mean, initial_train=300, step=50, horizon=1)
    expected = X["a"].rolling(3).mean()
    np.testing.assert_allclose(p.iloc[300:].to_numpy(), expected.iloc[300:].to_numpy())


def test_walkforward_rejects_initial_train_larger_than_data(df):
    X, y = build_features(df), build_labels(df, 5)
    with pytest.raises(ValueError):
        walk_forward_predict(X, y, Recorder, initial_train=len(df) + 1, step=50, horizon=5)


def test_random_probabilities_show_no_skill(df):
    class RandomModel:
        def __init__(self):
            self.rng = np.random.default_rng(0)

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            p = self.rng.uniform(0, 1, len(X))
            return np.column_stack([1 - p, p])

    X, y = build_features(df), build_labels(df, 5)
    p = walk_forward_predict(X, y, RandomModel, 500, 100, 5)
    auc = classification_metrics(y, p)["auc"]
    assert abs(auc - 0.5) < 0.06


def test_shuffled_labels_show_no_skill(df):
    X, y = build_features(df), build_labels(df, 5)
    rng = np.random.default_rng(0)
    y_shuf = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
    p = walk_forward_predict(X, y_shuf, partial(make_tree_model, "logistic"), 500, 100, 5)
    auc = classification_metrics(y_shuf, p)["auc"]
    assert 0.44 < auc < 0.56


def test_planted_signal_is_found():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((1500, 4)), columns=list("abcd"))
    y = pd.Series(((X["a"] + 0.5 * rng.standard_normal(1500)) > 0).astype(float))
    p = walk_forward_predict(X, y, partial(make_tree_model, "logistic"), 500, 100, horizon=1)
    assert classification_metrics(y, p)["auc"] > 0.8


def _future_invariance(run, t0=800):
    d = make_synthetic_ohlcv(1200, seed=3)
    a = run(d)
    b = run(mangle_future(d, t0))
    pd.testing.assert_series_equal(a.iloc[:t0 + 1], b.iloc[:t0 + 1], check_exact=False, rtol=1e-7, atol=1e-9)
    assert not a.iloc[t0 + 1:].equals(b.iloc[t0 + 1:]), "sanity: the perturbation must matter after t0"


@pytest.mark.parametrize("kind", ["logistic", "xgboost"])
def test_tree_pipeline_ignores_the_future(kind):
    def run(d):
        return walk_forward_predict(build_features(d), build_labels(d, 5),
                                    partial(make_tree_model, kind), 600, 100, 5)
    _future_invariance(run)


def test_sequence_pipeline_ignores_the_future():
    def run(d):
        factory = partial(make_sequence_model, lookback=10, hidden=8, epochs=3, seed=0)
        return walk_forward_predict(build_features(d), build_labels(d, 5), factory, 600, 200, 5)
    _future_invariance(run)


def test_regimes_ignore_the_future():
    def run(d):
        return walk_forward_regimes(regime_features(d), 3, 600, 100)
    d = make_synthetic_ohlcv(1200, seed=3)
    a, b = run(d), run(mangle_future(d, 800))
    pd.testing.assert_series_equal(a.iloc[:801], b.iloc[:801])


# ------------------------------------------------------------- tree helpers
def test_probabilities_to_signal_threshold_and_hysteresis():
    p = pd.Series([0.4, 0.6, 0.55, 0.5, 0.44, 0.6, np.nan])
    plain = probabilities_to_signal(p, 0.55)
    assert plain.iloc[:6].tolist() == [0, 1, 0, 0, 0, 1] and np.isnan(plain.iloc[6])
    hyst = probabilities_to_signal(p, threshold=0.58, exit_threshold=0.47)
    assert hyst.iloc[:6].tolist() == [0, 1, 1, 1, 0, 1] and np.isnan(hyst.iloc[6])


def test_feature_importance_averages_over_folds(df):
    X, y = build_features(df), build_labels(df, 5)
    _, models = walk_forward_predict(X, y, partial(make_tree_model, "random_forest", n_estimators=30),
                                     700, 200, 5, return_models=True)
    imp = average_feature_importance(models, list(X.columns))
    assert len(imp) == X.shape[1] and abs(imp.sum() - 1) < 1e-6


def test_unknown_model_kind_is_rejected():
    with pytest.raises(ValueError):
        make_tree_model("magic")


# ------------------------------------------------------------------ regimes
def test_regime_ranking_is_stable_and_meaningful(df):
    rf = regime_features(df)
    for end in (800, 1200):
        m = RegimeModel(3).fit(rf.iloc[:end])
        assert m.describe()["vol_20"].is_monotonic_increasing  # 0 = calmest ... 2 = most volatile
    reg = walk_forward_regimes(rf, 3, 700, 100)
    assert reg.iloc[:700].isna().all()
    assert set(reg.dropna().unique()) <= {0.0, 1.0, 2.0}
    perf = performance_by_regime(df["Close"].pct_change(), reg)
    assert perf.loc[2, "ann_vol"] > perf.loc[0, "ann_vol"]


def test_regime_predict_proba_columns_follow_the_ranking(df):
    rf = regime_features(df)
    m = RegimeModel(3).fit(rf.iloc[:900])
    proba = m.predict_proba(rf.iloc[900:])
    hard = m.predict(rf.iloc[900:])
    valid = hard.notna()
    assert (proba[valid].to_numpy().argmax(axis=1) == hard[valid].to_numpy()).all()
    np.testing.assert_allclose(proba[valid].sum(axis=1), 1.0)


# ------------------------------------------------------------ sequence model
def test_make_sequences_windows_and_alignment():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = (np.arange(20) % 2).astype(float)
    W, t, idx = make_sequences(X, y, 5)
    assert W.shape == (16, 5, 2) and idx[0] == 4
    np.testing.assert_array_equal(W[0], X[0:5])
    np.testing.assert_array_equal(t, y[idx])  # target = label at the window's LAST row
    X2 = X.copy()
    X2[7, 0] = np.nan
    W2, _, idx2 = make_sequences(X2, y, 5)
    assert len(W2) == 16 - 5 and not {7, 8, 9, 10, 11} & set(idx2)


def test_make_sequences_short_input():
    W, t, idx = make_sequences(np.zeros((3, 2)), np.zeros(3), 5)
    assert len(W) == 0 and len(idx) == 0


def test_sequence_model_learns_planted_signal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1600, 3))
    y = (X[:, 0] > 0).astype(float)  # depends on the most recent row of the window
    m = SequenceClassifier(lookback=10, hidden=16, epochs=25, patience=5, seed=0, device="cpu")
    m.fit(X[:1100], y[:1100])
    proba = m.predict_proba(X[1090:])
    assert proba.shape == (510, 2)
    assert np.isnan(proba[:9]).all() and not np.isnan(proba[9:]).any()  # context rows have no full window
    p = pd.Series(proba[9:, 1])
    auc = classification_metrics(pd.Series(y[1099:]), p)["auc"]
    assert auc > 0.9


def test_sequence_model_probabilities_are_valid_and_seeded():
    rng = np.random.default_rng(1)
    X, y = rng.standard_normal((400, 3)), rng.integers(0, 2, 400).astype(float)
    a = SequenceClassifier(lookback=8, hidden=8, epochs=2, seed=5, device="cpu").fit(X, y).predict_proba(X)
    b = SequenceClassifier(lookback=8, hidden=8, epochs=2, seed=5, device="cpu").fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)  # same seed -> same result
    assert np.nanmin(a) >= 0 and np.nanmax(a) <= 1


def test_sequence_model_gru_and_multiseed_average():
    rng = np.random.default_rng(2)
    X, y = rng.standard_normal((400, 3)), rng.integers(0, 2, 400).astype(float)
    m = SequenceClassifier(lookback=8, hidden=8, epochs=2, cell="gru", n_seeds=2, device="cpu").fit(X, y)
    assert len(m.nets_) == 2 and m.predict_proba(X).shape == (400, 2)


def test_sequence_model_needs_enough_data():
    with pytest.raises(ValueError):
        SequenceClassifier(lookback=10, device="cpu").fit(np.zeros((30, 2)), np.zeros(30))


# ---------------------------------------------------------------- ensemble
def test_combine_probabilities_requires_all_members():
    a = pd.Series([0.6, 0.8, np.nan, 0.5])
    b = pd.Series([0.4, np.nan, 0.7, 0.9])
    out = combine_probabilities([a, b])
    assert out.iloc[0] == pytest.approx(0.5) and out.iloc[3] == pytest.approx(0.7)
    assert out.iloc[[1, 2]].isna().all()
    w = combine_probabilities([a, b], weights=[3, 1])
    assert w.iloc[0] == pytest.approx(0.55)


def test_regime_gate():
    sig = pd.Series([1.0, 1.0, 1.0, np.nan])
    reg = pd.Series([0.0, 1.0, np.nan, 0.0])
    out = apply_regime_gate(sig, reg, {0})
    assert out.iloc[:3].tolist() == [1.0, 0.0, 0.0] and np.isnan(out.iloc[3])


def test_learn_allowed_regimes():
    rng = np.random.default_rng(0)
    regime = pd.Series(np.repeat([0, 1], 200)).astype(float)
    ret = pd.Series(np.concatenate([rng.normal(0.002, 0.01, 200), rng.normal(-0.002, 0.01, 200)]))
    assert learn_allowed_regimes(ret, regime) == {0}
    assert learn_allowed_regimes(ret, regime, min_days=500) == set()


def test_size_position_only_scales_down():
    sig = pd.Series([1.0, 1.0, 0.0])
    vol = pd.Series([0.04, 0.001, 0.02])  # daily; annualised ~63%, ~1.6%, ~32%
    pos = size_position(sig, vol, target_vol=0.15)
    assert pos.iloc[0] == pytest.approx(0.15 / (0.04 * np.sqrt(252)))
    assert pos.iloc[1] == 1.0 and pos.iloc[2] == 0.0


def test_smooth_position_suppresses_small_changes():
    out = smooth_position(pd.Series([0, 0.05, 0.12, 0.15, 0.4, 0.0]), min_change=0.1)
    np.testing.assert_allclose(out.to_numpy(), [0, 0, 0.12, 0.12, 0.4, 0])


# ---------------------------------------------------------- evaluate / backtest
def test_classification_metrics_perfect_and_baseline():
    y = pd.Series([1, 0, 1, 1, 0, 1.0])
    perfect = classification_metrics(y, y * 0.8 + 0.1)
    assert perfect["accuracy"] == 1.0 and perfect["auc"] == 1.0
    assert perfect["baseline_accuracy"] == pytest.approx(4 / 6)


def test_backtest_signal_always_long_equals_buy_and_hold(df):
    sig = pd.Series(np.nan, index=df.index)
    sig.iloc[400:] = 1.0
    bt, stats = backtest_signal(df, sig, fee_bps=0)
    assert len(bt) == len(df) - 400  # window starts at the first valid signal
    assert bt["strategy_equity"].iloc[-1] == pytest.approx(bt["buy_hold_equity"].iloc[-1])
    assert stats["Strategy"]["Trades"] == 1


def test_position_is_signal_shifted_by_one_day(df):
    sig = pd.Series(np.nan, index=df.index)
    sig.iloc[400:] = 0.0
    sig.iloc[500] = 1.0
    d = attach_ml_position(df, sig)
    held = d.index[d["position"] == 1.0].tolist()
    assert held == [501 - 400]  # decided at close of row 500, held during row 501


def test_backtest_signal_fees_reduce_return(df):
    rng = np.random.default_rng(0)
    sig = pd.Series(np.nan, index=df.index)
    sig.iloc[400:] = rng.integers(0, 2, len(df) - 400).astype(float)  # trades constantly
    _, free = backtest_signal(df, sig, fee_bps=0)
    _, costly = backtest_signal(df, sig, fee_bps=20)
    assert costly["Strategy"]["Total Return"] < free["Strategy"]["Total Return"]
    assert costly["Strategy"]["Turnover"] == pytest.approx(free["Strategy"]["Turnover"])


def test_start_and_end_narrow_the_window(df):
    sig = pd.Series(1.0, index=df.index)
    bt, _ = backtest_signal(df, sig, start=1000, end=1099)
    assert len(bt) == 100


def test_ablation_table_shape():
    row = {"Total Return": .1, "CAGR": .05, "Annualized Volatility": .2, "Sharpe Ratio": .3,
           "Max Drawdown": -.2, "Trades": 4, "Avg Exposure": .5}
    t = ablation_table([("a", row), ("b", row)])
    assert list(t.index) == ["a", "b"] and "Sharpe Ratio" in t.columns


# ---------------------------------------------------------------------- io
def test_prediction_roundtrip(tmp_path, df):
    frame = pd.DataFrame({"signal_ml": np.where(df.index >= 700, 0.5, np.nan)}, index=df.index)
    save_predictions("abc", df, frame, directory=str(tmp_path))
    back = load_saved_signal("ABC", df, directory=str(tmp_path))
    pd.testing.assert_series_equal(back, frame["signal_ml"], check_names=False)
    assert load_saved_signal("NOPE", df, directory=str(tmp_path)) is None


# ------------------------------------------------------------------ timing test
from ml.evaluate import timing_test  # noqa: E402


def test_timing_test_matches_engine_sharpe(df):
    rng = np.random.default_rng(0)
    sig = pd.Series(np.nan, index=df.index)
    sig.iloc[400:] = rng.integers(0, 2, len(df) - 400).astype(float)
    _, stats = backtest_signal(df, sig, fee_bps=5)
    res = timing_test(df, sig, fee_bps=5, n_shifts=50)
    # the test skips the structurally-flat first day, so it agrees with the engine to ~1 day in 1000
    assert res["sharpe"] == pytest.approx(stats["Strategy"]["Sharpe Ratio"], rel=0.02)


def test_timing_test_flags_perfect_foresight_and_not_random_timing(df):
    ret = df["Close"].pct_change()
    foresight = (ret.shift(-1) > 0).astype(float)  # cheats: uses tomorrow's return
    foresight.iloc[:400] = np.nan
    assert timing_test(df, foresight, fee_bps=0)["p_value"] < 0.02

    rng = np.random.default_rng(1)
    noise = pd.Series(np.nan, index=df.index)
    noise.iloc[400:] = rng.integers(0, 2, len(df) - 400).astype(float)
    assert timing_test(df, noise, fee_bps=0)["p_value"] > 0.05


def test_timing_test_gives_no_credit_for_mere_exposure(df):
    always_long = pd.Series(np.nan, index=df.index)
    always_long.iloc[400:] = 1.0
    res = timing_test(df, always_long, fee_bps=0, n_shifts=100)
    assert res["p_value"] == pytest.approx(1.0)  # every shifted copy is identical to the real one


# ------------------------------------------------------------- rolling regime gate
from ml.ensemble import apply_gate_mask, rolling_regime_gate  # noqa: E402


def _regime_world(n=1200, seed=0):
    """Regime 0 earns money, regime 1 loses money; the regime alternates in long runs."""
    rng = np.random.default_rng(seed)
    regime = pd.Series(np.repeat(rng.integers(0, 2, n // 50 + 1), 50)[:n]).astype(float)
    mu = np.where(regime == 0, 0.0015, -0.0015)
    ret = pd.Series(mu + 0.01 * rng.standard_normal(n))
    return ret, regime


def test_rolling_gate_blocks_the_losing_regime_and_allows_the_winning_one():
    ret, regime = _regime_world()
    ok, sched = rolling_regime_gate(ret, regime, refresh=100, window=500, min_days=30, n_regimes=2)
    late = sched[sched["block_start"] >= 600]
    assert all(a == (0,) for a in late["allowed"])           # only the profitable regime, once evidence exists
    rows = np.arange(600, len(ret))
    assert ok.iloc[rows][regime.iloc[rows] == 0].all()
    assert not ok.iloc[rows][regime.iloc[rows] == 1].any()


def test_rolling_gate_warmup_allows_everything_because_there_is_no_evidence_yet():
    ret, regime = _regime_world()
    _, sched = rolling_regime_gate(ret, regime, refresh=100, window=500, min_days=30, n_regimes=2)
    assert sched["allowed"].iloc[0] == (0, 1)


@pytest.mark.parametrize("s", [600, 700, 800])
def test_rolling_gate_decision_for_a_block_never_uses_that_block_or_later_data(s):
    ret, regime = _regime_world()
    kw = dict(refresh=100, window=500, min_days=30, n_regimes=2)
    _, sch1 = rolling_regime_gate(ret, regime, **kw)

    # Replace EVERYTHING from row s onward with extreme, random data. A gate that peeks at its own
    # block (or later) must change its answer; a causal one cannot. (A mild perturbation is not
    # enough: it can leave the decision unchanged by luck, which hides real lookahead bugs.)
    rng = np.random.default_rng(99)
    ret2, reg2 = ret.copy(), regime.copy()
    ret2.iloc[s:] = rng.normal(-0.3, 0.3, len(ret) - s)
    reg2.iloc[s:] = rng.integers(0, 2, len(ret) - s).astype(float)
    _, sch2 = rolling_regime_gate(ret2, reg2, **kw)

    before = lambda sch: sch[sch["block_start"] <= s]["allowed"].tolist()
    after = lambda sch: sch[sch["block_start"] > s]["allowed"].tolist()
    assert before(sch1) == before(sch2), "a block's allowed set changed when only its own/later data changed"
    assert after(sch1) != after(sch2), "sanity: later blocks must react to the changed data"


def test_rolling_gate_adapts_when_the_world_changes():
    # regime 1 is good for the first half, then turns bad: a static decision learned early
    # would keep trading it forever; the rolling gate must drop it
    rng = np.random.default_rng(3)
    n = 2000
    regime = pd.Series(np.repeat(rng.integers(0, 2, n // 50), 50)).astype(float)
    mu1 = np.where(np.arange(n) < 1000, 0.0015, -0.0020)
    ret = pd.Series(np.where(regime == 1, mu1, 0.0002) + 0.01 * rng.standard_normal(n))
    _, sched = rolling_regime_gate(ret, regime, refresh=100, window=400, min_days=30, n_regimes=2)
    early = sched[(sched["block_start"] >= 500) & (sched["block_start"] < 900)]["allowed"]
    late = sched[sched["block_start"] >= 1600]["allowed"]
    assert all(1 in a for a in early) and all(1 not in a for a in late)
    static = learn_allowed_regimes(ret.iloc[:800], regime.iloc[:800])
    assert 1 in static  # the one-off decision would still allow it in the second half


def test_unknown_regimes_are_allowed_only_when_asked():
    ret = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 100))
    regime = pd.Series([0.0] * 95 + [1.0] * 5)             # regime 1 has only 5 observations
    strict = learn_allowed_regimes(ret, regime, min_days=30)
    lenient = learn_allowed_regimes(ret, regime, min_days=30, allow_unknown=True, n_regimes=3)
    assert 1 not in strict and 2 not in strict
    assert {1, 2} <= lenient                                # rare (1) and never-seen (2) regimes: no evidence against
    assert learn_allowed_regimes(pd.Series(dtype=float), pd.Series(dtype=float),
                                 allow_unknown=True, n_regimes=3) == {0, 1, 2}


def test_unknown_regime_closes_the_gate_like_the_static_gate():
    ret, regime = _regime_world(400)
    regime.iloc[300:320] = np.nan
    ok, _ = rolling_regime_gate(ret, regime, refresh=100, window=300, min_days=30, n_regimes=2)
    assert not ok.iloc[300:320].any()


def test_apply_gate_mask_zeroes_blocked_rows_and_keeps_nan():
    sig = pd.Series([1.0, 1.0, 1.0, np.nan])
    out = apply_gate_mask(sig, pd.Series([True, False, True, True]))
    assert out.iloc[:3].tolist() == [1.0, 0.0, 1.0] and np.isnan(out.iloc[3])


# ------------------------------------------------------- runner: gate modes + diagnostics
from experiments.run_ablation import Config, exposure_warnings, print_diagnostics, run_ticker  # noqa: E402


@pytest.fixture(scope="module")
def syn_df():
    return make_synthetic_ohlcv(2600, seed=0, momentum=0.1)


@pytest.mark.parametrize("mode", ["static", "rolling", "off"])
def test_runner_supports_every_gate_mode(syn_df, mode, capsys):
    res = run_ticker(syn_df, "T", Config(kind="logistic", initial_train=800, step=126,
                                          gate_mode=mode, gate_refresh=126, gate_window=504))
    labels = list(res["table"].index)
    assert any(l.startswith(f"+ regime gate ({mode})") for l in labels)
    print_diagnostics(res)
    out = capsys.readouterr().out
    assert "by regime on the VALIDATION slice" in out


def test_gate_off_equals_the_ungated_ensemble(syn_df):
    res = run_ticker(syn_df, "T", Config(kind="logistic", initial_train=800, step=126, gate_mode="off"))
    t = res["table"]
    assert t.loc["+ regime gate (off)", "Avg Exposure"] == pytest.approx(t.loc["Level 1: xgboost".replace("xgboost", "logistic"), "Avg Exposure"])


def test_rolling_gate_exposes_a_schedule_and_never_uses_the_future(syn_df):
    cfg = Config(kind="logistic", initial_train=800, step=126, gate_mode="rolling", gate_refresh=126, gate_window=504)
    res = run_ticker(syn_df, "T", cfg)
    assert res["info"]["gate_schedule"] is not None and len(res["info"]["gate_schedule"]) > 5
    # the whole pipeline with the rolling gate still ignores the future
    t0 = 1800
    a = res["predictions"]["signal_ml"]
    b = run_ticker(mangle_future(syn_df, t0), "T", cfg)["predictions"]["signal_ml"]
    pd.testing.assert_series_equal(a.iloc[:t0 + 1], b.iloc[:t0 + 1], check_exact=False, rtol=1e-7, atol=1e-9)


def test_unknown_gate_mode_is_rejected(syn_df):
    with pytest.raises(ValueError):
        run_ticker(syn_df, "T", Config(kind="logistic", initial_train=800, gate_mode="magic"))


def test_exposure_warning_names_the_layer_that_collapses_the_exposure():
    idx = ["Buy & hold", "Level 1: xgboost", "+ regime gate (static)", "+ vol sizing"]
    t = pd.DataFrame({"Avg Exposure": [np.nan, 0.75, 0.048, 0.013]}, index=idx)
    msgs = exposure_warnings(t)
    assert len(msgs) == 1 and "+ regime gate (static)" in msgs[0]            # sizing dropped from an already-tiny base: not blamed
    healthy = pd.DataFrame({"Avg Exposure": [np.nan, 0.5, 0.47, 0.44]}, index=idx)
    assert exposure_warnings(healthy) == []


def test_exposure_warning_flags_the_ensemble_step_when_it_averages_models_into_cash():
    idx = ["Buy & hold", "Level 1: xgboost", "Level 3: sequence model", "Ensemble (avg P)", "+ regime gate (static)"]
    t = pd.DataFrame({"Avg Exposure": [np.nan, 0.752, 0.657, 0.095, 0.001]}, index=idx)   # the AAPL --lstm log
    msgs = exposure_warnings(t)
    assert any("Ensemble (avg P)" in m for m in msgs)


def test_era_shift_generator_is_continuous_and_has_two_eras():
    from ml.synthetic import make_era_shift_ohlcv
    d = make_era_shift_ohlcv(n_first=500, n_second=700, seed=1)
    assert len(d) == 1200 and d["Date"].is_monotonic_increasing and d["Date"].is_unique
    assert (d[["Open", "High", "Low", "Close"]] > 0).all().all()
    r = d["Close"].pct_change()
    assert abs(r.iloc[500]) < 0.15                       # no price gap at the splice
    ac = lambda x: x.autocorr(1)
    assert ac(r.iloc[501:]) > ac(r.iloc[1:500]) + 0.1    # era 2 carries the autocorrelation structure


def test_model_free_baseline_does_not_depend_on_the_model(syn_df):
    base = "Baseline: vol-targeted buy & hold (no model)"
    rows = {}
    for kind in ("logistic", "hist_gbm"):
        t = run_ticker(syn_df, "T", Config(kind=kind, initial_train=800, step=126, gate_mode="off"))["table"]
        assert base in t.index and list(t.index)[-1] == base
        rows[kind] = t.loc[base]
    pd.testing.assert_series_equal(rows["logistic"], rows["hist_gbm"])      # identical whatever the model is
    assert 0 < rows["logistic"]["Avg Exposure"] <= 1.0


def test_diagnostics_compare_the_model_against_the_baseline(syn_df, capsys):
    res = run_ticker(syn_df, "T", Config(kind="logistic", initial_train=800, step=126, gate_mode="off"))
    print_diagnostics(res)
    out = capsys.readouterr().out
    assert "Model vs model-free volatility targeting" in out and "not evidence of model skill" in out
