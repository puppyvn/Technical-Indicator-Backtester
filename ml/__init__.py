"""
ML extension for the technical-indicator backtester.

Pipeline (each stage outputs a pandas object aligned to the price DataFrame):

    features.py     -> X   (features known at the close of day t)
    labels.py       -> y   (was the price higher `horizon` days later?)
    walkforward.py  -> out-of-sample P(up) for ANY model with fit / predict_proba
    tree_model.py   -> Level 1: logistic / random forest / XGBoost
    regime.py       -> Level 2: market-regime detection (Gaussian mixture)
    lstm_model.py   -> Level 3: LSTM / GRU sequence classifier (PyTorch)
    ensemble.py     -> combine probabilities, gate by regime, size by volatility
    evaluate.py     -> classification metrics + backtest of a signal with the existing engine
    io.py           -> save / load predictions (used by the Streamlit app)
    data_cache.py   -> download OHLCV once, cache to disk
    synthetic.py    -> offline test data with known structure
"""
