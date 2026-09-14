DEFAULT_TICKER = "AAPL"

TICKER_PRESETS = {
    "Mega Cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
    "Dow Industrials Sample": ["JPM", "UNH", "HD", "GS", "CAT", "MCD"],
    "Popular ETFs": ["SPY", "QQQ", "VTI", "IWM", "DIA"],
}

PERIOD_OPTIONS = ["6mo", "1y", "2y", "5y", "10y", "max"]
INTERVAL_OPTIONS = ["1d", "1wk", "1mo"]

DEFAULT_PERIOD = '2y'
DEFAULT_INTERVAL = "1d"

SMA_SHORT_DEFAULT = 20
SMA_LONG_DEFAULT = 50
EMA_SHORT_DEFAULT = 12
EMA_LONG_DEFAULT = 65
RSI_PERIOD_DEFAULT = 14
RSI_OVERBOUGHT = 70
RSI_OVERFSOLD = 30
MACD_SIGNAL_DEFAULT = 9
BOLLINGER_WINDOW_DEFAULT = 20
BOLLINGER_STD_DEFAULT = 2

INITIAL_CAPITAL = 10_000
TRADING_FEE_BPS = 5 # 5 basis points per trade, applied to simulate realistic costs

CACHE_TTL_SECONDS = 3600 # 1 hour - avoids hammering yfinace on every rerun
