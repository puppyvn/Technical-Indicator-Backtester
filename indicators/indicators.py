"""
indicators/indicators.py

Phase 3 — Technical Indicators
Mỗi hàm là pure function: nhận vào DataFrame OHLCV (đã qua Phase 1-2,
tức là đã clean), trả về DataFrame với (các) cột chỉ báo mới được thêm vào.

Quy ước:
- KHÔNG side effect (không print, không plot, không đọc/ghi file ở đây).
- KHÔNG sửa df gốc trực tiếp -> dùng df = df.copy() ở đầu mỗi hàm.
- Input df bắt buộc có ít nhất cột 'Close' (một số hàm cần thêm 'High'/'Low').
- N dòng đầu (chưa đủ window) sẽ là NaN -> đây là hành vi ĐÚNG, không phải bug.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# 1. SMA — Simple Moving Average
# ---------------------------------------------------------------------------
def add_sma(df: pd.DataFrame, window: int = 20, price_col: str = "Close") -> pd.DataFrame:
    """
    Công thức:
        SMA_t = (P_t + P_{t-1} + ... + P_{t-window+1}) / window

    Diễn giải: trung bình cộng đơn giản của `window` phiên gần nhất.
    - window lớn  -> mượt hơn, phản ứng chậm hơn.
    - window nhỏ  -> nhạy hơn, nhiễu hơn.
    """
    df = df.copy()
    col_name = f"SMA_{window}"
    df[col_name] = df[price_col].rolling(window=window).mean()
    return df


# ---------------------------------------------------------------------------
# 2. EMA — Exponential Moving Average
# ---------------------------------------------------------------------------
def add_ema(df: pd.DataFrame, span: int = 20, price_col: str = "Close") -> pd.DataFrame:
    """
    Công thức (đệ quy):
        alpha = 2 / (span + 1)
        EMA_t = P_t * alpha + EMA_{t-1} * (1 - alpha)
        (EMA_0 thường khởi tạo = P_0, tùy convention)

    Trước khi dùng pandas .ewm(), nên tự implement bằng for-loop một lần
    để hiểu vì sao EMA phản ứng nhanh hơn SMA (trọng số giảm dần theo
    hàm mũ cho các phiên cũ, thay vì trọng số đều nhau như SMA).

    adjust=False: dùng công thức đệ quy chuẩn tài chính (không phải
    weighted average theo kiểu "adjust=True" của pandas, vốn cho kết quả
    khác ở những giá trị đầu tiên).
    """
    df = df.copy()
    col_name = f"EMA_{span}"
    df[col_name] = df[price_col].ewm(span=span, adjust=False).mean()
    return df


def _ema_by_hand(prices: pd.Series, span: int) -> pd.Series:
    """
    Bản tự viết bằng tay (dùng để đối chiếu / học, KHÔNG dùng trong pipeline
    chính thức — chỉ để bạn verify add_ema() ở trên cho ra cùng kết quả).
    """
    alpha = 2 / (span + 1)
    ema_values = []
    ema_prev = None
    for price in prices:
        if ema_prev is None:
            ema_prev = price  # khởi tạo EMA đầu tiên = giá đầu tiên
        else:
            ema_prev = price * alpha + ema_prev * (1 - alpha)
        ema_values.append(ema_prev)
    return pd.Series(ema_values, index=prices.index)


# ---------------------------------------------------------------------------
# 3. RSI — Relative Strength Index
# ---------------------------------------------------------------------------
def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "Close") -> pd.DataFrame:
    """
    Công thức (Wilder's Smoothing):
        delta_t     = P_t - P_{t-1}
        gain_t      = delta_t  if delta_t > 0 else 0
        loss_t      = -delta_t if delta_t < 0 else 0

        avg_gain    = Wilder's Smoothing of gain (alpha = 1/window)
        avg_loss    = Wilder's Smoothing of loss (alpha = 1/window)
        RS          = avg_gain / avg_loss
        RSI         = 100 - (100 / (1 + RS))

    RSI bị chặn trong [0, 100] -> đó là lý do ngưỡng overbought/oversold
    (thường 70/30) có ý nghĩa cố định, khác với SMA (không có thang chuẩn).
    """
    df = df.copy()
    delta = df[price_col].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Sử dụng Wilder's Smoothing (tương đương alpha=1/window trong EWM)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Xử lý trường hợp avg_loss = 0 để tránh chia cho 0 (RSI = 100)
    df[f"RSI_{window}"] = rsi.mask(avg_loss == 0, 100)
    return df


# ---------------------------------------------------------------------------
# 4. MACD — Moving Average Convergence Divergence
# ---------------------------------------------------------------------------
def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Công thức:
        EMA_fast   = EMA(price, span=fast)
        EMA_slow   = EMA(price, span=slow)
        MACD_line  = EMA_fast - EMA_slow
        Signal_line = EMA(MACD_line, span=signal)
        Histogram  = MACD_line - Signal_line

    Diễn giải: MACD_line đo chênh lệch giữa 2 EMA (xu hướng ngắn vs dài).
    Khi MACD_line cắt lên Signal_line -> tín hiệu tăng; cắt xuống -> giảm.
    """
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    df["MACD_line"] = macd_line
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = histogram
    return df


# ---------------------------------------------------------------------------
# 5. Bollinger Bands
# ---------------------------------------------------------------------------
def add_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Công thức:
        Middle_band = SMA(price, window)
        std_t       = rolling std của price trong `window`
        Upper_band  = Middle_band + num_std * std_t
        Lower_band  = Middle_band - num_std * std_t

    Diễn giải: đo độ lệch của giá so với trung bình gần đây theo đơn vị
    độ lệch chuẩn (thống kê) -> giá chạm dải trên/dưới nghĩa là đang lệch
    bất thường so với biến động (volatility) gần đây.
    """
    df = df.copy()
    middle = df[price_col].rolling(window=window).mean()
    std = df[price_col].rolling(window=window).std()

    df[f"BB_mid_{window}"] = middle
    df[f"BB_upper_{window}"] = middle + num_std * std
    df[f"BB_lower_{window}"] = middle - num_std * std
    return df


# ---------------------------------------------------------------------------
# Quick manual sanity-check script (chạy trực tiếp file này để test nhanh)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Testing with real data from the collection layer
    from data.collector import fetch_price_history

    try:
        # Fetch 1 year of daily data for AAPL
        df = fetch_price_history("AAPL", period="1y", interval="1d")
        print(f"Successfully fetched {len(df)} rows of data for AAPL.\n")

        # Apply all five indicators
        df = add_sma(df, window=20)
        df = add_ema(df, span=20)
        df = add_rsi(df, window=14)
        df = add_macd(df)
        df = add_bollinger(df, window=20)

        # 1. Check if all expected columns exist
        expected_cols = ["SMA_20", "EMA_20", "RSI_14", "MACD_line", "MACD_signal", "MACD_hist", "BB_mid_20", "BB_upper_20", "BB_lower_20"]
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if not missing_cols:
            print("✅ All indicator columns were created successfully.")
        else:
            print(f"❌ Missing columns: {missing_cols}")

        # 2. Sanity check for RSI (must be between 0 and 100)
        rsi_values = df["RSI_14"].dropna()
        if not rsi_values.empty and rsi_values.between(0, 100).all():
            print("✅ RSI values are within the valid [0, 100] range.")
        else:
            print("❌ RSI values are out of range or no data available.")

        # 3. Sanity check for Bollinger Bands (Price should be between Lower and Upper bands)
        # We check the last 100 rows to avoid too many NaNs at the start
        tail_df = df.tail(100)
        bb_check = (tail_df["Close"] >= tail_df["BB_lower_20"]) & (tail_df["Close"] <= tail_df["BB_upper_20"])
        # This is not strictly always true if we are using the price at the end of the window,
        # but for a standard BB, the price of the CURRENT bar is what's compared to the bands.
        # Actually, BBs are just a range, the price can be outside.
        # A better check is that Lower < Mid < Upper.
        bb_range_check = (tail_df["BB_lower_20"] < tail_df["BB_mid_20"]) & (tail_df["BB_mid_20"] < tail_df["BB_upper_20"])
        if bb_range_check.all():
            print("✅ Bollinger Band order is correct (Lower < Mid < Upper).")
        else:
            print("❌ Bollinger Band order is incorrect.")

        print("\nLast 5 rows of the resulting DataFrame:")
        print(df.tail())

    except Exception as e:
        print(f"An error occurred during testing: {e}")
