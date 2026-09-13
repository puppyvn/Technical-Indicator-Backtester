import pandas as pd
import numpy as np

"""
backtest/signals.py
Phase 4 — Signal Generation Layer

Trong tài chính, chúng ta cần phân biệt rạch ròi giữa:
1. Signal (Tín hiệu): "Dựa trên dữ liệu hôm nay, tôi NÊN làm gì?"
2. Position (Vị thế): "Ngày mai tôi THỰC SỰ nắm giữ cái gì?"

Sự phân biệt này để tránh 'Lookahead Bias' (Nhìn trước tương lai).
Nếu bạn tính tín hiệu dựa trên giá đóng cửa ngày T, bạn chỉ có thể thực hiện
giao dịch đó vào ngày T+1.
"""

def sma_crossover_signal(df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.DataFrame:
    """
    Chiến lược Giao cắt SMA (Trend Following):
    - Long (1) khi đường trung bình nhanh cắt lên trên đường trung bình chậm.
    - Flat (0) khi ngược lại.
    """
    df = df.copy()

    # Signal: Tính toán dựa trên giá trị hiện tại.
    # Nếu SMA nhanh > SMA chậm => Tín hiệu là 1 (Mua/Giữ)
    df['signal'] = (df[fast_col] > df[slow_col]).astype(int)

    # Position: Dịch chuyển tín hiệu xuống 1 dòng.
    # GIẢI THÍCH: Nếu hôm nay (ngày T) ta thấy SMA nhanh > chậm, ta quyết định mua.
    # Nhưng ta chỉ có thể thực hiện lệnh này sau khi phiên T đóng cửa.
    # Vì vậy, vị thế nắm giữ của ngày T+1 mới là 1.
    # .shift(1) đảm bảo ta không dùng dữ liệu của tương lai để tính lợi nhuận hôm nay.
    df['position'] = df['signal'].shift(1)

    return df

def rsi_mean_reversion_signal(df: pd.DataFrame, rsi_col: str, lower=30, upper=70) -> pd.DataFrame:
    """
    Chiến lược Đảo chiều RSI (Mean Reversion):
    - Long (1) khi RSI < lower (Quá bán - Oversold): Giá quá rẻ, kỳ vọng tăng lại.
    - Flat/Exit (0) khi RSI > upper (Quá mua - Overbought): Giá quá đắt, kỳ vọng giảm.
    """
    df = df.copy()

    # Tạo cột signal khởi tạo bằng NaN
    df['signal'] = np.nan

    # Điều kiện mua: RSI thấp hơn ngưỡng lower
    df.loc[df[rsi_col] < lower, 'signal'] = 1
    # Điều kiện bán/thoát: RSI cao hơn ngưỡng upper
    df.loc[df[rsi_col] > upper, 'signal'] = 0

    # Vì RSI không lúc nào cũng <30 hoặc >70, ta dùng .ffill() (forward fill).
    # Nó có nghĩa là: "Nếu hôm nay không có tín hiệu mới, hãy giữ nguyên vị thế của hôm qua".
    df['signal'] = df['signal'].ffill().fillna(0)

    # Tương tự, shift(1) để tránh Lookahead Bias.
    # Vị thế thực tế ngày T+1 phụ thuộc vào tín hiệu nhìn thấy ngày T.
    df['position'] = df['signal'].shift(1)

    return df

def macd_crossover_signal(df: pd.DataFrame, macd_col: str, signal_col: str) -> pd.DataFrame:
    """
    Chiến lược Giao cắt MACD:
    - Long (1) khi MACD line nằm trên Signal line.
    - Flat (0) khi ngược lại.
    """
    df = df.copy()

    # Signal: 1 nếu MACD > Signal line
    df['signal'] = (df[macd_col] > df[signal_col]).astype(int)

    # Shift(1) để chuyển tín hiệu thành vị thế thực tế cho ngày tiếp theo.
    df['position'] = df['signal'].shift(1)

    return df
