import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.collector import fetch_price_history
from indicators.indicators import add_sma, add_rsi, add_macd
from backtest.signals import sma_crossover_signal, rsi_mean_reversion_signal, macd_crossover_signal

def run_backtest(df: pd.DataFrame, position_col: str, price_col="Close", fee=0.001) -> pd.DataFrame:
    """
    Hàm cốt lõi để chạy backtest từ vị thế (position).

    Args:
        df: DataFrame chứa giá và cột position (đã shift).
        position_col: Tên cột chứa vị thế (0 hoặc 1).
        price_col: Cột giá để tính return.
        fee: Phí giao dịch mỗi lần thay đổi vị thế (ví dụ 0.001 = 0.1%).
    """
    df = df.copy()

    # 1. Tính daily return của tài sản (Asset Return)
    # Công thức: r_t = (Price_t / Price_{t-1}) - 1
    df['asset_return'] = df[price_col].pct_change()

    # 2. Tính strategy return
    # Lợi nhuận chiến thuật = Vị thế hôm nay * Lợi nhuận của tài sản hôm nay.
    # Nếu position = 1, ta nhận đủ asset_return. Nếu 0, ta nhận 0.
    df['strat_return'] = df[position_col] * df['asset_return']

    # 3. Trừ phí giao dịch (Transaction Costs)
    # Ta chỉ tốn phí khi THAY ĐỔI vị thế (ví dụ: từ 0 lên 1 hoặc 1 xuống 0).
    # .diff().abs() sẽ bằng 1 nếu có sự thay đổi, và 0 nếu giữ nguyên.
    df['turnover'] = df[position_col].diff().abs()
    df['cost'] = df['turnover'] * fee

    # Lợi nhuận ròng = Lợi nhuận chiến thuật - Phí
    df['net_return'] = df['strat_return'] - df['cost']

    # 4. Tính Equity Curve (Đường cong tài sản)
    # GIẢI THÍCH: Tại sao dùng .cumprod() (nhân dồn) mà không dùng .cumsum() (cộng dồn)?
    # Trong tài chính, lợi nhuận là lãi kép (compounding).
    # Ví dụ: Bạn có 100$, lỗ 50% (còn 50$), sau đó lãi 50% (50$ * 1.5 = 75$).
    # Nếu cộng dồn: -50% + 50% = 0% (vẫn 100$) -> SAI.
    # Nếu nhân dồn: (1 - 0.5) * (1 + 0.5) = 0.5 * 1.5 = 0.75 -> ĐÚNG (còn 75$).

    # Xử lý NaN: Các dòng đầu sẽ bị NaN do pct_change() và shift().
    # Ta thay NaN bằng 0 để (1 + 0) = 1, không làm hỏng phép nhân dồn.
    df['net_return'] = df['net_return'].fillna(0)
    df['equity'] = (1 + df['net_return']).cumprod()

    # Baseline: Buy and Hold (Mua và nắm giữ)
    # Chỉ đơn giản là tích lũy lợi nhuận của tài sản mà không thay đổi vị thế.
    df['bh_return'] = df['asset_return'].fillna(0)
    df['bh_equity'] = (1 + df['bh_return']).cumprod()

    return df

def compute_summary_stats(equity_series: pd.Series, periods_per_year=252) -> dict:
    """
    Tính toán các chỉ số đo lường hiệu quả danh mục.
    """
    # Tính daily returns từ equity curve: r_t = (E_t / E_{t-1}) - 1
    returns = equity_series.pct_change().fillna(0)

    # 1. CAGR (Compound Annual Growth Rate) - Tỷ lệ tăng trưởng hàng năm
    # Công thức: CAGR = (Equity_final / Equity_initial) ^ (252 / total_days) - 1
    total_days = len(equity_series)
    final_val = equity_series.iloc[-1]
    initial_val = equity_series.iloc[0]
    cagr = (final_val / initial_val) ** (periods_per_year / total_days) - 1

    # 2. Volatility - Độ biến động (Annualized Standard Deviation)
    # Công thức: Vol = Std(Daily_Returns) * sqrt(252)
    vol = returns.std() * np.sqrt(periods_per_year)

    # 3. Sharpe Ratio - Tỷ lệ lợi nhuận trên rủi ro
    # Công thức: Sharpe = (Mean_Return - RiskFree_Rate) / Std_Return * sqrt(252)
    # Giả định: Risk-free rate = 0 (vì đây là bài tập học tập)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() != 0 else 0

    # 4. Max Drawdown (MDD) - Mức sụt giảm vốn lớn nhất
    # Công thức: Drawdown_t = (Equity_t / Max_Equity_until_t) - 1
    # MDD = min(Drawdown_series)
    running_max = equity_series.cummax()
    drawdown = (equity_series / running_max) - 1
    mdd = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd
    }

if __name__ == "__main__":
    # --- DEMO CHÍNH ---
    print("🚀 Starting Backtest Demo...")
    df = fetch_price_history("AAPL", period="2y", interval="1d")

    # Phase 3: Thêm Indicators
    df = add_sma(df, window=20)
    df = add_sma(df, window=50)
    df = add_rsi(df, window=14)
    df = add_macd(df)

    # Định nghĩa các chiến lược
    strategies = {
        "SMA Crossover": lambda d: sma_crossover_signal(d, "SMA_20", "SMA_50"),
        "RSI Mean Rev": lambda d: rsi_mean_reversion_signal(d, "RSI_14"),
        "MACD Crossover": lambda d: macd_crossover_signal(d, "MACD_line", "MACD_signal"),
    }

    results = {}

    # Chạy backtest cho từng chiến lược
    for name, sig_fn in strategies.items():
        df_sig = sig_fn(df)
        df_bt = run_backtest(df_sig, "position")
        results[name] = compute_summary_stats(df_bt['equity'])

    # Chạy baseline Buy & Hold
    df_bh = df.copy()
    df_bh['position'] = 1
    df_bh_res = run_backtest(df_bh, "position")
    results["Buy & Hold"] = compute_summary_stats(df_bh_res['equity'])

    # In bảng so sánh
    print("\n--- Strategy Comparison ---")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)

    # --- DEMO LOOKAHEAD BIAS ---
    print("\n\n⚠️ Demo: Lookahead Bias")

    # 1. Kết quả ĐÚNG (có shift)
    df_correct = sma_crossover_signal(df, "SMA_20", "SMA_50")
    bt_correct = run_backtest(df_correct, "position")
    sharpe_correct = compute_summary_stats(bt_correct['equity'])['Sharpe Ratio']

    # 2. Kết quả SAI (không có shift)
    # Tạm thời ghi đè hàm signal để bỏ shift
    def sma_biased(d, f, s):
        d = d.copy()
        d['signal'] = (d[f] > d[s]).astype(int)
        d['position'] = d['signal'] # Bỏ .shift(1) -> Dùng tín hiệu hôm nay để trade hôm nay
        return d

    df_biased = sma_biased(df, "SMA_20", "SMA_50")
    bt_biased = run_backtest(df_biased, "position")
    sharpe_biased = compute_summary_stats(bt_biased['equity'])['Sharpe Ratio']

    print(f"Sharpe Ratio (Correct - shifted): {sharpe_correct:.4f}")
    print(f"Sharpe Ratio (Biased - NOT shifted): {sharpe_biased:.4f}")
    print(f"Difference: {sharpe_biased - sharpe_correct:.4f}")
    print("=> Nhận xét: Khi bỏ .shift(1), Sharpe ratio thường tăng vọt vì thuật toán 'biết trước' giá đóng cửa.")
