import os
import pandas as pd
import matplotlib.pyplot as plt
from data.collector import fetch_price_history
from indicators.indicators import add_sma, add_ema, add_rsi, add_macd, add_bollinger

def setup_plots_dir():
    """Ensure a directory exists to save the sanity check plots."""
    os.makedirs("plots", exist_ok=True)

def test_and_plot_indicator(df_raw, indicator_fn, name, **kwargs):
    """
    Applies an indicator, plots it against price, and saves the result.
    """
    df = indicator_fn(df_raw, **kwargs)

    # Find the indicator columns added
    new_cols = [col for col in df.columns if col not in df_raw.columns]

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # Plot Price
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='black', alpha=0.7)
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Date')
    ax1.legend(loc='upper left')

    # Determine if the indicator needs a separate Y-axis (e.g., RSI, MACD)
    # Indicators like SMA, EMA, BB are in price units.
    # RSI is 0-100. MACD is a difference.
    needs_second_axis = any(col in name or "RSI" in col or "MACD" in col for col in new_cols)

    if needs_second_axis:
        ax2 = ax1.twinx()
        for col in new_cols:
            ax2.plot(df['Date'], df[col], label=col)
        ax2.set_ylabel('Indicator Value')
        ax2.legend(loc='upper right')
    else:
        for col in new_cols:
            ax1.plot(df['Date'], df[col], label=col)
        ax1.legend(loc='upper left')

    plt.title(f"Sanity Check: {name} on AAPL")
    plt.grid(True, alpha=0.3)

    file_path = f"plots/{name.lower().replace(' ', '_')}.png"
    plt.savefig(file_path)
    plt.close()

    print(f"✅ Plot saved to {file_path}")
    return df, new_cols

def main():
    setup_plots_dir()

    print("Fetching data for AAPL...")
    try:
        df_raw = fetch_price_history("AAPL", period="1y", interval="1d")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Define indicators to test
    indicators_to_test = [
        (add_sma, "SMA", {"window": 20}),
        (add_ema, "EMA", {"span": 20}),
        (add_rsi, "RSI", {"window": 14}),
        (add_macd, "MACD", {}),
        (add_bollinger, "Bollinger Bands", {"window": 20}),
    ]

    print("\n--- Running Visual Sanity Checks ---")
    for fn, name, params in indicators_to_test:
        df_res, cols = test_and_plot_indicator(df_raw, fn, name, **params)

        # Edge Case Test: First window rows
        # The window is usually the first param in the dict or a default
        window = params.get("window", params.get("span", 20))
        print(f"Checking edge cases for {name} (first {window} rows):")
        print(df_res[cols[0]].head(window + 1))
        print("-" * 30)

    print("\n--- Visual Sanity Check Guide ---")
    print("1. SMA: Should follow the price but be smoother and lag behind turns.")
    print("2. EMA: Should follow price more closely than SMA, reacting faster to recent moves.")
    print("3. RSI: Should oscillate between 0 and 100. Check for overbought (>70) and oversold (<30).")
    print("4. MACD: MACD Line should cross Signal Line. Histogram should flip sign at crossovers.")
    print("5. BB: Price should mostly stay within the Upper and Lower bands. Mid band is just SMA.")
    print("\nAll plots are available in the 'plots/' directory.")

if __name__ == "__main__":
    main()
