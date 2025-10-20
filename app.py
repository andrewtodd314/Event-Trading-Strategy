# app.py
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Load your trading dataset here
# ----------------------------
# Ensure CSV has columns:
# 'ticker', 'sentiment', 'entry_date', 'exit_date', 'entry_price', 'exit_price'
trading_df = pd.read_csv("your_trading_dataset.csv")

# Ensure date columns are datetime
trading_df['entry_date'] = pd.to_datetime(trading_df['entry_date'])
trading_df['exit_date'] = pd.to_datetime(trading_df['exit_date'])

# ----------------------------
# Define strategy function
# ----------------------------
def run_strategy(tickers, lower, upper, start_date, end_date):
    df = trading_df[
        (trading_df['ticker'].isin(tickers)) &
        (trading_df['entry_date'] >= pd.to_datetime(start_date)) &
        (trading_df['exit_date'] <= pd.to_datetime(end_date))
    ].copy()
    
    if df.empty:
        return None, "No trades for these parameters.", None

    # Generate long/short/no-trade signals
    df['signal'] = df['sentiment'].apply(
        lambda x: 1 if x >= upper else (-1 if x <= lower else 0)
    )
    
    # Filter only trades with active signals
    df = df[df['signal'] != 0].copy()
    
    if df.empty:
        return None, "No trades after applying thresholds.", None

    # Calculate returns based on signal
    transaction_cost = 0.001  # 0.1% per trade

    def compute_return(row):
        if row['signal'] == 1:  # Long
            r = (row['exit_price'] - row['entry_price']) / row['entry_price']
        elif row['signal'] == -1:  # Short
            r = (row['entry_price'] - row['exit_price']) / row['entry_price']
        else:
            r = 0
        return r - 2 * transaction_cost

    df['return_after_cost'] = df.apply(compute_return, axis=1)
    
    # Cumulative return
    df = df.sort_values('entry_date')
    df['cumulative_return'] = (1 + df['return_after_cost']).cumprod() - 1

    # Key metrics
    total_return = df['cumulative_return'].iloc[-1]
    hit_rate = (df['return_after_cost'] > 0).mean()
    annualized_sharpe = df['return_after_cost'].mean() / df['return_after_cost'].std() * np.sqrt(252) if df['return_after_cost'].std() > 0 else np.nan

    metrics_text = f"""
Total Return: {total_return:.2%}
Hit Rate: {hit_rate:.2%}
Annualized Sharpe: {annualized_sharpe:.2f}
"""

    # Equity curve plot
    plt.figure(figsize=(10,5))
    plt.plot(df['entry_date'], df['cumulative_return'], marker='o', color='blue')
    plt.xlabel("Entry Date")
    plt.ylabel("Cumulative Return")
    plt.title("Earnings Call Sentiment Strategy Equity Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.close()  # prevent auto-display

    # Top-performing tickers
    top_tickers = df.groupby('ticker')['return_after_cost'].sum().sort_values(ascending=False).reset_index()

    return plt, metrics_text, top_tickers.head(10)

# ----------------------------
# Define Gradio interface
# ----------------------------
interface = gr.Interface(
    fn=run_strategy,
    inputs=[
        gr.Dropdown(trading_df['ticker'].unique().tolist(), label="Select Tickers", multiselect=True),
        gr.Slider(-1, 0.0, 0.01, label="Lower Threshold (Short)"),
        gr.Slider(0.0, 1.0, 0.01, label="Upper Threshold (Long)"),
        gr.Date(label="Start Date"),
        gr.Date(label="End Date")
    ],
    outputs=[
        gr.Plot(label="Equity Curve"),
        gr.Textbox(label="Strategy Metrics"),
        gr.Dataframe(label="Top Performing Tickers")
    ],
    title="Earnings Call Sentiment Strategy Backtest",
    description="Interactive backtest with long/short/no-trade signals. Adjust thresholds and date range to explore performance."
)

# ----------------------------
# Launch the app
# ----------------------------
interface.launch(share=True)
