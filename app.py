# app.py
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load trading dataset
trading_df = pd.read_csv("SP100-trading-dataset.csv")
trading_df['entry_date'] = pd.to_datetime(trading_df['entry_date'])
trading_df['exit_date'] = pd.to_datetime(trading_df['exit_date'])

# Strategy function
def run_strategy(tickers, lower, upper, start_date, end_date):
    # Convert strings to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    df = trading_df[
        (trading_df['ticker'].isin(tickers)) &
        (trading_df['entry_date'] >= start_date) &
        (trading_df['exit_date'] <= end_date)
    ].copy()
    
    if df.empty:
        return None, "0%", "0%", "0", None, None

    # Long/short/no-trade signals
    df['signal'] = df['sentiment'].apply(
        lambda x: 1 if x >= upper else (-1 if x <= lower else 0)
    )
    
    df = df[df['signal'] != 0].copy()
    if df.empty:
        return None, "0%", "0%", "0", None, None

    # Compute returns
    transaction_cost = 0.001
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
    annualized_sharpe = (
        df['return_after_cost'].mean() / df['return_after_cost'].std() * np.sqrt(252)
        if df['return_after_cost'].std() > 0 else np.nan
    )

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['entry_date'], df['cumulative_return'], marker='o', color='blue')
    ax.set_xlabel("Entry Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Earnings Call Sentiment Strategy Equity Curve")
    ax.grid(True)
    plt.tight_layout()

    # Top-performing tickers
    top_tickers = df.groupby('ticker')['return_after_cost'].sum().sort_values(ascending=False).reset_index()

    return (
        fig,
        f"{total_return:.2%}",
        f"{hit_rate:.2%}",
        f"{annualized_sharpe:.2f}",
        top_tickers.head(10)
    )

# Build Gradio interface
with gr.Blocks(title="Earnings Call Sentiment Strategy Backtest") as app:
    gr.Markdown("## Earnings Call Sentiment Strategy Backtest")
    gr.Markdown("Adjust thresholds, select tickers, and define the backtest period to explore trading performance ⚠️ Please select dates between 2014-01-01 and 2020-01-01")

    with gr.Row():
        tickers_input = gr.Dropdown(trading_df['ticker'].unique().tolist(), label="Select Tickers", multiselect=True)
        lower_input = gr.Slider(-1.0, 1.0, value=-0.5, step=0.05, label="Lower Threshold (Short)")
        upper_input = gr.Slider(-1.0, 1.0, value=0.5, step=0.05, label="Upper Threshold (Long)")
        start_input = gr.Textbox(label="Start Date (YYYY-MM-DD)")
        end_input = gr.Textbox(label="End Date (YYYY-MM-DD)")

    run_button = gr.Button("Run Backtest 🚀")

    with gr.Row():
        total_return_box = gr.Textbox(label="Total Return", interactive=False)
        hit_rate_box = gr.Textbox(label="Hit Rate", interactive=False)
        sharpe_box = gr.Textbox(label="Annualized Sharpe", interactive=False)

    equity_plot = gr.Plot(label="Equity Curve")
    top_tickers_table = gr.Dataframe(label="Top Performing Tickers")

    run_button.click(
        fn=run_strategy,
        inputs=[tickers_input, lower_input, upper_input, start_input, end_input],
        outputs=[equity_plot, total_return_box, hit_rate_box, sharpe_box, top_tickers_table]
    )

app.launch(share=True)
