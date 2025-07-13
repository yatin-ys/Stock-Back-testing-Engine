# Stock-Backtesting-Engine
# Overview
The Stock Backtesting Engine is a Streamlit application designed to help users backtest stock trading strategies using historical data. Users can select one or more stocks, configure their portfolio, and run a backtest on different trading strategies. The app provides detailed performance metrics, including returns, win/loss ratios, Sharpe ratio, maximum drawdown, and more. It also visualizes portfolio performance compared to a buy-and-hold strategy (HODL) and displays buy/sell signals based on selected strategies.

# Features
**Stock Selection:** Choose from a variety of global indices (e.g., Dow Jones, S&P 500, Nifty50).


**Backtest Configuration:**

Set portfolio allocation for each selected stock.

Choose initial capital, investment style (Aggressive, Moderate, Passive), and transaction costs.

Define stop-loss and take-profit percentages.


# **Strategy Options:**


**Moving Average Crossover:** Backtest a strategy based on short-term vs long-term moving averages.

**Bollinger Bands:** Buy when the stock price goes below the lower Bollinger Band and sell when it goes above the upper band.

**Combined Strategy:** A mix of both Moving Average Crossover and Bollinger Bands.

**Performance Metrics:** Evaluate the strategy's performance with metrics like total return, annualized return, Sharpe ratio, and more.

# How to Use
**Step 1:** Select Stocks

In the sidebar, select one or more stocks from the list of global indices.
You can choose from the following options:

    Dow Jones Industrial Average
    
    S&P 500
    
    Nasdaq Composite
    
    Nifty50
    
    Bank Nifty
    
    Sensex
    
    Nikkei 225


**Step 2:**
Configure Backtest

**Portfolio Allocation:** Define the allocation percentage for each selected stock. The total must sum to 100%.

**Initial Capital:** Set the initial capital for your portfolio.

**Start and End Date:** Specify the backtest period (e.g., "2023-01-01" to "2024-01-01").


# Investment Style: 

Choose between Aggressive, Moderate, or Passive styles. This affects the Bollinger Bands strategy's parameters.

**Strategy Type:** Select one of the following strategies:

    Moving Average Crossover
        
    Bollinger Bands
        
    Combined

**Transaction Costs:** Define the percentage of transaction costs (default is 0.1%).

**Stop Loss:** Set the stop loss percentage (default is 5%).

**Take Profit:** Set the take profit percentage (default is 10%).


**Step 3:** Run Backtest

Once the configuration is complete, click Run Backtest.

The app will fetch historical stock data, apply the selected strategy, and calculate portfolio performance.


**Step 4: Review Results**

The results are displayed for each stock individually, followed by a combined summary if multiple stocks were selected. The output includes:

**Strategy Performance:** Final portfolio value, total return, Sharpe ratio, and other performance metrics for each stock.

**Combined Portfolio Performance:** A summary of the total portfolio value and return across all selected stocks.

**Trade History:** A detailed table of all trades executed during the backtest for each stock.

**Visualizations:** Interactive charts to analyze performance (see "Visualizations" section below).

# Performance Metrics

**Final Portfolio Value:** The total value of your portfolio at the end of the backtest.

**Total Return:** The overall return percentage for the strategy.

**Sharpe Ratio:** Measures the risk-adjusted return of the strategy.

**Max Drawdown:** The maximum loss from a peak to a trough during the backtest period.

**Annualized Return:** The yearly return based on the total return over the backtest period.

**Total Trades:** The total number of trades executed.

**Winning Trades:** The number of profitable trades.

**Losing Trades:** The number of unprofitable trades.

**Win/Loss Ratio:** The ratio of profitable to unprofitable trades.

**Average Profit per Trade:** The average percentage profit for each trade executed.

# Visualizations

The application provides several interactive charts to help you analyze the backtest results:

- **Portfolio Value vs. HODL**: Compares the growth of your initial capital using the selected strategy versus a simple buy-and-hold approach.
- **Buy/Sell Signals**: Displays the stock's price chart along with the strategy indicators (Moving Averages or Bollinger Bands) and marks the exact points where buy and sell trades were executed.
- **Sharpe Ratio by Stock**: A bar chart comparing the Sharpe ratio of each stock in the portfolio, which is useful for evaluating risk-adjusted performance across different assets.
- **Final Value per Stock**: Shows the final value contributed by each stock's backtest to the total portfolio, illustrating which assets were the top performers.

# Notes

The app currently supports stock data from Yahoo Finance (via `yfinance`), so historical data may not be available for all tickers or may be limited based on the stock.

Make sure the total allocation for your portfolio sums to 100%. The app will notify you if the allocation is incorrect.

The backtest duration should be at least a few months to generate meaningful results, especially for strategies like moving averages or Bollinger Bands.
