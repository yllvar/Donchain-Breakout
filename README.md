# Donchian Breakout Trading Strategy with Logistic Regression Filter

This repository contains a Python script implementing a Donchian Breakout trading strategy for the BTC/USDT perpetual futures contract on KuCoin, enhanced with a machine learning (ML) filter using logistic regression. The strategy backtests trades based on technical indicators, applies an ML model to refine entries, and visualizes the equity curve.

## Features
- Fetches historical data from KuCoin Futures (15-minute timeframe).
- Implements Donchian Channel breakout with volume, trend, and volatility filters.
- Uses Average True Range (ATR) for stop-loss and take-profit levels.
- Incorporates a logistic regression model trained on past trade data to filter entries.
- Calculates performance metrics (win rate, total profit, CAGR, max drawdown).
- Plots the equity curve using Matplotlib.

## Requirements
- Python 3.11 or later
- Required packages:
  - `ccxt` (for KuCoin API)
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

## Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:yllvar/Donchain-Breakout.git
   cd Donchain-Breakout
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   .venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install ccxt pandas numpy scikit-learn matplotlib
   ```
4. Ensure you have a stable internet connection to fetch data from KuCoin.

## Usage
Run the script directly to perform a backtest:
```bash
python main.py
```
- The script fetches 365 days of 15-minute BTC/USDT data.
- It executes the strategy, prints performance metrics, and displays an equity curve plot.
- Output includes the number of trades for training and performance statistics.

## Configuration
Edit the following variables at the top of `main.py` to customize the strategy:
- `symbol`: Trading pair (default: `'BTC/USDT:USDT'`).
- `timeframe`: Candlestick interval (default: `'15m'`, can be `'1h'`).
- `lookback_period`: Donchian Channel lookback period (default: `20`).
- `atr_period`: ATR calculation period (default: `14`).
- `volume_ma_period`: Volume moving average period (default: `20`).
- `volume_spike_factor`: Volume threshold multiplier (default: `1.5`, e.g., 1.5x moving average).
- `risk_per_trade`: Risk percentage per trade (default: `0.01` or 1%).
- `leverage`: Futures leverage (default: `5`).
- `initial_capital`: Starting capital in USDT (default: `10000`).
- `commission_rate`: Trading commission rate (default: `0.0006` or 0.06%).

## How It Works
1. **Data Fetching**: Retrieves historical OHLCV data using the `ccxt` library.
2. **Indicator Calculation**: Computes Donchian Channels, ATR, volume moving average, and EMAs (50 and 200 periods).
3. **Trade Logic**:
   - Enters long trades when price breaks above the upper Donchian Channel with a volume spike, bullish trend, and momentum.
   - Enters short trades when price breaks below the lower Donchian Channel with a volume spike, bearish trend, and momentum.
   - Uses ATR-based stop-loss (2.5x ATR), take-profit (4x ATR), and trailing stop (1.5x ATR).
4. **ML Filter**: Trains a logistic regression model on past trade features (ATR, volume ratio, momentum) and uses it to filter future entries (probability > 0.7).
5. **Backtesting**: Simulates trades, tracks equity, and calculates performance metrics.
6. **Visualization**: Plots the equity curve over time.

## Performance Metrics
- **Total Trades**: Number of executed trades.
- **Win Rate**: Percentage of profitable trades.
- **Total Profit**: Sum of all trade profits in USDT.
- **Final Equity**: Ending capital after all trades.
- **CAGR**: Compound Annual Growth Rate (based on 15m timeframe annualized).
- **Max Drawdown**: Maximum percentage drop from peak equity.

## Customization
- **Adjust Parameters**: Modify `volume_spike_factor` or `range_threshold` in the `backtest_strategy` function to generate more trades if data is insufficient.
- **Extend Data**: Increase the `limit` in `fetch_kucoin_data` (e.g., to 4000) or adjust the `timedelta` for longer historical data.
- **ML Features**: Add more features (e.g., RSI) to `prepare_training_data` to improve model accuracy.

## Troubleshooting
- **Insufficient Data**: If the ML model fails to train (requires ≥2 trades), check the "Number of trades for training" output. Loosen entry conditions or extend the data period.
- **API Errors**: Ensure KuCoin API access is available and rate limits are not exceeded.
- **Dependencies**: Verify all packages are installed; use `pip show <package>` to check versions.

## License
This project is unlicensed. Feel free to use and modify it for personal or educational purposes.

## Contributing
Contributions are welcome! Submit issues or pull requests to improve the strategy or documentation.


