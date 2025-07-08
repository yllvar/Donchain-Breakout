import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
symbol = 'BTC/USDT:USDT'  # KuCoin Futures BTC/USDT perpetual contract
timeframe = '15m'  # '15m' or '1h'
lookback_period = 20  # Donchian Channel lookback period
atr_period = 14  # ATR period
volume_ma_period = 20  # Volume moving average period
volume_spike_factor = 1.5  # Volume must be 1.5x the moving average
risk_per_trade = 0.01  # Risk 1% of equity per trade
leverage = 5  # Futures leverage
initial_capital = 10000  # Starting capital in USDT
commission_rate = 0.0006  # KuCoin Futures commission (0.06% per trade)

# Fetch historical data from KuCoin Futures
def fetch_kucoin_data(symbol, timeframe, limit=2000):
    exchange = ccxt.kucoinfutures()
    since_timestamp = int(
        (datetime.now() - timedelta(days=365)).timestamp() * 1000
    )  # Fetch 365 days of data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Calculate indicators
def calculate_indicators(df, lookback_period, atr_period, volume_ma_period):
    # Donchian Channel
    df['upper_donchian'] = df['high'].rolling(window=lookback_period).max().shift(1)
    df['lower_donchian'] = df['low'].rolling(window=lookback_period).min().shift(1)
    df['middle_donchian'] = (df['upper_donchian'] + df['lower_donchian']) / 2

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=atr_period).mean()

    # Volume Moving Average
    df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
    return df

# Backtesting logic
# Function to prepare training data from past trades
def prepare_training_data(df, trades):
    features = []
    labels = []
    for trade in trades:
        idx_entry = df.index[df['timestamp'] == trade['entry_time']].tolist()[0]
        idx_exit = df.index[df['timestamp'] == trade['exit_time']].tolist()[0]
        if idx_entry > 0 and idx_exit > idx_entry:
            atr = df['atr'].iloc[idx_entry]
            volume_ratio = df['volume'].iloc[idx_entry] / df['volume_ma'].iloc[idx_entry]
            momentum = (df['close'].iloc[idx_entry] - df['open'].iloc[idx_entry]) / df['open'].iloc[idx_entry]
            features.append([atr, volume_ratio, momentum])
            # Use actual profit from the trade
            labels.append(1 if trade['profit'] > 0 else 0)
    return np.array(features), np.array(labels)

# Function to train and predict with ML model
def train_ml_model(features, labels):
    if len(features) < 2 or len(labels) < 2:  # Need at least 2 samples to train
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Updated backtest_strategy with ML filter
def backtest_strategy(df, risk_per_trade, leverage, initial_capital):
    df = calculate_indicators(df, lookback_period=50, atr_period=14, volume_ma_period=20)
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trailing_stop = 0
    equity = initial_capital
    trades = []
    equity_curve = [initial_capital]  # Start with initial equity

    # Initial run to generate trades for training
    for i in range(1, len(df)):  # Iterate up to len(df) - 1 to avoid index issues
        volume_spike = df['volume'].iloc[i] > df['volume_ma'].iloc[i] * 1.5  # Reduced from 1.8
        trend_long = df['ema50'].iloc[i] > df['ema200'].iloc[i]
        trend_short = df['ema50'].iloc[i] < df['ema200'].iloc[i]
        range_threshold = (df['upper_donchian'].iloc[i] - df['lower_donchian'].iloc[i]) / df['close'].iloc[i] > 0.005  # Reduced from 0.01
        volatility_filter = df['atr'].iloc[i] / df['close'].iloc[i] > 0.005

        if position == 0 and range_threshold and volatility_filter:
            if (
                df['close'].iloc[i] > df['upper_donchian'].iloc[i]
                and volume_spike
                and trend_long
                and df['close'].iloc[i] > df['open'].iloc[i] * 1.005
            ):
                entry_price = df['close'].iloc[i]
                atr = df['atr'].iloc[i]
                stop_loss = entry_price - 2.5 * atr
                take_profit = entry_price + 4 * atr
                trailing_stop = entry_price - 1.5 * atr
                risk_amount = equity * 0.005
                stop_distance = entry_price - stop_loss
                position_size = (risk_amount / stop_distance) * leverage
                position = 1
                trades.append({
                    'entry_time': df['timestamp'].iloc[i],
                    'type': 'long',
                    'entry_price': entry_price,
                    'position_size': position_size
                })
            elif (
                df['close'].iloc[i] < df['lower_donchian'].iloc[i]
                and volume_spike
                and trend_short
                and df['close'].iloc[i] < df['open'].iloc[i] * 0.995
            ):
                entry_price = df['close'].iloc[i]
                atr = df['atr'].iloc[i]
                stop_loss = entry_price + 2.5 * atr
                take_profit = entry_price - 4 * atr
                trailing_stop = entry_price + 1.5 * atr
                risk_amount = equity * 0.005
                stop_distance = stop_loss - entry_price
                position_size = (risk_amount / stop_distance) * leverage
                position = -1
                trades.append({
                    'entry_time': df['timestamp'].iloc[i],
                    'type': 'short',
                    'entry_price': entry_price,
                    'position_size': position_size
                })

        if position == 1:
            current_trailing = max(trailing_stop, df['low'].rolling(window=2).min().iloc[i])
            if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit or df['low'].iloc[i] <= current_trailing:
                exit_price = min(max(stop_loss, current_trailing), take_profit) if df['low'].iloc[i] <= current_trailing else (take_profit if df['high'].iloc[i] >= take_profit else df['low'].iloc[i])
                profit = (exit_price - entry_price) * position_size - (entry_price + exit_price) * position_size * commission_rate
                equity += profit
                trades[-1].update({'exit_time': df['timestamp'].iloc[i], 'exit_price': exit_price, 'profit': profit})
                position = 0
        elif position == -1:
            current_trailing = min(trailing_stop, df['high'].rolling(window=2).max().iloc[i])
            if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit or df['high'].iloc[i] >= current_trailing:
                exit_price = max(min(stop_loss, current_trailing), take_profit) if df['high'].iloc[i] >= current_trailing else (take_profit if df['low'].iloc[i] <= take_profit else df['high'].iloc[i])
                profit = (entry_price - exit_price) * position_size - (entry_price + exit_price) * position_size * commission_rate
                equity += profit
                trades[-1].update({'exit_time': df['timestamp'].iloc[i], 'exit_price': exit_price, 'profit': profit})
                position = 0

        equity_curve.append(equity)  # Append equity after each iteration

    print(f"Number of trades for training: {len(trades)}")  # Debug print
    # Train ML model
    features, labels = prepare_training_data(df, trades)
    model, scaler = train_ml_model(features, labels)
    if model is None:
        print("Insufficient data to train ML model. Proceeding without filter.")
        new_trades = trades  # Fallback to original trades
    else:
        # Re-run with ML filter
        position = 0
        equity = initial_capital
        new_trades = []
        equity_curve = [initial_capital]  # Reset for second pass

        for i in range(1, len(df)):
            volume_spike = df['volume'].iloc[i] > df['volume_ma'].iloc[i] * 1.5
            trend_long = df['ema50'].iloc[i] > df['ema200'].iloc[i]
            trend_short = df['ema50'].iloc[i] < df['ema200'].iloc[i]
            range_threshold = (df['upper_donchian'].iloc[i] - df['lower_donchian'].iloc[i]) / df['close'].iloc[i] > 0.005
            volatility_filter = df['atr'].iloc[i] / df['close'].iloc[i] > 0.005

            if position == 0 and range_threshold and volatility_filter:
                atr = df['atr'].iloc[i]
                volume_ratio = df['volume'].iloc[i] / df['volume_ma'].iloc[i]
                momentum = (df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i]
                feature = scaler.transform([[atr, volume_ratio, momentum]])
                probability = model.predict_proba(feature)[0][1] if model else 1.0

                if (
                    df['close'].iloc[i] > df['upper_donchian'].iloc[i]
                    and volume_spike
                    and trend_long
                    and df['close'].iloc[i] > df['open'].iloc[i] * 1.005
                    and probability > 0.7
                ):
                    entry_price = df['close'].iloc[i]
                    atr = df['atr'].iloc[i]
                    stop_loss = entry_price - 2.5 * atr
                    take_profit = entry_price + 4 * atr
                    trailing_stop = entry_price - 1.5 * atr
                    risk_amount = equity * 0.005
                    stop_distance = entry_price - stop_loss
                    position_size = (risk_amount / stop_distance) * leverage
                    position = 1
                    new_trades.append({
                        'entry_time': df['timestamp'].iloc[i],
                        'type': 'long',
                        'entry_price': entry_price,
                        'position_size': position_size
                    })
                elif (
                    df['close'].iloc[i] < df['lower_donchian'].iloc[i]
                    and volume_spike
                    and trend_short
                    and df['close'].iloc[i] < df['open'].iloc[i] * 0.995
                    and probability > 0.7
                ):
                    entry_price = df['close'].iloc[i]
                    atr = df['atr'].iloc[i]
                    stop_loss = entry_price + 2.5 * atr
                    take_profit = entry_price - 4 * atr
                    trailing_stop = entry_price + 1.5 * atr
                    risk_amount = equity * 0.005
                    stop_distance = stop_loss - entry_price
                    position_size = (risk_amount / stop_distance) * leverage
                    position = -1
                    new_trades.append({
                        'entry_time': df['timestamp'].iloc[i],
                        'type': 'short',
                        'entry_price': entry_price,
                        'position_size': position_size
                    })

            if position == 1:
                current_trailing = max(trailing_stop, df['low'].rolling(window=2).min().iloc[i])
                if df['low'].iloc[i] <= stop_loss or df['high'].iloc[i] >= take_profit or df['low'].iloc[i] <= current_trailing:
                    exit_price = min(max(stop_loss, current_trailing), take_profit) if df['low'].iloc[i] <= current_trailing else (take_profit if df['high'].iloc[i] >= take_profit else df['low'].iloc[i])
                    profit = (exit_price - entry_price) * position_size - (entry_price + exit_price) * position_size * commission_rate
                    equity += profit
                    new_trades[-1].update({'exit_time': df['timestamp'].iloc[i], 'exit_price': exit_price, 'profit': profit})
                    position = 0
            elif position == -1:
                current_trailing = min(trailing_stop, df['high'].rolling(window=2).max().iloc[i])
                if df['high'].iloc[i] >= stop_loss or df['low'].iloc[i] <= take_profit or df['high'].iloc[i] >= current_trailing:
                    exit_price = max(min(stop_loss, current_trailing), take_profit) if df['high'].iloc[i] >= current_trailing else (take_profit if df['low'].iloc[i] <= take_profit else df['high'].iloc[i])
                    profit = (entry_price - exit_price) * position_size - (entry_price + exit_price) * position_size * commission_rate
                    equity += profit
                    new_trades[-1].update({'exit_time': df['timestamp'].iloc[i], 'exit_price': exit_price, 'profit': profit})
                    position = 0

            equity_curve.append(equity)

    df['equity'] = equity_curve
    return df, new_trades

# Performance metrics
def calculate_performance(trades, df, initial_capital):
    total_trades = len(trades)
    wins = len([t for t in trades if t['profit'] > 0])
    win_rate = wins / total_trades if total_trades > 0 else 0
    total_profit = sum(t['profit'] for t in trades)
    final_equity = df['equity'].iloc[-1]
    cagr = ((final_equity / initial_capital) ** (1 / (len(df) / (24 * 4 * 30))) - 1) if final_equity > 0 else 0
    drawdowns = (df['equity'] / df['equity'].cummax() - 1) * 100
    max_drawdown = drawdowns.min()

    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total Profit: {total_profit:.2f} USDT")
    print(f"Final Equity: {final_equity:.2f} USDT")
    print(f"CAGR: {cagr:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")

# Run backtest
df = fetch_kucoin_data(symbol, timeframe)
df, trades = backtest_strategy(df, risk_per_trade, leverage, initial_capital)
calculate_performance(trades, df, initial_capital)

# Plot equity curve
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['equity'], label='Equity Curve')
plt.title(f'Equity Curve for {symbol} ({timeframe})')
plt.xlabel('Date')
plt.ylabel('Equity (USDT)')
plt.legend()
plt.show()