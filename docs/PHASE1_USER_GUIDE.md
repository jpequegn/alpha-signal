# Phase 1 User Guide - Trading Indicators

A comprehensive guide to using AlphaSignal Phase 1 indicators for technical trading analysis.

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Indicator Overview](#indicator-overview)
- [Usage Examples](#usage-examples)
- [Signal Generation](#signal-generation)
- [Common Strategies](#common-strategies)
- [Best Practices](#best-practices)
- [FAQ](#faq)

## Introduction

Phase 1 of AlphaSignal implements **5 fundamental trading indicators** that cover trend-following, momentum, and volatility analysis. These indicators are built from first principles using NumPy, providing transparent and efficient price analysis.

**Key Principles:**
- Built from scratch, not using black-box libraries
- Fully documented mathematical foundations
- Comprehensive test coverage (90%+)
- Production-ready code

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alpha-signal.git
cd alpha-signal

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from src.indicators import SMA, EMA, RSI, MACD, BollingerBands

# Create sample price data
prices = np.array([100, 101, 102, 101, 103, 104, 105, 104, 106, 107])

# Calculate indicators
sma = SMA(period=3)
ema = EMA(period=3)
rsi = RSI(period=3)

sma_values = sma(prices)
ema_values = ema(prices)
rsi_values = rsi(prices)

print("SMA:", sma_values)
print("EMA:", ema_values)
print("RSI:", rsi_values)
```

## Indicator Overview

### 1. Simple Moving Average (SMA)

**Purpose:** Smooths price data to identify trends by averaging prices over a period.

**Formula:** `SMA(t) = (P(t) + P(t-1) + ... + P(t-n+1)) / n`

**Key Characteristics:**
- Gives equal weight to all prices
- Slower to respond to changes
- Better for identifying long-term trends
- Less noise than raw prices

**Common Periods:**
- **20-period**: Short-term trend (4 weeks of daily data)
- **50-period**: Medium-term trend (10 weeks)
- **200-period**: Long-term trend (40 weeks)

**Interpretation:**
- Price above SMA: Uptrend
- Price below SMA: Downtrend
- SMA slope: Trend strength

---

### 2. Exponential Moving Average (EMA)

**Purpose:** Weighted moving average that emphasizes recent prices.

**Formula:** `EMA(t) = Price(t) × α + EMA(t-1) × (1-α)` where `α = 2/(period+1)`

**Key Characteristics:**
- Responds faster to price changes than SMA
- Higher alpha (α) = more responsive
- Reduces lag in trend identification
- Good for short-term trading signals

**Common Periods:**
- **12-period**: Fast EMA (used in MACD)
- **26-period**: Slow EMA (used in MACD)
- **5-period**: Ultra-responsive for scalping

**Interpretation:**
- Price above EMA: Bullish bias
- Price below EMA: Bearish bias
- EMA slope angle: Trend momentum

**EMA vs SMA:**
```
High volatility period:
- SMA: Smooth, lags behind
- EMA: Follows more closely, earlier entry/exit signals
```

---

### 3. Relative Strength Index (RSI)

**Purpose:** Momentum oscillator measuring overbought/oversold conditions (0-100 scale).

**Formula:**
```
RSI = 100 - (100 / (1 + RS))
RS = Avg Gain / Avg Loss
```

**Key Characteristics:**
- Bounded between 0 and 100
- Measures momentum, not direction
- Based on Wilder's smoothing technique
- Good for identifying reversals

**Signal Levels:**
- **RSI > 70**: Overbought (potential sell signal)
- **RSI < 30**: Oversold (potential buy signal)
- **RSI = 50**: Neutral momentum
- **RSI 40-60**: Consolidation/sideways market

**Interpretation:**
- Divergences: Price makes new high but RSI doesn't = reversal signal
- Failure swings: RSI breaks support/resistance
- Extreme readings: Often lead to pullbacks

**Example:**
```
Uptrend with rising prices but RSI declining
= Weakening momentum = Potential reversal
```

---

### 4. MACD (Moving Average Convergence Divergence)

**Purpose:** Trend-following indicator combining two EMAs showing momentum and direction.

**Components:**
- **MACD Line**: EMA(12) - EMA(26)
- **Signal Line**: EMA(9) of MACD
- **Histogram**: MACD - Signal

**Key Characteristics:**
- Demonstrates indicator composition
- Good for trend confirmation
- Shows momentum acceleration/deceleration
- Useful for exit signals

**Trading Signals:**
- **MACD > Signal**: Bullish (buy signal)
- **MACD < Signal**: Bearish (sell signal)
- **Histogram positive & growing**: Strong uptrend
- **Histogram negative & growing**: Strong downtrend
- **Histogram shrinking**: Momentum weakening

**Divergence Trading:**
```
Bearish Divergence: Price makes new high, MACD doesn't
= Weakening momentum = Potential top
```

---

### 5. Bollinger Bands

**Purpose:** Volatility indicator showing upper/lower price bands for mean reversion trading.

**Components:**
- **Middle Band**: SMA(20)
- **Upper Band**: Middle + (2 × StdDev)
- **Lower Band**: Middle - (2 × StdDev)

**Key Characteristics:**
- ~95% of prices fall within bands
- Band width measures volatility
- Good for mean reversion strategies
- Opposite approach to trend-following

**Band Interpretation:**
- **Narrow bands**: Low volatility, consolidation
- **Wide bands**: High volatility, strong trend
- **Price near upper band**: Overbought
- **Price near lower band**: Oversold

**The Squeeze Setup:**
```
1. Bands narrow (low volatility consolidation)
2. Bands widen (breakout begins)
3. Trade in breakout direction
```

**Bandwidth Percentage:**
```
Bandwidth % = (Upper - Lower) / Middle × 100
Measures relative volatility (comparable across price levels)
```

## Usage Examples

### Example 1: Trend Identification with SMA

```python
import numpy as np
from src.indicators import SMA

# Daily closing prices over 50 days
prices = np.random.normal(100, 2, 50)

# Calculate moving averages
sma_fast = SMA(period=10)  # Short-term trend
sma_slow = SMA(period=20)  # Medium-term trend

fast_values = sma_fast(prices)
slow_values = sma_slow(prices)

# Find crossover signals
for i in range(1, len(prices)):
    # Check if fast SMA crossed above slow SMA
    if fast_values[i-1] < slow_values[i-1] and fast_values[i] > slow_values[i]:
        print(f"Day {i}: BULLISH crossover at price {prices[i]:.2f}")

    # Check if fast SMA crossed below slow SMA
    elif fast_values[i-1] > slow_values[i-1] and fast_values[i] < slow_values[i]:
        print(f"Day {i}: BEARISH crossover at price {prices[i]:.2f}")
```

### Example 2: RSI Mean Reversion

```python
from src.indicators import RSI

prices = np.random.normal(100, 2, 100)
rsi = RSI(period=14)
rsi_values = rsi(prices)

# Generate signals
signals = rsi.get_signals(rsi_values, overbought=70, oversold=30)

# Identify trading opportunities
for i in range(len(prices)):
    if signals[i] == -1:
        print(f"Day {i}: OVERSOLD (RSI={rsi_values[i]:.1f}) - BUY signal at {prices[i]:.2f}")
    elif signals[i] == 1:
        print(f"Day {i}: OVERBOUGHT (RSI={rsi_values[i]:.1f}) - SELL signal at {prices[i]:.2f}")
```

### Example 3: MACD Crossover Strategy

```python
from src.indicators import MACD

prices = np.random.normal(100, 2, 100)
macd = MACD()  # Default: fast=12, slow=26, signal=9
macd_line, signal_line, histogram = macd(prices)

signals = macd.get_signals(macd_line, signal_line)

# Identify crossover signals
for i in range(1, len(prices)):
    if signals[i-1] == 0 and signals[i] == 1:
        print(f"Day {i}: MACD bullish crossover at {prices[i]:.2f}")
    elif signals[i-1] == 0 and signals[i] == -1:
        print(f"Day {i}: MACD bearish crossover at {prices[i]:.2f}")
```

### Example 4: Bollinger Bands Volatility Analysis

```python
from src.indicators import BollingerBands

prices = np.random.normal(100, 2, 100)
bb = BollingerBands(period=20, num_std=2)
upper, middle, lower = bb(prices)

# Calculate bandwidth percentage
bandwidth_pct = bb.get_bandwidth_percent(upper, middle, lower)

# Generate signals
signals = bb.get_signals(prices, upper, lower)

# Analyze volatility
for i in range(len(prices)):
    if not np.isnan(bandwidth_pct[i]):
        if bandwidth_pct[i] < 10:
            print(f"Day {i}: SQUEEZE (BW%={bandwidth_pct[i]:.1f}%) - Breakout likely")
        elif bandwidth_pct[i] > 30:
            print(f"Day {i}: HIGH VOLATILITY (BW%={bandwidth_pct[i]:.1f}%)")

# Check band touches
for i in range(len(prices)):
    if signals[i] == 1:
        print(f"Day {i}: Price {prices[i]:.2f} touches UPPER band - Overbought")
    elif signals[i] == -1:
        print(f"Day {i}: Price {prices[i]:.2f} touches LOWER band - Oversold")
```

## Signal Generation

Each indicator provides `get_signals()` method for automated signal generation:

```python
# RSI signals: -1 (oversold), 0 (neutral), 1 (overbought)
rsi_signals = rsi.get_signals(rsi_values, overbought=70, oversold=30)

# MACD signals: -1 (bearish), 0 (neutral), 1 (bullish)
macd_signals = macd.get_signals(macd_line, signal_line)

# Bollinger Bands signals: -1 (lower band), 0 (neutral), 1 (upper band)
bb_signals = bb.get_signals(prices, upper, lower)
```

## Common Strategies

### Strategy 1: Trend Following with EMA

**Setup:**
- Fast EMA (5-period) crosses above Slow EMA (20-period) = Buy signal
- Fast EMA crosses below Slow EMA = Sell signal

**Advantage:** Simple, responds quickly to trends
**Risk:** Whipsaws in sideways markets

### Strategy 2: RSI Mean Reversion

**Setup:**
- RSI < 30 for N bars = Oversold, potential bounce
- Buy on oversold, sell when RSI > 50

**Advantage:** Catches reversals, good risk/reward
**Risk:** Can continue falling in strong downtrends

### Strategy 3: MACD Trend Confirmation

**Setup:**
1. Price above 200-period SMA (uptrend)
2. MACD > Signal (confirm uptrend)
3. Enter long position

**Advantage:** Higher probability trades
**Risk:** Slower to enter than pure momentum

### Strategy 4: Bollinger Bands Squeeze Breakout

**Setup:**
1. Bandwidth drops below 10% (squeeze)
2. Wait for price break above upper band
3. Buy/Sell in breakout direction

**Advantage:** Captures large moves after consolidation
**Risk:** False breakouts during ranging markets

## Best Practices

### 1. Data Quality
```python
# Ensure clean price data
assert not np.any(np.isnan(prices)), "Prices contain NaN values"
assert np.all(prices > 0), "Prices must be positive"
assert len(prices) >= 100, "Need sufficient data for analysis"
```

### 2. Period Selection

**Timeframe Mapping:**
- **1-minute chart**: Period 5-20
- **5-minute chart**: Period 5-20
- **Daily chart**: Period 10-50
- **Weekly chart**: Period 10-26

**Rule of thumb:** Period = (number of bars in desired lookback) / 2

### 3. Multi-Indicator Confirmation

```python
# Don't trade on single indicator
# Confirm with multiple indicators:
- EMA for trend direction
- RSI for momentum confirmation
- MACD for entry timing
- Volume for strength confirmation
```

### 4. Risk Management

```python
# Always define stop-loss and take-profit
stop_loss_pct = 0.02  # 2% below entry
take_profit_pct = 0.05  # 5% above entry

entry_price = prices[-1]
stop_loss = entry_price * (1 - stop_loss_pct)
take_profit = entry_price * (1 + take_profit_pct)
```

### 5. Backtesting Before Trading

```python
# Test strategy on historical data
# Check:
- Win rate (% of profitable trades)
- Sharpe ratio (risk-adjusted returns)
- Max drawdown (largest peak-to-trough decline)
- Number of trades (sufficient sample size)
```

## FAQ

### Q: Which indicator should I use?

**A:** It depends on your trading style:
- **Trend traders**: SMA, EMA, MACD
- **Mean reversion traders**: RSI, Bollinger Bands
- **Momentum traders**: RSI, MACD histogram
- **Volatile markets**: Bollinger Bands
- **Ranging markets**: RSI

### Q: What period should I use?

**A:** Depends on timeframe:
- **Scalping (1-5 min)**: Period 5-12
- **Day trading (hourly)**: Period 14-20
- **Swing trading (daily)**: Period 20-50
- **Position trading (weekly)**: Period 50-200

### Q: How many indicators should I use?

**A:** 2-3 indicators maximum. More leads to:
- Analysis paralysis
- Conflicting signals
- Diminishing returns

**Recommended combinations:**
- Trend (EMA) + Momentum (RSI) + Timing (MACD)
- Trend (SMA) + Volatility (Bollinger Bands)

### Q: What about false signals?

**A:** All indicators produce false signals. Mitigation:
- Confirm with multiple indicators
- Use appropriate timeframes
- Check volume for strength
- Set strict stop-losses
- Track statistics (win rate, profit factor)

### Q: Can I use these indicators for different assets?

**A:** Yes! These are universal:
- **Stocks**: Yes, use daily data
- **Cryptocurrencies**: Yes, use 1-hour+ timeframes
- **Forex**: Yes, use 4-hour+ timeframes
- **Futures**: Yes, adjust periods for contract liquidity

### Q: How do I handle NaN values?

**A:** First `period-1` values are NaN (insufficient data):

```python
# Skip NaN values when analyzing
valid_idx = ~np.isnan(indicator_values)
valid_prices = prices[valid_idx]
valid_signals = signals[valid_idx]
```

## Next Steps

- **Advanced Analysis**: Check docs/phase2 for advanced indicators (ATR, Stochastic)
- **Strategy Development**: See docs/STRATEGIES.md for complete trading strategies
- **Integration**: See docs/BACKTESTER.md for backtesting framework
- **Production Deployment**: See docs/PRODUCTION.md for live trading setup

## Additional Resources

- Mathematical Details: See docs/phase1/TASK_*.md for each indicator
- API Reference: See docs/API_REFERENCE.md
- Architecture: See docs/ARCHITECTURE.md
- Contributing: See docs/CONTRIBUTING.md
