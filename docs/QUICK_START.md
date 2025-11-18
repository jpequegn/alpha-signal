# Quick Start Guide - 5 Minutes to First Signals

Get started with AlphaSignal in less than 5 minutes.

## Installation (1 minute)

```bash
# Clone repository
git clone https://github.com/yourusername/alpha-signal.git
cd alpha-signal

# Install
pip install -r requirements.txt
```

## Your First Indicator (1 minute)

```python
import numpy as np
from src.indicators import SMA

# Get price data
prices = np.array([100, 101, 102, 103, 104, 105])

# Create and calculate indicator
sma = SMA(period=3)
result = sma(prices)

print(result)
# Output: [nan, nan, 101.0, 102.0, 103.0, 104.0]
```

## All 5 Indicators (2 minutes)

```python
import numpy as np
from src.indicators import SMA, EMA, RSI, MACD, BollingerBands

# Sample price data
prices = np.array([100, 101, 102, 101, 103, 104, 105, 104, 106, 107] * 10)

# 1. Simple Moving Average
sma = SMA(period=5)
sma_result = sma(prices)

# 2. Exponential Moving Average
ema = EMA(period=5)
ema_result = ema(prices)

# 3. Relative Strength Index
rsi = RSI(period=14)
rsi_result = rsi(prices)
rsi_signals = rsi.get_signals(rsi_result)

# 4. MACD
macd = MACD(fast=12, slow=26, signal=9)
macd_line, signal_line, histogram = macd(prices)
macd_signals = macd.get_signals(macd_line, signal_line)

# 5. Bollinger Bands
bb = BollingerBands(period=20, num_std=2.0)
upper, middle, lower = bb(prices)
bb_signals = bb.get_signals(prices, upper, lower)

# Print summary
print(f"SMA:  {sma_result[-1]:.2f}")
print(f"EMA:  {ema_result[-1]:.2f}")
print(f"RSI:  {rsi_result[-1]:.1f} (signal: {rsi_signals[-1]})")
print(f"MACD: {macd_line[-1]:.4f}")
print(f"BB:   Upper={upper[-1]:.2f}, Middle={middle[-1]:.2f}, Lower={lower[-1]:.2f}")
```

## Understanding Output (1 minute)

### NaN Values (Initialization Period)

First `period-1` values are NaN (not enough data):

```python
prices = np.array([100, 101, 102, 103, 104])
sma = SMA(period=3)
result = sma(prices)
# result = [NaN, NaN, 101.0, 102.0, 103.0]
#           ^^^ First 2 (period-1) are NaN
```

**Why?** Indicator needs sufficient data before calculating.

### Signal Values

Each indicator's `get_signals()` returns:
- **-1**: Sell/Oversold/Bearish
- **0**: Neutral/Hold
- **1**: Buy/Overbought/Bullish

```python
rsi = RSI()
rsi_values = rsi(prices)
signals = rsi.get_signals(rsi_values)
# signals = [0, 0, -1, 0, 1, 0, -1, ...]
#                 ↑ Oversold  ↑ Overbought
```

## Common Tasks

### Get Latest Signal

```python
prices = np.array([...100 daily prices...])
rsi = RSI(period=14)
rsi_values = rsi(prices)
signals = rsi.get_signals(rsi_values)

latest_signal = signals[-1]
if latest_signal == 1:
    print("Buy signal!")
elif latest_signal == -1:
    print("Sell signal!")
else:
    print("Hold")
```

### Filter Valid Data (Skip NaN)

```python
prices = np.array([...])
sma = SMA(period=20)
sma_values = sma(prices)

# Get only valid (non-NaN) values
valid_idx = ~np.isnan(sma_values)
valid_prices = prices[valid_idx]
valid_sma = sma_values[valid_idx]

print(f"Valid data: {len(valid_sma)} values")
```

### Multi-Indicator Confirmation

```python
prices = np.array([...])

# Get signals from multiple indicators
ema = EMA(period=12)
rsi = RSI(period=14)

ema_values = ema(prices)
rsi_values = rsi(prices)
rsi_signals = rsi.get_signals(rsi_values)

# Confirm: Price above EMA AND RSI not overbought
price_above_ema = prices[-1] > ema_values[-1]
rsi_not_overbought = rsi_signals[-1] != 1

if price_above_ema and rsi_not_overbought:
    print("Bullish confirmation!")
```

### Trend Following with EMA

```python
prices = np.array([...])

ema_fast = EMA(period=5)
ema_slow = EMA(period=20)

fast = ema_fast(prices)
slow = ema_slow(prices)

# Check latest crossover
if fast[-1] > slow[-1]:
    print("Fast EMA above Slow EMA - Uptrend")
else:
    print("Fast EMA below Slow EMA - Downtrend")
```

### Volatility Analysis

```python
prices = np.array([...])
bb = BollingerBands(period=20)
upper, middle, lower = bb(prices)

# Calculate bandwidth percentage
bandwidth_pct = bb.get_bandwidth_percent(upper, middle, lower)

if bandwidth_pct[-1] < 10:
    print("Squeeze! Low volatility")
elif bandwidth_pct[-1] > 20:
    print("Expansion! High volatility")
else:
    print("Normal volatility")
```

## Error Handling

### Always Validate Input

```python
import numpy as np
from src.indicators import SMA

# ❌ Wrong - List instead of numpy array
prices = [100, 101, 102]
sma = SMA(period=3)
try:
    result = sma(prices)  # TypeError!
except TypeError as e:
    print(f"Error: {e}")

# ✅ Correct - Convert to numpy
prices = np.array([100, 101, 102])
result = sma(prices)
```

### Handle Insufficient Data

```python
prices = np.array([100, 101, 102])  # Only 3 values
sma = SMA(period=20)
result = sma(prices)

print(result)  # [NaN, NaN, NaN] - All NaN (insufficient data)

# Use only valid values
valid_idx = ~np.isnan(result)
if np.any(valid_idx):
    print(f"Valid SMA values: {result[valid_idx]}")
else:
    print("No valid SMA values yet")
```

## Next Steps

### 1. Read the Full User Guide
See `docs/PHASE1_USER_GUIDE.md` for:
- Detailed indicator explanations
- Trading strategies
- Best practices
- Frequently asked questions

### 2. Explore Examples
Check `examples/` directory for:
- Complete trading strategies
- Backtesting examples
- Real data analysis

### 3. Understand the API
See `docs/API_REFERENCE.md` for:
- Complete method documentation
- All parameters and options
- Return value specifications
- Error conditions

### 4. Architecture Deep Dive
Read `docs/ARCHITECTURE.md` for:
- System design
- Design patterns
- Code organization
- Future extensibility

### 5. Build Your Strategy
Use indicators to:
- Identify trends (SMA, EMA)
- Confirm momentum (RSI, MACD)
- Measure volatility (Bollinger Bands)
- Combine signals for better accuracy

## Common Questions

**Q: Why are first values NaN?**
A: Indicators need a full lookback period before calculating. First `period-1` values are insufficient data.

**Q: Can I use different periods?**
A: Yes! Each indicator accepts a `period` parameter. Lower = more responsive, higher = smoother.

**Q: How do I backtest?**
A: See `src/backtester/engine.py` and examples. We'll add more in Phase 2.

**Q: What timeframes work?**
A: All of them! 1-minute, 5-minute, hourly, daily, weekly, monthly. Just adjust the period.

**Q: Can I combine indicators?**
A: Absolutely! See "Multi-Indicator Confirmation" section above.

## Troubleshooting

**"ModuleNotFoundError: No module named 'src'"**
→ Run from the project root directory

**"All NaN values"**
→ Not enough data; use prices with length > period

**"TypeError: data must be numpy array"**
→ Convert input to numpy: `prices = np.array(prices)`

**Signal stays 0**
→ Normal. Not every price is overbought/oversold.

## Performance Tips

```python
# Precompute once, reuse multiple times
prices = np.array([...])
ema = EMA(period=12)
ema_values = ema(prices)  # Compute once

# Use result multiple times
signal1 = ema_values > 100
signal2 = ema_values[-1] > prices[-1]

# ✅ Good - Compute once
# ❌ Bad - Recomputing each time
```

---

**Ready to start trading? See `docs/PHASE1_USER_GUIDE.md` for complete guide!**
