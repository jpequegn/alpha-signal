# Phase 1: Implementation Guide - Build Custom Indicators

**Duration**: 2 weeks
**Effort**: ~20 hours
**Goal**: Implement 5 core trading indicators from first principles, mastering each deeply.

---

## Introduction: Why Build from Scratch?

Most traders use libraries like TA-Lib or pandas-ta. These are black boxes - you don't understand what's inside, so when they fail or give unexpected results, you're stuck.

**The Karpathy Method approach**:
1. Understand the math deeply
2. Implement from scratch
3. Test against known implementations
4. Now you own the knowledge

By the end of Phase 1, you'll understand:
- Why each indicator works
- What edge cases exist
- How to optimize them
- How to modify them for different use cases

---

## Project Structure Setup

### Step 1: Create Project Structure

```bash
cd /Users/julienpequegnot/Code/alpha-signal
mkdir -p src/indicators tests notebooks data
touch src/__init__.py src/indicators/__init__.py tests/__init__.py
```

### Step 2: Create Base Files

**src/indicators/base.py** - Base class for all indicators

```python
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class Indicator(ABC):
    """Base class for all technical indicators."""

    def __init__(self, period: int):
        """
        Initialize indicator with period.

        Args:
            period: Lookback period for calculation (e.g., 20 for SMA20)
        """
        self.period = period
        self.values = np.array([])

    @abstractmethod
    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate indicator values.

        Args:
            data: Input data (typically closing prices)

        Returns:
            Array of indicator values (same length as input)
        """
        pass

    def validate_input(self, data: np.ndarray) -> None:
        """Validate input data."""
        if len(data) < self.period:
            raise ValueError(
                f"Need at least {self.period} data points, got {len(data)}"
            )
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be numpy array")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Calculate indicator when called."""
        self.validate_input(data)
        return self.calculate(data)
```

**requirements.txt**

```
numpy>=1.24.0
pandas>=2.0.0
pytest>=7.0.0
pytest-cov>=4.0.0
alpaca-trade-api>=2.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
```

### Step 3: Create Backtester Framework

**src/backtester/engine.py** - Simple backtesting engine

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BacktestResult:
    """Results from backtesting."""
    total_return: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    trades: int

    def __str__(self):
        return f"""
Backtest Results:
  Total Return: {self.total_return:.2%}
  Win Rate: {self.win_rate:.2%}
  Profit Factor: {self.profit_factor:.2f}
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.2%}
  Total Trades: {self.trades}
        """.strip()

def backtest_signal(
    prices: np.ndarray,
    signals: np.ndarray,  # 1 = BUY, -1 = SELL, 0 = HOLD
    starting_capital: float = 10000
) -> BacktestResult:
    """
    Simple backtest: go long on BUY, exit on SELL.

    Args:
        prices: Close prices
        signals: Trading signals (-1, 0, 1)
        starting_capital: Initial capital

    Returns:
        BacktestResult with metrics
    """
    position = 0  # 0 = no position, 1 = long
    trades = []
    equity = starting_capital
    entry_price = 0

    for i in range(len(signals)):
        signal = signals[i]
        price = prices[i]

        # Entry
        if signal == 1 and position == 0:
            position = 1
            entry_price = price

        # Exit
        elif signal == -1 and position == 1:
            position = 0
            profit = (price - entry_price) / entry_price
            trades.append(profit)
            equity *= (1 + profit)

    # Calculate metrics
    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0)

    trades_array = np.array(trades)
    wins = np.sum(trades_array > 0)
    losses = np.sum(trades_array < 0)

    win_rate = wins / len(trades) if trades else 0
    gross_profit = np.sum(trades_array[trades_array > 0])
    gross_loss = abs(np.sum(trades_array[trades_array < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    total_return = (equity - starting_capital) / starting_capital
    sharpe_ratio = np.mean(trades_array) / np.std(trades_array) if len(trades_array) > 1 else 0

    # Max drawdown (simplified)
    cumulative = np.cumprod(1 + trades_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

    return BacktestResult(
        total_return=total_return,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        trades=len(trades)
    )
```

---

## Task Breakdown

### Task 1.1: Simple Moving Average (SMA)

**Time**: 3 hours
**Difficulty**: ⭐ (Easy)
**Concepts**: Numpy operations, windows, rolling calculations

#### What is SMA?

SMA is the average of closing prices over N periods:
```
SMA(20) = (C1 + C2 + ... + C20) / 20
```

It's simple but reveals **trends** - prices above SMA suggest uptrend, below suggests downtrend.

#### Implementation

Create **src/indicators/moving_average.py**:

```python
import numpy as np
from .base import Indicator

class SMA(Indicator):
    """Simple Moving Average indicator."""

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate SMA using numpy convolution.

        Why convolution? It's efficient for sliding windows.
        Equivalent to: sma[i] = np.mean(data[i-period+1:i+1])

        Args:
            data: Price data

        Returns:
            SMA values (first period-1 values are NaN)
        """
        # Create window of ones
        window = np.ones(self.period) / self.period

        # Use numpy's convolve for efficient rolling mean
        # 'valid' mode: output only where window fully overlaps
        sma_valid = np.convolve(data, window, mode='valid')

        # Pad with NaN for first period-1 values
        sma = np.concatenate([
            np.full(self.period - 1, np.nan),
            sma_valid
        ])

        return sma

# Alternative: Using loops (less efficient but clearer logic)
class SMA_Loop(Indicator):
    """SMA implementation using explicit loops (for learning)."""

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """Calculate SMA with explicit loops."""
        sma = np.full_like(data, np.nan, dtype=float)

        for i in range(self.period - 1, len(data)):
            # Average of period values ending at i
            sma[i] = np.mean(data[i - self.period + 1 : i + 1])

        return sma
```

#### Unit Tests

Create **tests/test_indicators.py**:

```python
import numpy as np
import pytest
from src.indicators.moving_average import SMA

class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_basic(self):
        """Test SMA calculation on simple data."""
        # Known values: [1, 2, 3, 4, 5]
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = SMA(period=2)
        result = sma(data)

        # First value should be NaN (not enough data)
        assert np.isnan(result[0])
        # Second value: (1+2)/2 = 1.5
        assert np.isclose(result[1], 1.5)
        # Third value: (2+3)/2 = 2.5
        assert np.isclose(result[2], 2.5)

    def test_sma_with_real_data(self):
        """Test SMA matches professional implementation."""
        # 10 periods of data
        data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108])
        sma = SMA(period=3)
        result = sma(data)

        # Manual calculation for verification
        # result[2] = (100 + 102 + 101) / 3 = 101
        assert np.isclose(result[2], 101.0)
        # result[3] = (102 + 101 + 103) / 3 = 102
        assert np.isclose(result[3], 102.0)

    def test_sma_insufficient_data(self):
        """Test SMA raises error with insufficient data."""
        data = np.array([1, 2, 3])
        sma = SMA(period=5)

        with pytest.raises(ValueError):
            sma(data)

    def test_sma_with_nan_values(self):
        """Test SMA handles NaN values properly."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        sma = SMA(period=2)
        result = sma(data)

        # Should propagate NaN appropriately
        assert np.isnan(result[2])

    def test_sma_constant_values(self):
        """Test SMA with constant input."""
        data = np.array([100.0] * 10)
        sma = SMA(period=5)
        result = sma(data)

        # SMA of constant should be constant
        assert np.allclose(result[4:], 100.0)

def test_sma_validation():
    """Test SMA validation against TradingView calculation."""
    # Load real S&P 500 data (you'll add this in Task 1.6)
    # Compare SMA(20) against TradingView values
    # This is a placeholder - implement after data loading
    pass
```

#### Validation Checklist

- [ ] Implement SMA using convolution (efficient)
- [ ] Implement SMA using loops (educational)
- [ ] All 5 unit tests pass
- [ ] Compare against TradingView/Yahoo Finance on real data
- [ ] Document why convolution is better than loops
- [ ] Run: `pytest tests/test_indicators.py::TestSMA -v`

---

### Task 1.2: Exponential Moving Average (EMA)

**Time**: 4 hours
**Difficulty**: ⭐⭐ (Moderate)
**Concepts**: Recursion, smoothing factor, numerical stability

#### What is EMA?

EMA gives more weight to recent prices:
```
EMA(t) = (Close(t) × α) + (EMA(t-1) × (1 - α))

where α (alpha) = 2 / (period + 1)
```

For a 12-period EMA: α = 2/13 ≈ 0.154

#### Key Difference from SMA

- SMA: All periods weighted equally
- EMA: Recent periods weighted more (95% of weight in recent period)

This makes EMA more responsive to recent price changes.

#### Implementation

```python
class EMA(Indicator):
    """Exponential Moving Average indicator."""

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate EMA using recursive formula.

        EMA(t) = (Price(t) × α) + (EMA(t-1) × (1 - α))

        Args:
            data: Price data

        Returns:
            EMA values
        """
        ema = np.full_like(data, np.nan, dtype=float)
        alpha = 2.0 / (self.period + 1)

        # Initialize EMA with SMA of first period
        ema[self.period - 1] = np.mean(data[:self.period])

        # Calculate EMA recursively
        for i in range(self.period, len(data)):
            ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))

        return ema

    @property
    def alpha(self) -> float:
        """Smoothing factor."""
        return 2.0 / (self.period + 1)
```

#### Unit Tests

```python
class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_basic(self):
        """Test EMA calculation on simple data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ema = EMA(period=2)
        result = ema(data)

        # First EMA = SMA of first 2: (1+2)/2 = 1.5
        assert np.isclose(result[1], 1.5)
        # Next: (3 × 0.667) + (1.5 × 0.333) ≈ 2.5
        assert np.isclose(result[2], 2.5, atol=0.01)

    def test_ema_responsiveness(self):
        """Test that EMA responds quickly to price changes."""
        data = np.array([100.0] * 5 + [110.0] * 5)
        sma = SMA(period=5)
        ema = EMA(period=5)

        sma_result = sma(data)
        ema_result = ema(data)

        # EMA should respond faster than SMA to the jump
        # After the jump, EMA should be closer to 110 than SMA
        assert ema_result[6] > sma_result[6]

    def test_ema_alpha_calculation(self):
        """Test alpha smoothing factor."""
        ema = EMA(period=12)
        # For period 12: α = 2/(12+1) ≈ 0.154
        assert np.isclose(ema.alpha, 2.0/13)

        ema26 = EMA(period=26)
        # For period 26: α = 2/(26+1) ≈ 0.074
        assert np.isclose(ema26.alpha, 2.0/27)
```

#### Validation Checklist

- [ ] Implement EMA with recursive formula
- [ ] Alpha smoothing factor calculated correctly
- [ ] EMA(12) and EMA(26) match professional implementations
- [ ] Unit tests pass
- [ ] Document why EMA is more responsive than SMA
- [ ] Run: `pytest tests/test_indicators.py::TestEMA -v`

---

### Task 1.3: Relative Strength Index (RSI)

**Time**: 5 hours
**Difficulty**: ⭐⭐⭐ (Moderate-Hard)
**Concepts**: Gain/loss calculations, averaging, normalization

#### What is RSI?

RSI measures momentum on a 0-100 scale:
```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
```

- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)
- 50: Neutral

#### Implementation

```python
class RSI(Indicator):
    """Relative Strength Index indicator."""

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate RSI.

        Process:
        1. Calculate price changes
        2. Separate into gains and losses
        3. Average gains and losses (using EMA)
        4. Calculate RS ratio
        5. Convert to 0-100 scale

        Args:
            data: Price data

        Returns:
            RSI values (0-100 scale)
        """
        rsi = np.full_like(data, np.nan, dtype=float)

        # Calculate price changes
        changes = np.diff(data)

        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        # Initialize with simple average
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        rsi[self.period] = 100 - (100 / (1 + (avg_gain / avg_loss))) \
            if avg_loss != 0 else 100

        # Use exponential smoothing for subsequent values
        alpha = 1.0 / self.period

        for i in range(self.period + 1, len(data)):
            gain = gains[i - 1]
            loss = losses[i - 1]

            # Smooth average gain and loss
            avg_gain = (avg_gain * (self.period - 1) + gain) / self.period
            avg_loss = (avg_loss * (self.period - 1) + loss) / self.period

            # Calculate RSI
            if avg_loss == 0:
                rsi[i] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi
```

#### Unit Tests

```python
class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_basic(self):
        """Test RSI on simple up/down data."""
        # Consistent gains: should have high RSI
        data_up = np.array([100, 101, 102, 103, 104, 105])
        rsi = RSI(period=2)
        result = rsi(data_up)

        # After calculation period, RSI should be high (>70)
        assert result[-1] > 70

    def test_rsi_bounds(self):
        """Test RSI stays in 0-100 range."""
        data = np.random.uniform(90, 110, size=100)
        rsi = RSI(period=14)
        result = rsi(data)

        # All non-NaN values should be 0-100
        valid = result[~np.isnan(result)]
        assert np.all((valid >= 0) & (valid <= 100))

    def test_rsi_overbought_oversold(self):
        """Test RSI correctly identifies extremes."""
        # Construct data with obvious overbought (up, up, up, ...)
        data_overbought = np.arange(100, 150)  # Consistent uptrend
        rsi = RSI(period=14)
        result = rsi(data_overbought)

        # Final RSI should be very high (> 90)
        assert result[-1] > 90

        # Construct oversold (downtrend)
        data_oversold = np.arange(150, 100, -1)
        result = rsi(data_oversold)

        # Final RSI should be very low (< 10)
        assert result[-1] < 10
```

---

### Task 1.4: MACD (Moving Average Convergence Divergence)

**Time**: 4 hours
**Difficulty**: ⭐⭐ (Moderate)
**Concepts**: Composition of indicators, signal lines, histogram

#### What is MACD?

MACD uses EMA12 and EMA26 to identify trends:
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD
Histogram = MACD - Signal
```

**Signals**:
- MACD crosses above Signal: Bullish
- MACD crosses below Signal: Bearish

#### Implementation

```python
class MACD(Indicator):
    """MACD (Moving Average Convergence Divergence) indicator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize with MACD parameters."""
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal

    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD line, signal line, and histogram.

        Args:
            data: Price data

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = EMA(self.fast_period)(data)
        ema_slow = EMA(self.slow_period)(data)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = EMA(self.signal_period)(macd_line)

        # Histogram (MACD - Signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def get_signals(self, histogram: np.ndarray) -> np.ndarray:
        """
        Generate trading signals from MACD histogram.

        Args:
            histogram: MACD histogram

        Returns:
            Array of signals: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        signals = np.zeros_like(histogram)

        for i in range(1, len(histogram)):
            # Histogram crosses above 0: bullish
            if histogram[i-1] < 0 and histogram[i] > 0:
                signals[i] = 1
            # Histogram crosses below 0: bearish
            elif histogram[i-1] > 0 and histogram[i] < 0:
                signals[i] = -1

        return signals
```

---

### Task 1.5: Bollinger Bands

**Time**: 3 hours
**Difficulty**: ⭐⭐ (Moderate)
**Concepts**: Standard deviation, volatility, mean reversion

#### What are Bollinger Bands?

Bands around SMA showing volatility:
```
Upper Band = SMA + (2 × StdDev)
Lower Band = SMA - (2 × StdDev)
Middle Band = SMA
```

**Signals**:
- Price touches Upper Band: Overbought
- Price touches Lower Band: Oversold
- Band width: Market volatility

#### Implementation

```python
class BollingerBands(Indicator):
    """Bollinger Bands indicator."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.

        Args:
            period: SMA period (typically 20)
            std_dev: Number of standard deviations (typically 2)
        """
        self.period = period
        self.std_dev = std_dev

    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price data

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Middle band (SMA)
        sma = SMA(self.period)(data)

        # Calculate rolling standard deviation
        std = np.full_like(data, np.nan, dtype=float)
        for i in range(self.period - 1, len(data)):
            window = data[i - self.period + 1 : i + 1]
            std[i] = np.std(window)

        # Upper and lower bands
        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)

        return upper_band, sma, lower_band

    def get_signals(
        self,
        prices: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray
    ) -> np.ndarray:
        """Generate mean-reversion signals."""
        signals = np.zeros_like(prices)

        for i in range(1, len(prices)):
            # Touch lower band: oversold, potential buy
            if prices[i-1] > lower_band[i-1] and prices[i] <= lower_band[i]:
                signals[i] = 1
            # Touch upper band: overbought, potential sell
            elif prices[i-1] < upper_band[i-1] and prices[i] >= upper_band[i]:
                signals[i] = -1

        return signals
```

---

## Testing and Validation

### Unit Test Strategy

For each indicator:
1. **Basic functionality** - Verify calculation on simple data
2. **Edge cases** - NaN, single value, constant values
3. **Professional comparison** - Match known implementations
4. **Signal generation** - Verify signals make sense

### Run All Tests

```bash
pytest tests/test_indicators.py -v --cov=src/indicators
```

### Data Validation

Compare outputs against:
- **TradingView**: Free charts with all indicators
- **Yahoo Finance**: Free historical data
- **Professional implementations**: TA-Lib, pandas-ta

---

## Phase 1 Summary

After completing all 5 tasks:

### Knowledge Gained
- ✅ Deep understanding of market indicators
- ✅ Numpy array operations and efficiency
- ✅ How to compose indicators (MACD uses EMA)
- ✅ Signal generation and mean reversion

### Code Quality
- ✅ 100% test coverage
- ✅ Well-documented, readable code
- ✅ Validated against professional implementations
- ✅ Optimized for performance

### Ready for Phase 2
- ✅ Can reliably calculate any of these 5 indicators
- ✅ Understand strengths and weaknesses of each
- ✅ Can build LLM agent that reasons about them

---

## Common Pitfalls & How to Avoid Them

| Pitfall | Issue | Solution |
|---------|-------|----------|
| Forward-looking bias | Using future data in past | Only calculate at time t using data ≤ t |
| NaN handling | Ignoring NaN propagation | Test with explicit NaN values |
| Off-by-one errors | Index mistakes in loops | Verify first few manual calculations |
| Numerical instability | Float precision issues | Use np.isclose() in tests, not == |
| Not validating | Assuming code is correct | Always compare against known good |

---

## Success Checklist

- [ ] All 5 indicator implementations complete
- [ ] 25+ unit tests written and passing
- [ ] 100% code coverage for indicator library
- [ ] Each indicator validated against professional implementation
- [ ] Clear documentation of each indicator's logic
- [ ] Code follows PEP 8 style guide
- [ ] Ready to use in Phase 2 daemon

---

## Next Steps

1. **Start with Task 1.1 (SMA)** - Simplest, builds foundation
2. **Each task** → Implement → Test → Validate
3. **Document learning** - Note insights about each indicator
4. **Create GitHub issues** - One per task, track progress
5. **Move to Phase 2** - When all 5 complete

---

**Let's build! 🚀**
