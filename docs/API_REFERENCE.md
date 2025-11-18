# API Reference - AlphaSignal Phase 1

Complete API documentation for all Phase 1 indicators and base classes.

## Table of Contents

- [Base Classes](#base-classes)
- [Indicators](#indicators)
- [Utility Methods](#utility-methods)
- [Error Handling](#error-handling)

## Base Classes

### `Indicator` (Abstract Base Class)

Base class for all indicators. Cannot be instantiated directly.

**Location:** `src/indicators/base.py`

#### Constructor

```python
class Indicator(ABC):
    def __init__(self, period: int):
        """
        Initialize indicator.

        Args:
            period: Lookback period (minimum 1)

        Raises:
            ValueError: If period < 1 or not integer
        """
```

#### Abstract Methods

```python
@abstractmethod
def calculate(self, data: np.ndarray) -> np.ndarray:
    """
    Calculate indicator values.

    Args:
        data: Price array, shape (N,)

    Returns:
        Indicator values, shape (N,) with first (period-1) as NaN

    Raises:
        TypeError: If data not numpy array
        ValueError: If data not 1D array
    """
```

#### Public Methods

```python
def __call__(self, data: np.ndarray):
    """
    Make indicator callable. Shorthand for calculate().

    Usage:
        result = indicator(prices)  # Same as indicator.calculate(prices)
    """

def _create_output(self, length: int) -> np.ndarray:
    """
    Create output array initialized with NaN.

    Args:
        length: Array length

    Returns:
        NumPy array of NaN values
    """

def _validate_input(self, data: np.ndarray):
    """
    Validate input data.

    Checks:
        - Is numpy array
        - Is 1D
        - Has length >= period

    Raises:
        TypeError/ValueError on validation failure
    """
```

#### Properties

```python
@property
def period(self) -> int:
    """Get the indicator's lookback period."""
```

---

## Indicators

### Simple Moving Average (SMA)

**Location:** `src/indicators/moving_average.py`

#### Constructor

```python
class SMA(Indicator):
    def __init__(self, period: int = 20):
        """
        Initialize SMA.

        Args:
            period: Lookback period (default 20)

        Example:
            sma = SMA(period=20)
        """
```

#### Methods

```python
def calculate(self, data: np.ndarray) -> np.ndarray:
    """
    Calculate Simple Moving Average.

    Formula:
        SMA(t) = (P(t) + P(t-1) + ... + P(t-n+1)) / n

    Args:
        data: Price array, shape (N,)

    Returns:
        SMA values, shape (N,) with first (period-1) as NaN

    Example:
        prices = np.array([100, 101, 102, 103, 104])
        sma = SMA(period=3)
        result = sma(prices)
        # Returns: [nan, nan, 101.0, 102.0, 103.0]
    """
```

#### Example

```python
from src.indicators import SMA
import numpy as np

prices = np.linspace(100, 110, 50)
sma20 = SMA(period=20)
sma_values = sma20(prices)

# Get valid (non-NaN) values
valid_idx = ~np.isnan(sma_values)
print(f"SMA values available from index {np.argmax(valid_idx)}")
```

---

### Exponential Moving Average (EMA)

**Location:** `src/indicators/moving_average.py`

#### Constructor

```python
class EMA(Indicator):
    def __init__(self, period: int = 20):
        """
        Initialize EMA.

        Args:
            period: Lookback period (default 20)

        Example:
            ema = EMA(period=12)
        """
```

#### Properties

```python
@property
def alpha(self) -> float:
    """
    Smoothing factor.

    Formula:
        α = 2 / (period + 1)

    Returns:
        Float between 0 and 1

    Example:
        ema = EMA(period=12)
        print(ema.alpha)  # ~0.154
    """
```

#### Methods

```python
def calculate(self, data: np.ndarray) -> np.ndarray:
    """
    Calculate Exponential Moving Average.

    Formula:
        EMA(t) = Price(t) × α + EMA(t-1) × (1-α)
        where α = 2 / (period + 1)

    Args:
        data: Price array, shape (N,)

    Returns:
        EMA values, shape (N,) with first (period-1) as NaN

    Example:
        prices = np.array([100, 101, 102, 103, 104])
        ema = EMA(period=3)
        result = ema(prices)
    """
```

#### Example

```python
from src.indicators import EMA

prices = np.linspace(100, 110, 50)
ema_fast = EMA(period=5)
ema_slow = EMA(period=20)

fast = ema_fast(prices)
slow = ema_slow(prices)

# Find crossovers
for i in range(1, len(prices)):
    if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
        print(f"Bullish EMA crossover at index {i}")
```

---

### Relative Strength Index (RSI)

**Location:** `src/indicators/momentum.py`

#### Constructor

```python
class RSI(Indicator):
    def __init__(self, period: int = 14):
        """
        Initialize RSI.

        Args:
            period: Lookback period (default 14)

        Example:
            rsi = RSI(period=14)
        """
```

#### Methods

```python
def calculate(self, data: np.ndarray) -> np.ndarray:
    """
    Calculate Relative Strength Index.

    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Avg_Gain / Avg_Loss

    Uses Wilder's smoothing for stability.

    Args:
        data: Price array, shape (N,)

    Returns:
        RSI values 0-100, shape (N,) with first (period) as NaN

    Example:
        prices = np.array([44.0, 44.34, 44.09, 43.61, 44.33, ...])
        rsi = RSI(period=14)
        result = rsi(prices)
    """

def get_signals(
    self,
    rsi_values: np.ndarray,
    overbought: float = 70,
    oversold: float = 30
) -> np.ndarray:
    """
    Generate trading signals from RSI.

    Args:
        rsi_values: RSI values from calculate()
        overbought: Threshold for overbought (default 70)
        oversold: Threshold for oversold (default 30)

    Returns:
        Signal array: 1 (overbought), -1 (oversold), 0 (neutral)

    Example:
        rsi = RSI(period=14)
        rsi_values = rsi(prices)
        signals = rsi.get_signals(rsi_values, overbought=70, oversold=30)
    """
```

#### Signal Interpretation

```
Signal = 1  : RSI > overbought → Potential sell
Signal = 0  : oversold < RSI < overbought → Neutral
Signal = -1 : RSI < oversold → Potential buy
```

#### Example

```python
from src.indicators import RSI

prices = np.random.uniform(95, 105, 100)
rsi = RSI(period=14)
rsi_values = rsi(prices)

# Generate signals with custom thresholds
signals = rsi.get_signals(rsi_values, overbought=75, oversold=25)

# Find oversold conditions
oversold_idx = np.where(signals == -1)[0]
print(f"Oversold on {len(oversold_idx)} days")
```

---

### MACD (Moving Average Convergence Divergence)

**Location:** `src/indicators/momentum.py`

#### Constructor

```python
class MACD(Indicator):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize MACD.

        Args:
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)

        Example:
            macd = MACD(fast=12, slow=26, signal=9)
        """
```

#### Methods

```python
def calculate(self, data: np.ndarray) -> tuple:
    """
    Calculate MACD components.

    Formula:
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(signal) of MACD
        Histogram = MACD - Signal

    Args:
        data: Price array, shape (N,)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
        All shape (N,) with first (slow-1) as NaN

    Example:
        prices = np.linspace(100, 110, 100)
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)
    """

def get_signals(
    self,
    macd_line: np.ndarray,
    signal_line: np.ndarray
) -> np.ndarray:
    """
    Generate trading signals from MACD crossovers.

    Args:
        macd_line: MACD line from calculate()
        signal_line: Signal line from calculate()

    Returns:
        Signal array: 1 (bullish), -1 (bearish), 0 (neutral)

    Example:
        macd_line, signal_line, histogram = macd(prices)
        signals = macd.get_signals(macd_line, signal_line)
    """
```

#### Signal Interpretation

```
Signal = 1  : MACD > Signal → Bullish (momentum increasing)
Signal = 0  : Uncertain or NaN values
Signal = -1 : MACD < Signal → Bearish (momentum decreasing)
```

#### Example

```python
from src.indicators import MACD

prices = np.random.normal(100, 2, 100)
macd = MACD()
macd_line, signal_line, histogram = macd(prices)

# Check histogram for momentum strength
strong_uptrend = histogram > 0.5
strong_downtrend = histogram < -0.5

print(f"Strong uptrend bars: {np.sum(strong_uptrend)}")
print(f"Strong downtrend bars: {np.sum(strong_downtrend)}")
```

---

### Bollinger Bands

**Location:** `src/indicators/volatility.py`

#### Constructor

```python
class BollingerBands(Indicator):
    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        Initialize Bollinger Bands.

        Args:
            period: Lookback period (default 20)
            num_std: Number of standard deviations (default 2.0)

        Example:
            bb = BollingerBands(period=20, num_std=2.0)
        """
```

#### Methods

```python
def calculate(self, data: np.ndarray) -> tuple:
    """
    Calculate Bollinger Bands.

    Formula:
        Middle = SMA(period)
        Upper = Middle + (num_std × StdDev)
        Lower = Middle - (num_std × StdDev)

    Args:
        data: Price array, shape (N,)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
        All shape (N,) with first (period-1) as NaN

    Example:
        prices = np.random.normal(100, 2, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)
    """

def get_bandwidth(
    self,
    upper_band: np.ndarray,
    lower_band: np.ndarray
) -> np.ndarray:
    """
    Calculate band width (absolute volatility).

    Formula:
        Bandwidth = Upper - Lower

    Args:
        upper_band: Upper band from calculate()
        lower_band: Lower band from calculate()

    Returns:
        Bandwidth values, shape (N,)

    Interpretation:
        - Wide bands: High volatility
        - Narrow bands: Low volatility, consolidation
    """

def get_bandwidth_percent(
    self,
    upper_band: np.ndarray,
    middle_band: np.ndarray,
    lower_band: np.ndarray
) -> np.ndarray:
    """
    Calculate bandwidth percentage (relative volatility).

    Formula:
        Bandwidth % = (Upper - Lower) / Middle × 100

    Args:
        upper_band: Upper band from calculate()
        middle_band: Middle band from calculate()
        lower_band: Lower band from calculate()

    Returns:
        Bandwidth percentage, shape (N,)

    Interpretation:
        - < 10%: Squeeze, low volatility
        - 10-20%: Normal volatility
        - > 20%: High volatility
    """

def get_signals(
    self,
    prices: np.ndarray,
    upper_band: np.ndarray,
    lower_band: np.ndarray
) -> np.ndarray:
    """
    Generate trading signals from band touches.

    Args:
        prices: Price array
        upper_band: Upper band from calculate()
        lower_band: Lower band from calculate()

    Returns:
        Signal array: 1 (upper band), -1 (lower band), 0 (neutral)

    Example:
        upper, middle, lower = bb(prices)
        signals = bb.get_signals(prices, upper, lower)
    """
```

#### Signal Interpretation

```
Signal = 1  : Price >= Upper → Overbought
Signal = 0  : Within bands → Neutral
Signal = -1 : Price <= Lower → Oversold
```

#### Example

```python
from src.indicators import BollingerBands

prices = np.random.normal(100, 2, 100)
bb = BollingerBands(period=20, num_std=2)
upper, middle, lower = bb(prices)

# Identify squeeze
bw_pct = bb.get_bandwidth_percent(upper, middle, lower)
squeeze_idx = bw_pct < 10
print(f"Squeeze periods: {np.sum(squeeze_idx)}")

# Get signals
signals = bb.get_signals(prices, upper, lower)
overbought = np.sum(signals == 1)
oversold = np.sum(signals == -1)
print(f"Overbought: {overbought}, Oversold: {oversold}")
```

---

## Utility Methods

### Creating Indicators

```python
# Single indicator
from src.indicators import SMA
sma = SMA(period=20)
result = sma(prices)

# Multiple indicators
from src.indicators import SMA, EMA, RSI, MACD, BollingerBands
indicators = {
    'sma': SMA(period=20),
    'ema': EMA(period=12),
    'rsi': RSI(period=14),
    'macd': MACD(),
    'bb': BollingerBands()
}

results = {name: ind(prices) for name, ind in indicators.items()}
```

### Handling NaN Values

```python
# Get valid (non-NaN) indices
valid_idx = ~np.isnan(indicator_values)

# Filter data
valid_prices = prices[valid_idx]
valid_values = indicator_values[valid_idx]

# Count NaN values
nan_count = np.sum(np.isnan(indicator_values))
print(f"{nan_count} NaN values (initialization period)")
```

### Working with Multiple Timeframes

```python
# Fast timeframe
ema_fast = EMA(period=5)
result_fast = ema_fast(prices_5min)

# Slow timeframe (aggregated data)
ema_slow = EMA(period=50)
result_slow = ema_slow(prices_daily)

# Combine signals
fast_signal = result_fast[-1] > prices_5min[-1]
slow_signal = result_slow[-1] > prices_daily[-1]
combo_signal = fast_signal and slow_signal
```

---

## Error Handling

### Common Errors and Solutions

#### TypeError: data must be numpy array

```python
# ❌ Wrong - List instead of numpy array
prices = [100, 101, 102]
sma = SMA(period=3)
result = sma(prices)  # TypeError!

# ✅ Correct - Use numpy array
prices = np.array([100, 101, 102])
result = sma(prices)
```

#### ValueError: data must be 1D array

```python
# ❌ Wrong - 2D array
prices = np.array([[100, 101], [102, 103]])
sma = SMA(period=3)
result = sma(prices)  # ValueError!

# ✅ Correct - 1D array
prices = np.array([100, 101, 102, 103])
result = sma(prices)
```

#### ValueError: data must have at least period elements

```python
# ❌ Wrong - Insufficient data
prices = np.array([100, 101])
sma = SMA(period=20)
result = sma(prices)  # All NaN (no valid output)

# ✅ Correct - Sufficient data
prices = np.array([100 + i for i in range(50)])
result = sma(prices)  # Valid from index 19+
```

#### ValueError: period must be positive integer

```python
# ❌ Wrong - Invalid periods
sma = SMA(period=0)  # ValueError!
sma = SMA(period=-5)  # ValueError!
sma = SMA(period=3.5)  # ValueError!

# ✅ Correct - Valid period
sma = SMA(period=20)
```

#### ValueError: num_std must be positive

```python
# ❌ Wrong - Invalid std deviations
bb = BollingerBands(period=20, num_std=0)  # ValueError!
bb = BollingerBands(period=20, num_std=-2)  # ValueError!

# ✅ Correct - Valid num_std
bb = BollingerBands(period=20, num_std=2.0)
```

### Best Practices

```python
def safe_calculate_indicators(prices):
    """Example of safe indicator usage."""

    # Validate input
    if not isinstance(prices, np.ndarray):
        raise TypeError("prices must be numpy array")
    if prices.ndim != 1:
        raise ValueError("prices must be 1D array")
    if len(prices) < 100:
        raise ValueError("need at least 100 data points")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive")
    if np.any(np.isnan(prices)):
        raise ValueError("prices contain NaN values")

    # Calculate indicators safely
    try:
        sma = SMA(period=20)
        ema = EMA(period=12)
        rsi = RSI(period=14)

        sma_vals = sma(prices)
        ema_vals = ema(prices)
        rsi_vals = rsi(prices)

        return {
            'sma': sma_vals,
            'ema': ema_vals,
            'rsi': rsi_vals
        }

    except ValueError as e:
        print(f"Error: {e}")
        return None
```

---

## Performance Characteristics

### Time Complexity

| Indicator | Time | Notes |
|-----------|------|-------|
| SMA | O(n) | Uses convolution for efficiency |
| EMA | O(n) | Single pass, linear |
| RSI | O(n) | Two passes for smoothing |
| MACD | O(n) | Calls EMA three times |
| Bollinger Bands | O(n) | StdDev calculation + SMA |

### Space Complexity

| Indicator | Space | Notes |
|-----------|-------|-------|
| SMA | O(n) | Output arrays |
| EMA | O(n) | Output array + SMA intermediate |
| RSI | O(n) | Output + smoothed gains/losses |
| MACD | O(n) | Three output arrays |
| Bollinger Bands | O(n) | Three bands + StdDev array |

### Memory Usage (10,000 prices)

```
SMA: 80 KB (float64 array)
EMA: 160 KB (output + SMA)
RSI: 320 KB (output + smoothing arrays)
MACD: 240 KB (three output arrays)
Bollinger Bands: 320 KB (bands + StdDev)
```

---

## Deprecations and Compatibility

### Version 1.0

- All Phase 1 indicators stable
- No planned deprecations
- Full backward compatibility

### Future Versions

- Phase 2: Advanced indicators
- Phase 3: Strategy combinations
- Phase 4: Backtesting integration

---

## Examples Repository

See `examples/` directory for:
- Complete trading strategies
- Backtesting examples
- Signal analysis notebooks
- Performance benchmarks
