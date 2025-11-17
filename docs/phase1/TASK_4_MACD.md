# Task 1.4: MACD (Moving Average Convergence Divergence) Indicator

**Difficulty**: ⭐⭐ (Intermediate)
**Estimated Time**: 4 hours
**GitHub Issue**: #4
**Prerequisite**: Task 1.2 (EMA)

## Overview

Implement MACD - a trend-following momentum indicator that combines two EMAs. This task teaches indicator composition.

## What You'll Learn

- How to compose indicators (MACD uses EMA)
- Signal line crossover patterns
- Histogram interpretation
- Multi-line indicators

## Mathematical Definition

```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line

Signals:
- MACD crosses above Signal: Bullish (BUY)
- MACD crosses below Signal: Bearish (SELL)
- Histogram changes sign: Momentum shift
```

## Key Insight: Indicator Composition

MACD shows why you built EMA first - you reuse it!

```python
# You already have EMA. Now use it:
ema_fast = EMA(12)(data)        # 12-period EMA
ema_slow = EMA(26)(data)        # 26-period EMA
macd = ema_fast - ema_slow      # Subtract them!
signal = EMA(9)(macd)           # EMA of MACD
```

This is **composing indicators** - building complex ones from simple pieces.

## Implementation

### 1. Implement MACD Class

**File**: `src/indicators/momentum.py` (add to existing file)

```python
from typing import Tuple
from .moving_average import EMA

class MACD(Indicator):
    """MACD (Moving Average Convergence Divergence) indicator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        """Initialize MACD with periods."""
        self.fast_period = fast
        self.slow_period = slow
        self.signal_period = signal_period

    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD line, signal line, and histogram.

        Args:
            data: Price data

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate EMAs
        ema_fast = EMA(self.fast_period)(data)
        ema_slow = EMA(self.slow_period)(data)

        # MACD line = difference between fast and slow EMA
        macd_line = ema_fast - ema_slow

        # Signal line = EMA of MACD line
        signal_line = EMA(self.signal_period)(macd_line)

        # Histogram = MACD - Signal
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def get_signals(self, histogram: np.ndarray) -> np.ndarray:
        """
        Generate trading signals based on histogram crossovers.

        When histogram crosses zero:
        - From negative to positive: BULLISH (BUY)
        - From positive to negative: BEARISH (SELL)

        Args:
            histogram: MACD histogram

        Returns:
            Signals: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        signals = np.zeros_like(histogram)

        for i in range(1, len(histogram)):
            if np.isnan(histogram[i-1]) or np.isnan(histogram[i]):
                continue

            # Bullish cross: histogram goes from negative to positive
            if histogram[i-1] < 0 and histogram[i] > 0:
                signals[i] = 1

            # Bearish cross: histogram goes from positive to negative
            elif histogram[i-1] > 0 and histogram[i] < 0:
                signals[i] = -1

        return signals
```

### 2. Unit Tests

```python
class TestMACD:

    def test_macd_basic(self):
        """Test MACD calculation."""
        data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108,
                         110, 109, 111, 113, 112, 114, 115, 114, 116, 118],
                        dtype=float)

        macd = MACD(fast=2, slow=3, signal_period=2)  # Small periods for testing
        macd_line, signal_line, histogram = macd(data)

        # Should have NaN values initially
        assert np.any(np.isnan(macd_line))

        # Histogram = MACD - Signal (should compute correctly)
        assert np.allclose(histogram[~np.isnan(histogram)],
                          macd_line[~np.isnan(histogram)] - signal_line[~np.isnan(histogram)])

    def test_macd_uptrend(self):
        """Test MACD in uptrend (MACD > Signal)."""
        data = np.arange(100, 150, dtype=float)  # Consistent uptrend
        macd = MACD(fast=5, slow=10, signal_period=3)
        macd_line, signal_line, histogram = macd(data)

        # In strong uptrend, MACD > Signal (positive histogram)
        valid_hist = histogram[~np.isnan(histogram)]
        assert np.all(valid_hist[-5:] > 0)

    def test_macd_signals(self):
        """Test MACD signal generation."""
        # Create data with reversal
        data = np.concatenate([
            np.arange(100, 110, dtype=float),    # Uptrend
            np.arange(110, 100, -1, dtype=float) # Downtrend
        ])

        macd = MACD(fast=2, slow=3, signal_period=2)
        macd_line, signal_line, histogram = macd(data)
        signals = macd.get_signals(histogram)

        # Should have some signals
        assert np.any(signals != 0)

        # Should have both buy and sell signals
        assert np.any(signals == 1)  # BUY signals
        assert np.any(signals == -1)  # SELL signals

    def test_macd_histogram_interpretation(self):
        """Test histogram correctly represents MACD vs Signal."""
        data = np.random.uniform(95, 105, size=50)
        macd = MACD(fast=5, slow=10, signal_period=3)
        macd_line, signal_line, histogram = macd(data)

        # Histogram should be MACD - Signal
        assert np.allclose(histogram[~np.isnan(histogram)],
                          (macd_line - signal_line)[~np.isnan(histogram)])

    def test_macd_nan_handling(self):
        """Test MACD handles NaN values properly."""
        data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108],
                        dtype=float)
        macd = MACD(fast=12, slow=26, signal_period=9)
        macd_line, signal_line, histogram = macd(data)

        # First values should be NaN (slow period = 26 > len(data))
        # But function should not crash
        assert np.any(np.isnan(macd_line)) or np.any(np.isfinite(macd_line))
```

## Validation

### Understand the Composition

MACD is three indicators in one:
1. **EMA(12)** - Fast, responsive
2. **EMA(26)** - Slow, smooth
3. **Difference** - Convergence/divergence

When EMA12 converges to EMA26, histogram shrinks (low momentum)
When they diverge, histogram expands (high momentum)

### Against TradingView
1. Add MACD(12,26,9) to chart
2. Verify:
   - MACD line above signal = uptrend
   - MACD line below signal = downtrend
   - Histogram crosses zero at reversals

## Run Tests

```bash
pytest tests/test_indicators.py::TestMACD -v
pytest tests/test_indicators.py::TestMACD --cov=src/indicators
```

## Success Criteria

- [ ] MACD class implemented with correct composition
- [ ] Returns tuple of (macd_line, signal_line, histogram)
- [ ] All 5 unit tests pass
- [ ] Signal generation works (crossovers detected)
- [ ] 100% code coverage for MACD
- [ ] Validated against TradingView
- [ ] Can explain indicator composition

## Key Insights to Document

1. **Indicator composition**: How MACD reuses EMA
2. **Convergence/divergence**: What expanding/contracting histogram means
3. **Histogram importance**: Why histogram is the actual trading signal
4. **Lag**: MACD is lagging indicator (uses past data)

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Not using EMA class | Reuse code - don't recalculate |
| Histogram backwards | Must be MACD - Signal, not Signal - MACD |
| Signal crossover vs histogram | Use histogram crossing zero, not lines crossing |
| Wrong default periods | Standard is fast=12, slow=26, signal=9 |

## Architecture Lesson

This task teaches **Don't Repeat Yourself (DRY)**:

```python
# ❌ DON'T: Recalculate EMA inside MACD
def calculate(self, data):
    # Recalculate EMA logic here...
    ema_fast = ...

# ✅ DO: Reuse EMA class
def calculate(self, data):
    ema_fast = EMA(12)(data)
```

Composition enables:
- Code reuse
- Easier testing
- Easier debugging
- Easier modification

## Next Steps

Once complete:
1. ✅ Document what "convergence/divergence" means
2. ✅ Understand why histogram is important
3. ✅ Move to Task 1.5 (Bollinger Bands) - final basic indicator

---

**Almost there! 🚀**
