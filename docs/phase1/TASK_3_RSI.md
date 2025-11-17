# Task 1.3: Relative Strength Index (RSI) Indicator

**Difficulty**: ⭐⭐⭐ (Intermediate-Advanced)
**Estimated Time**: 5 hours
**GitHub Issue**: #3
**Prerequisite**: Task 1.1 (SMA)

## Overview

Implement RSI - a momentum indicator (0-100 scale) that identifies overbought/oversold conditions.

## What You'll Learn

- Gain/loss calculations and separation
- Averaging gains and losses separately
- Momentum analysis
- Threshold-based signals (>70, <30)

## Mathematical Definition

```
Step 1: Calculate price changes
  Change[i] = Price[i] - Price[i-1]

Step 2: Separate gains and losses
  Gains[i] = max(0, Change[i])
  Losses[i] = max(0, -Change[i])

Step 3: Average gains and losses (over period)
  AvgGain = average of Gains over last N periods
  AvgLoss = average of Losses over last N periods

Step 4: Calculate RS ratio
  RS = AvgGain / AvgLoss

Step 5: Convert to 0-100 scale
  RSI = 100 - (100 / (1 + RS))

Interpretation:
  RSI > 70: Overbought (potential sell)
  RSI < 30: Oversold (potential buy)
  50: Neutral
```

## Implementation

### 1. Implement RSI Class

**File**: `src/indicators/momentum.py` (create new file)

```python
import numpy as np
from .base import Indicator

class RSI(Indicator):
    """Relative Strength Index indicator."""

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate RSI (0-100 scale).

        Process:
        1. Calculate price changes
        2. Separate into gains and losses
        3. Average gains and losses
        4. Calculate RS ratio
        5. Convert to 0-100 scale

        Args:
            data: Price data

        Returns:
            RSI values (0-100 scale)
        """
        rsi = np.full_like(data, np.nan, dtype=float)

        # Step 1: Price changes
        changes = np.diff(data)

        # Step 2: Separate gains and losses
        gains = np.where(changes > 0, changes, 0.0)
        losses = np.where(changes < 0, -changes, 0.0)

        # Step 3: Initialize with simple average
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        # Step 4: Calculate first RSI
        if avg_loss == 0:
            rsi[self.period] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[self.period] = 100 - (100 / (1 + rs))

        # Step 5: Smooth using Wilder's smoothing
        for i in range(self.period + 1, len(data)):
            current_gain = gains[i - 1]
            current_loss = losses[i - 1]

            # Smooth: weight previous average + current change
            avg_gain = (avg_gain * (self.period - 1) + current_gain) / self.period
            avg_loss = (avg_loss * (self.period - 1) + current_loss) / self.period

            # Calculate RSI
            if avg_loss == 0:
                rsi[i] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    def get_signals(self, rsi_values: np.ndarray) -> np.ndarray:
        """Generate trading signals based on RSI thresholds."""
        signals = np.zeros_like(rsi_values)

        for i in range(1, len(rsi_values)):
            if np.isnan(rsi_values[i]):
                continue

            # Oversold: buy signal
            if rsi_values[i] < 30:
                signals[i] = 1
            # Overbought: sell signal
            elif rsi_values[i] > 70:
                signals[i] = -1

        return signals
```

### 2. Unit Tests

```python
class TestRSI:

    def test_rsi_bounds(self):
        """Test RSI stays within 0-100 range."""
        data = np.random.uniform(90, 110, size=100)
        rsi = RSI(period=14)
        result = rsi(data)

        valid = result[~np.isnan(result)]
        assert np.all((valid >= 0) & (valid <= 100))

    def test_rsi_uptrend(self):
        """Test RSI identifies uptrend (high RSI)."""
        # Consistent uptrend
        data = np.arange(100, 150, dtype=float)
        rsi = RSI(period=14)
        result = rsi(data)

        # Should be very high (>90)
        assert result[-1] > 90

    def test_rsi_downtrend(self):
        """Test RSI identifies downtrend (low RSI)."""
        # Consistent downtrend
        data = np.arange(150, 100, -1, dtype=float)
        rsi = RSI(period=14)
        result = rsi(data)

        # Should be very low (<10)
        assert result[-1] < 10

    def test_rsi_overbought_oversold(self):
        """Test RSI generates correct signals."""
        # Create data with oscillation
        data = np.array([100, 95, 100, 95, 100, 95, 120, 115, 120, 115],
                        dtype=float)
        rsi = RSI(period=2)
        signals = rsi.get_signals(rsi(data))

        # Should have some signals
        assert np.sum(np.abs(signals)) > 0

    def test_rsi_neutral(self):
        """Test RSI near 50 for neutral market."""
        # Alternating up/down
        data = np.array([100, 101, 100, 101, 100, 101, 100, 101],
                        dtype=float)
        rsi = RSI(period=2)
        result = rsi(data)

        # Should be around 50 (balanced gains/losses)
        valid = result[~np.isnan(result)]
        assert np.all(np.abs(valid - 50) < 30)

    def test_rsi_gap_handling(self):
        """Test RSI handles large price gaps."""
        data = np.array([100, 90, 120, 85, 130], dtype=float)
        rsi = RSI(period=2)
        result = rsi(data)

        # Should not crash on large moves
        assert np.all(np.isfinite(result[~np.isnan(result)]))

    def test_rsi_extreme_values(self):
        """Test RSI correctly identifies extreme conditions."""
        # All gains
        data_gains = np.array([100, 101, 102, 103, 104, 105], dtype=float)
        rsi = RSI(period=2)
        result = rsi(data_gains)

        # Should be 100
        assert np.isclose(result[-1], 100)

        # All losses
        data_losses = np.array([105, 104, 103, 102, 101, 100], dtype=float)
        result = rsi(data_losses)

        # Should be 0
        assert np.isclose(result[-1], 0)
```

## Validation

### Against TradingView
1. Add chart with RSI(14)
2. Verify overbought (>70) and oversold (<30) zones
3. Compare specific RSI values

### Against Professional Code
```python
import pandas as pd

data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108],
                dtype=float)

your_rsi = RSI(period=14)(data)

# Can validate logic by checking gains/losses separately
changes = np.diff(data)
gains = np.where(changes > 0, changes, 0)
losses = np.where(changes < 0, -changes, 0)
# Verify your RSI uses these correctly
```

## Run Tests

```bash
pytest tests/test_indicators.py::TestRSI -v
pytest tests/test_indicators.py::TestRSI --cov=src/indicators
```

## Success Criteria

- [ ] RSI class implemented with correct gain/loss separation
- [ ] Wilder's smoothing applied correctly
- [ ] All 7 unit tests pass
- [ ] Correctly identifies overbought (>70) and oversold (<30)
- [ ] 100% code coverage for RSI
- [ ] Signal generation works properly
- [ ] Handles edge cases (all gains, all losses, gaps)

## Key Insights to Document

1. **Gain/Loss separation**: Why must gains and losses be separate?
2. **Wilder's smoothing**: Why weight previous average differently?
3. **Overbought/oversold**: What do >70 and <30 mean practically?
4. **Divergence**: When RSI makes new high but price doesn't - what does it mean?

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Not separating gains/losses | Gains and losses must be calculated separately |
| Wrong smoothing formula | Use (prev × (period-1) + current) / period |
| Division by zero | Check `if avg_loss == 0` before dividing |
| Off-by-one indexing | changes = diff(data) is 1 element shorter |
| Not handling no-change periods | When gain=0 and loss=0, RSI should be 50 |

## References

- RSI definition: https://en.wikipedia.org/wiki/Relative_strength_index
- Wilder's smoothing: Original technical analysis book

## Next Steps

Once complete:
1. ✅ Document how Wilder's smoothing differs from EMA
2. ✅ Test on real data to verify signal quality
3. ✅ Move to Task 1.4 (MACD) - combines indicators

---

**Great progress! 🚀**
