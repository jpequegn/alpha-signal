# Task 1.2: Exponential Moving Average (EMA) Indicator

**Difficulty**: ⭐⭐ (Intermediate)
**Estimated Time**: 4 hours
**GitHub Issue**: #2
**Prerequisite**: Task 1.1 (SMA)

## Overview

Implement EMA - a weighted moving average that emphasizes recent prices. EMA responds faster to price changes than SMA.

## What You'll Learn

- Recursive calculations and recursive algorithms
- Smoothing factors and exponential decay
- Why recent data matters in trading
- Composing indicators (EMA is foundation for MACD)

## Mathematical Definition

```
EMA(t) = (Price(t) × α) + (EMA(t-1) × (1 - α))

where α (alpha) = 2 / (period + 1)

Interpretation:
- α is the smoothing factor (weight for current price)
- (1 - α) is the weight for previous EMA
- For period=12: α = 2/13 ≈ 0.1538
```

### Why Recursion?

EMA depends on previous EMA value, not just current prices:
- SMA: Simple calculation, all periods equal weight
- EMA: Recursive calculation, exponentially decaying weights

This makes EMA more responsive to recent changes.

## Implementation

### 1. Implement EMA Class

**File**: `src/indicators/moving_average.py` (add to existing file)

```python
class EMA(Indicator):
    """Exponential Moving Average indicator."""

    @property
    def alpha(self) -> float:
        """Smoothing factor: 2 / (period + 1)"""
        return 2.0 / (self.period + 1)

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate EMA using recursive formula.

        Process:
        1. Initialize EMA with SMA of first period
        2. For each subsequent point: EMA(t) = (Price × α) + (EMA(t-1) × (1 - α))

        Args:
            data: Price data

        Returns:
            EMA values
        """
        ema = np.full_like(data, np.nan, dtype=float)
        alpha = self.alpha

        # Initialize: EMA starts at first period with SMA
        ema[self.period - 1] = np.mean(data[:self.period])

        # Recursive calculation
        for i in range(self.period, len(data)):
            ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))

        return ema
```

### 2. Unit Tests

```python
class TestEMA:

    def test_ema_basic(self):
        """Test EMA calculation on simple data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ema = EMA(period=2)
        result = ema(data)

        # First EMA = SMA of first 2: (1+2)/2 = 1.5
        assert np.isclose(result[1], 1.5)

        # Next: (3 × α) + (1.5 × (1-α)) where α = 2/3
        alpha = 2.0 / 3  # period=2: 2/(2+1)
        expected = (3.0 * alpha) + (1.5 * (1 - alpha))
        assert np.isclose(result[2], expected)

    def test_ema_responsiveness(self):
        """Test EMA responds faster than SMA to price jumps."""
        # Constant values, then jump
        data = np.array([100.0, 100.0, 100.0, 100.0, 100.0,
                         110.0, 110.0, 110.0, 110.0, 110.0])

        sma = SMA(period=5)
        ema = EMA(period=5)

        sma_result = sma(data)
        ema_result = ema(data)

        # After jump, EMA should move toward 110 faster than SMA
        # At index 6 (one bar after jump), EMA should be higher than SMA
        assert ema_result[6] > sma_result[6]

    def test_ema_alpha_factor(self):
        """Test alpha smoothing factor calculation."""
        ema12 = EMA(period=12)
        # α = 2 / (12 + 1) = 2/13 ≈ 0.1538
        assert np.isclose(ema12.alpha, 2.0 / 13)

        ema26 = EMA(period=26)
        # α = 2 / (26 + 1) = 2/27 ≈ 0.0741
        assert np.isclose(ema26.alpha, 2.0 / 27)

    def test_ema_nan_handling(self):
        """Test EMA correctly handles first period-1 values as NaN."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ema = EMA(period=3)
        result = ema(data)

        # First 2 values should be NaN (insufficient data)
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Third value starts calculation
        assert not np.isnan(result[2])

    def test_ema_convergence(self):
        """Test EMA converges to constant values."""
        data = np.array([100.0] * 20)  # Constant values
        ema = EMA(period=5)
        result = ema(data)

        # After initialization, EMA should be 100
        assert np.allclose(result[4:], 100.0)

    def test_ema_vs_professional(self):
        """Validate EMA against pandas ewm (exponential weighted mean)."""
        data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108],
                        dtype=float)

        ema = EMA(period=5)(data)

        # Pandas ewm: span=period, adjust=False matches our calculation
        import pandas as pd
        pandas_ema = pd.Series(data).ewm(span=5, adjust=False).mean().values

        assert np.allclose(ema[4:], pandas_ema[4:], atol=0.01)
```

### 3. Understand Recursion

Key insight: Each EMA value depends on the *previous* EMA value:
```
EMA[0] = NaN (not enough data)
EMA[1] = NaN
EMA[2] = NaN
EMA[3] = NaN
EMA[4] = (Price[4] × α) + (EMA[3] × (1-α))  # First valid, uses SMA
         = (Price[4] × α) + (SMA × (1-α))
EMA[5] = (Price[5] × α) + (EMA[4] × (1-α))  # Uses previous EMA
```

This is why:
1. ✅ EMA must be calculated sequentially (can't parallelize)
2. ✅ EMA is more responsive (recent prices weighted more)
3. ✅ EMA never equals simple average (except for constant data)

## Validation

### Against TradingView
1. Add chart with EMA(12) and EMA(26)
2. Record values on specific dates
3. Compare with your implementation

### Against Professional Code
```python
import pandas as pd

data = np.random.uniform(95, 105, size=100)
your_ema = EMA(period=12)(data)

# Pandas ewm with span=12
pandas_ema = pd.Series(data).ewm(span=12, adjust=False).mean().values

# Should match (within rounding)
assert np.allclose(your_ema[11:], pandas_ema[11:], atol=0.01)
```

## Run Tests

```bash
pytest tests/test_indicators.py::TestEMA -v
pytest tests/test_indicators.py::TestEMA --cov=src/indicators
```

## Success Criteria

- [ ] EMA class implemented with recursive calculation
- [ ] Alpha factor property calculated correctly
- [ ] All 6 unit tests pass
- [ ] Validated against pandas ewm()
- [ ] Shows responsiveness to price changes vs SMA
- [ ] 100% code coverage for EMA
- [ ] Handles NaN values correctly

## Key Insights to Document

1. **Why recursive?**: Explain why each EMA depends on previous EMA
2. **Alpha significance**: What does α = 0.15 mean? (15% recent, 85% historical)
3. **Responsiveness**: Why EMA(12) responds faster than SMA(12)?
4. **Use in MACD**: EMA is component of MACD (next task)

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Starting EMA at index 0 | Must start at index period-1 with SMA |
| Wrong alpha calculation | Must be 2/(period+1), not 1/period |
| Not using previous EMA | Each value must use previous EMA recursively |
| Parallel calculation | Can't parallelize - must go sequentially |

## References

- EMA definition: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
- Pandas ewm: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html

## Next Steps

Once complete:
1. ✅ Document insights on responsiveness
2. ✅ Understand why alpha matters
3. ✅ Move to Task 1.3 (RSI) - uses gain/loss calculations

---

**Keep building! 🚀**
