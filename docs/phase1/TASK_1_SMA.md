# Task 1.1: Simple Moving Average (SMA) Indicator

**Difficulty**: ⭐ (Beginner)
**Estimated Time**: 3 hours
**GitHub Issue**: #1

## Overview

Implement the Simple Moving Average indicator from scratch using NumPy. SMA is the average of closing prices over N periods.

## What You'll Learn

- NumPy array operations and windowing
- How to use convolution for efficient rolling calculations
- Edge case handling (NaN, insufficient data)
- How to structure indicator code for reusability

## Mathematical Definition

```
SMA(t, n) = (P(t) + P(t-1) + ... + P(t-n+1)) / n

Where:
  t = current time/index
  n = period (e.g., 20 for SMA20)
  P = price
```

### Example

```
Prices: [100, 102, 101, 103, 105]
SMA(period=2):
  i=0: NaN (not enough data)
  i=1: (100 + 102) / 2 = 101
  i=2: (102 + 101) / 2 = 101.5
  i=3: (101 + 103) / 2 = 102
  i=4: (103 + 105) / 2 = 104
```

## Implementation Requirements

### 1. Create Base Indicator Class

**File**: `src/indicators/base.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class Indicator(ABC):
    """Base class for all technical indicators."""

    def __init__(self, period: int):
        self.period = period

    @abstractmethod
    def calculate(self, data: np.ndarray) -> np.ndarray:
        pass

    def validate_input(self, data: np.ndarray) -> None:
        if len(data) < self.period:
            raise ValueError(f"Need at least {self.period} data points")
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be numpy array")

    def __call__(self, data: np.ndarray) -> np.ndarray:
        self.validate_input(data)
        return self.calculate(data)
```

### 2. Implement SMA Class

**File**: `src/indicators/moving_average.py`

Use `np.convolve()` for efficiency:

```python
import numpy as np
from .base import Indicator

class SMA(Indicator):
    """Simple Moving Average indicator."""

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate SMA using numpy convolution.

        Convolution is efficient for sliding windows:
        - Creates window of [1/n, 1/n, ..., 1/n]
        - Convolves with data to compute rolling average
        - Much faster than loop-based approach

        Args:
            data: Price data (numpy array)

        Returns:
            SMA values (first period-1 values are NaN)
        """
        window = np.ones(self.period) / self.period
        sma_valid = np.convolve(data, window, mode='valid')

        # Pad with NaN for first period-1 values
        sma = np.concatenate([
            np.full(self.period - 1, np.nan),
            sma_valid
        ])

        return sma
```

### 3. Write Unit Tests

**File**: `tests/test_indicators.py`

```python
import numpy as np
import pytest
from src.indicators.moving_average import SMA

class TestSMA:

    def test_sma_basic(self):
        """Test SMA on simple known data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = SMA(period=2)
        result = sma(data)

        assert np.isnan(result[0])  # First value NaN
        assert np.isclose(result[1], 1.5)  # (1+2)/2
        assert np.isclose(result[2], 2.5)  # (2+3)/2
        assert np.isclose(result[3], 3.5)  # (3+4)/2
        assert np.isclose(result[4], 4.5)  # (4+5)/2

    def test_sma_with_constant_values(self):
        """Test SMA with constant input."""
        data = np.array([100.0] * 10)
        sma = SMA(period=5)
        result = sma(data)

        # SMA of constant should be constant
        assert np.allclose(result[4:], 100.0)

    def test_sma_insufficient_data(self):
        """Test SMA raises error with too little data."""
        data = np.array([1, 2, 3])
        sma = SMA(period=5)

        with pytest.raises(ValueError):
            sma(data)

    def test_sma_type_validation(self):
        """Test SMA validates input type."""
        sma = SMA(period=5)

        # Should accept numpy array
        result = sma(np.array([1, 2, 3, 4, 5]))
        assert result is not None

        # Should reject list
        with pytest.raises(TypeError):
            sma([1, 2, 3, 4, 5])

    def test_sma_different_periods(self):
        """Test SMA with different periods."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        sma5 = SMA(period=5)(data)
        sma2 = SMA(period=2)(data)

        # SMA(5) = (10+20+30+40+50)/5 = 30
        assert np.isclose(sma5[4], 30.0)

        # SMA(2) = (40+50)/2 = 45
        assert np.isclose(sma2[4], 45.0)

    def test_sma_nan_handling(self):
        """Test SMA handles NaN properly."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        sma = SMA(period=2)
        result = sma(data)

        # NaN in window propagates
        assert np.isnan(result[2])
        assert np.isnan(result[3])
```

## Validation Against Professional Implementations

### Method 1: TradingView
1. Add SPY chart
2. Add SMA indicator with periods: 5, 20, 50, 200
3. Record values on a specific date
4. Compare with your implementation

### Method 2: Yahoo Finance
1. Download S&P 500 historical data
2. Calculate SMA(20) manually
3. Compare with indicators shown on charts

### Method 3: NumPy Validation
```python
# Your SMA should match pandas rolling mean
import pandas as pd

data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108])
your_sma = SMA(period=3)(data)

# Compare with pandas
pandas_sma = pd.Series(data).rolling(window=3).mean().values

assert np.allclose(your_sma[2:], pandas_sma[2:], equal_nan=True)
```

## Run Tests

```bash
# Run only SMA tests
pytest tests/test_indicators.py::TestSMA -v

# Run with coverage
pytest tests/test_indicators.py::TestSMA --cov=src/indicators

# Run single test
pytest tests/test_indicators.py::TestSMA::test_sma_basic -v
```

## Success Criteria

- [ ] SMA class implemented in `src/indicators/moving_average.py`
- [ ] Base Indicator class in `src/indicators/base.py`
- [ ] All 6 unit tests pass
- [ ] 100% code coverage for SMA
- [ ] Validated against professional implementations
- [ ] Code follows PEP 8 style
- [ ] Clear docstrings explaining logic

## Key Insights to Document

1. **Why convolution works**: Explain how `np.convolve()` with window computes rolling average
2. **NaN handling**: Why first `period-1` values are NaN
3. **Efficiency**: Compare convolution vs. loop-based approach (speed difference)
4. **Use cases**: When to use SMA (trend identification, support/resistance)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| SMA values don't match TradingView | Check if you're comparing aligned indices (SMA starts at index period-1) |
| NaN propagates incorrectly | Verify np.isnan() is used, not manual checks |
| Convolution seems wrong | Review convolve() documentation for 'valid' mode |
| Tests fail on edge cases | Check handling of single value, empty array |

## References

- NumPy `np.convolve()`: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
- SMA definition: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average

## Next Steps

Once complete:
1. ✅ Mark task as done
2. ✅ Document 1-2 insights learned
3. ✅ Move to Task 1.2 (EMA) - which builds on SMA knowledge

---

**Happy coding! 🚀**
