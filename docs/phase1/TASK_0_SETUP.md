# Task 0: Repository Setup & Infrastructure

**Prerequisite for all Phase 1 tasks**

---

## Overview

Before implementing any indicators, we need to establish the foundational infrastructure that all 5 indicators will build upon. This task involves:

1. Creating the actual directory structure
2. Writing the `Indicator` base class
3. Building the `Backtester` framework
4. Setting up pytest configuration and fixtures
5. Creating reusable test utilities
6. Validating everything works

This is the **critical foundation** that enables clean, testable, reusable code throughout Phase 1.

---

## What You'll Learn

✅ **Python package structure** - How to organize a professional Python project
✅ **Abstract base classes** - Using ABC to define interfaces
✅ **Composition patterns** - Building reusable components
✅ **Test fixtures** - Creating reusable test data
✅ **NumPy design patterns** - Memory-efficient array handling
✅ **Professional structure** - How trading libraries are organized in production

---

## Technology Choices & Rationale

### Why NumPy (not Pandas)?

**NumPy**: Low-level array operations, memory efficient, mathematical operations
**Pandas**: Higher-level, more overhead, better for tabular data with labels

**Our choice: NumPy** because:
- Indicators operate on pure numerical arrays (prices, returns)
- Performance matters (real-time trading signals)
- Simpler mental model for mathematical operations
- Easier to understand what's happening mechanically

We'll use **Pandas for data loading** in backtester, but indicators work on NumPy arrays.

### Why pytest?

**unittest**: Verbose, heavy boilerplate
**pytest**: Concise, powerful fixtures, great for data-driven testing
**Our choice: pytest** because:
- Cleaner test syntax (no class boilerplate)
- Excellent fixture system for test data
- Better assertion messages
- Easier parameterized testing

### Why Abstract Base Classes?

```python
from abc import ABC, abstractmethod

class Indicator(ABC):
    @abstractmethod
    def calculate(self, data: np.ndarray) -> np.ndarray:
        pass
```

**Benefits:**
- Enforces interface consistency across all indicators
- Prevents incomplete implementations (must implement `calculate`)
- Self-documenting code (clear what subclasses must do)
- Enables duck typing with type checking

### Why This Directory Structure?

```
src/
├── indicators/          # All indicator implementations
│   ├── __init__.py
│   ├── base.py          # Indicator base class
│   ├── moving_average.py # SMA, EMA
│   ├── momentum.py       # RSI, MACD
│   └── volatility.py     # Bollinger Bands
├── data/
│   ├── __init__.py
│   └── loader.py        # Market data loading
└── backtester/
    ├── __init__.py
    └── engine.py        # Backtesting framework

tests/
├── __init__.py
├── conftest.py          # Pytest fixtures (shared test data)
├── test_indicators.py   # All indicator tests
└── fixtures/
    └── sample_data.npy  # Test data file
```

**Why this structure:**
- `src/` separates source code from tests (importable as package)
- `indicators/` groups all indicator logic
- Each indicator type in separate module (moving averages, momentum, volatility)
- `tests/conftest.py` centralizes test fixtures (reusable test data)
- `tests/fixtures/` stores binary test data files

---

## Implementation Guide

### Step 1: Create Base Indicator Class

**File**: `src/indicators/base.py`

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.

    All indicators inherit from this class and must implement:
    1. __init__(self, ...) - Store parameters
    2. calculate(self, data) - Compute the indicator

    Example usage:
        sma = SMA(period=20)
        result = sma.calculate(prices)  # or sma(prices)
    """

    def __init__(self, period: int):
        """
        Initialize indicator.

        Args:
            period: Lookback period for calculation

        Raises:
            ValueError: If period < 1
        """
        if not isinstance(period, int) or period < 1:
            raise ValueError(f"period must be positive integer, got {period}")

        self.period = period
        self._last_result = None

    @abstractmethod
    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the indicator on input data.

        Args:
            data: Array of prices/values, shape (N,)

        Returns:
            Array of indicator values, shape (N,)
            First (period-1) values typically NaN

        Raises:
            ValueError: If data is invalid
            TypeError: If data type is wrong
        """
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Allow indicator to be called as function: indicator(data)"""
        return self.calculate(data)

    def _validate_input(self, data: np.ndarray, min_length: Optional[int] = None) -> None:
        """
        Validate input data before calculation.

        Args:
            data: Input array to validate
            min_length: Minimum required length (defaults to period)

        Raises:
            TypeError: If data is not numpy array
            ValueError: If data shape is invalid
            ValueError: If data has insufficient length
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(data)}")

        if data.ndim != 1:
            raise ValueError(f"data must be 1D array, got shape {data.shape}")

        min_len = min_length or self.period
        if len(data) < min_len:
            raise ValueError(
                f"data length {len(data)} < minimum required {min_len}"
            )

    def _create_output(self, length: int) -> np.ndarray:
        """Create output array initialized with NaN."""
        return np.full(length, np.nan, dtype=np.float64)
```

### Step 2: Create Backtester

**File**: `src/backtester/engine.py`

```python
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    signals: np.ndarray          # Buy (1), Sell (-1), Hold (0)
    entry_prices: np.ndarray     # Price when signal triggered
    exit_prices: np.ndarray      # Price when position closed
    trade_returns: np.ndarray    # Per-trade return percentage
    cumulative_return: float     # Total return %
    sharpe_ratio: float          # Risk-adjusted return
    max_drawdown: float          # Worst peak-to-trough decline
    win_rate: float              # % of profitable trades
    num_trades: int              # Total number of trades

    def summary(self) -> str:
        """Print human-readable summary."""
        return f"""
Backtest Results:
  Total Return: {self.cumulative_return:.2f}%
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.2f}%
  Win Rate: {self.win_rate:.2f}%
  Total Trades: {self.num_trades}
"""


def backtest_signal(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001
) -> BacktestResult:
    """
    Simple backtester for indicator signals.

    Args:
        prices: Array of closing prices
        signals: Array of signals (1=buy, -1=sell, 0=hold)
        initial_capital: Starting capital
        transaction_cost: Percentage cost per trade (0.001 = 0.1%)

    Returns:
        BacktestResult with performance metrics
    """
    if len(prices) != len(signals):
        raise ValueError("prices and signals must have same length")

    if len(prices) < 2:
        raise ValueError("Need at least 2 price points")

    # Track trades
    position = 0              # 0=no position, 1=long
    entry_price = None
    trades = []

    # Generate trade prices
    entry_prices = np.full(len(prices), np.nan)
    exit_prices = np.full(len(prices), np.nan)

    for i, (price, signal) in enumerate(zip(prices, signals)):
        # Entry signal
        if signal == 1 and position == 0:
            position = 1
            entry_price = price * (1 + transaction_cost)
            entry_prices[i] = entry_price

        # Exit signal
        elif signal == -1 and position == 1:
            exit_price = price * (1 - transaction_cost)
            exit_prices[i] = exit_price

            trade_return = (exit_price - entry_price) / entry_price
            trades.append(trade_return)

            position = 0
            entry_price = None

    # Calculate metrics
    trade_returns = np.array(trades) if trades else np.array([0.0])
    cumulative_return = np.prod(1 + trade_returns) - 1

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(trades) > 1 and np.std(trades) > 0:
        sharpe_ratio = np.mean(trades) / np.std(trades) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumul = np.cumprod(1 + trade_returns)
    running_max = np.maximum.accumulate(cumul)
    drawdown = (cumul - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Win rate
    wins = np.sum(trade_returns > 0)
    win_rate = wins / len(trades) if trades else 0.0

    return BacktestResult(
        signals=signals,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        trade_returns=trade_returns,
        cumulative_return=cumulative_return * 100,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown * 100,
        win_rate=win_rate * 100,
        num_trades=len(trades)
    )
```

### Step 3: Create Package `__init__.py` Files

**File**: `src/indicators/__init__.py`

```python
"""Technical indicators for trading signals."""

from .base import Indicator

__all__ = ["Indicator"]
```

**File**: `src/backtester/__init__.py`

```python
"""Backtesting framework."""

from .engine import backtest_signal, BacktestResult

__all__ = ["backtest_signal", "BacktestResult"]
```

**File**: `src/__init__.py`

```python
"""AlphaSignal - LLM-powered trading signal generation."""

__version__ = "0.1.0"
```

### Step 4: Create pytest Configuration

**File**: `tests/conftest.py`

```python
"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_prices():
    """
    Generate sample price data for testing.

    Returns:
        Array of 100 prices with realistic movements
    """
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 100)
    prices = 100 * np.cumprod(1 + returns)
    return prices


@pytest.fixture
def constant_prices():
    """Array of constant prices (for edge case testing)."""
    return np.full(50, 100.0)


@pytest.fixture
def small_price_array():
    """Very small array (10 prices)."""
    return np.array([100, 101, 102, 101, 100, 99, 98, 99, 100, 101], dtype=float)


@pytest.fixture
def prices_with_nan():
    """Prices containing NaN values."""
    prices = np.array([100, 101, np.nan, 102, 103, 104, 105, 106, 107, 108])
    return prices


@pytest.fixture
def uptrend_prices():
    """Strongly uptrending prices."""
    return np.linspace(100, 150, 50)


@pytest.fixture
def downtrend_prices():
    """Strongly downtrending prices."""
    return np.linspace(150, 100, 50)
```

**File**: `tests/__init__.py`

```python
"""Tests for AlphaSignal indicators."""
```

### Step 5: Create Setup Validation Tests

**File**: `tests/test_setup.py`

```python
"""Validate repository setup and infrastructure."""

import pytest
import numpy as np
from src.indicators.base import Indicator


def test_indicator_abstract_base_class():
    """Indicator base class should be abstract."""
    with pytest.raises(TypeError):
        # Cannot instantiate abstract class
        Indicator(period=20)


def test_indicator_subclass_must_implement_calculate():
    """Subclass must implement calculate method."""

    class IncompleteIndicator(Indicator):
        pass

    with pytest.raises(TypeError):
        IncompleteIndicator(period=20)


def test_indicator_validation_period():
    """Test period validation."""

    class DummyIndicator(Indicator):
        def calculate(self, data):
            return data

    # Valid period
    ind = DummyIndicator(period=20)
    assert ind.period == 20

    # Invalid periods
    with pytest.raises(ValueError):
        DummyIndicator(period=0)

    with pytest.raises(ValueError):
        DummyIndicator(period=-1)

    with pytest.raises(ValueError):
        DummyIndicator(period=1.5)  # Not int


def test_indicator_validation_input(sample_prices):
    """Test input validation."""

    class DummyIndicator(Indicator):
        def calculate(self, data):
            self._validate_input(data)
            return data

    ind = DummyIndicator(period=20)

    # Valid input
    result = ind(sample_prices)
    assert len(result) == len(sample_prices)

    # Invalid: not numpy array
    with pytest.raises(TypeError):
        ind([1, 2, 3])

    # Invalid: 2D array
    with pytest.raises(ValueError):
        ind(np.array([[1, 2], [3, 4]]))

    # Invalid: too short
    with pytest.raises(ValueError):
        ind(np.array([1, 2, 3]))  # Only 3 values, need period=20


def test_indicator_callable():
    """Indicator should be callable as function."""

    class DummyIndicator(Indicator):
        def calculate(self, data):
            return data * 2

    ind = DummyIndicator(period=1)
    prices = np.array([1, 2, 3])

    # Should work both ways
    result1 = ind.calculate(prices)
    result2 = ind(prices)

    np.testing.assert_array_equal(result1, result2)


def test_backtester_import():
    """Backtester module should be importable."""
    from src.backtester import backtest_signal, BacktestResult

    assert callable(backtest_signal)
    assert BacktestResult is not None


def test_backtester_basic():
    """Backtester should work with simple signals."""
    from src.backtester import backtest_signal

    prices = np.array([100, 101, 102, 103, 104, 105])
    signals = np.array([1, 0, 0, -1, 0, 0])  # Buy, hold, hold, sell, hold, hold

    result = backtest_signal(prices, signals)

    assert result.num_trades == 1
    assert result.cumulative_return > 0  # Simple uptrend, should profit
    assert 0 <= result.win_rate <= 100


def test_backtester_validation():
    """Backtester should validate inputs."""
    from src.backtester import backtest_signal

    prices = np.array([100, 101, 102])
    signals = np.array([1, 0])  # Wrong length

    with pytest.raises(ValueError):
        backtest_signal(prices, signals)


def test_fixtures_available(sample_prices, constant_prices, small_price_array):
    """Test fixtures should be available."""
    assert len(sample_prices) == 100
    assert np.all(constant_prices == 100)
    assert len(small_price_array) == 10
```

### Step 6: Install and Validate

```bash
# Install package in development mode
pip install -e .

# Run setup tests
pytest tests/test_setup.py -v

# Run all tests
pytest tests/ -v --tb=short

# Check code coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Unit Tests to Implement

All unit tests are in `tests/test_setup.py`. Run them to validate:

1. ✅ Indicator base class is abstract
2. ✅ Subclasses must implement calculate
3. ✅ Period validation works
4. ✅ Input validation works (type, shape, length)
5. ✅ Callable interface works
6. ✅ Backtester imports correctly
7. ✅ Backtester calculates metrics
8. ✅ Backtester validates inputs
9. ✅ Test fixtures are available

---

## Success Criteria

- [ ] Directory structure created exactly as specified
- [ ] `src/indicators/base.py` implemented with full `Indicator` class
- [ ] `src/backtester/engine.py` implemented with `backtest_signal` and `BacktestResult`
- [ ] All `__init__.py` files created
- [ ] `tests/conftest.py` created with 5 fixtures
- [ ] `tests/test_setup.py` created with 9 validation tests
- [ ] All 9 tests passing: `pytest tests/test_setup.py -v`
- [ ] 100% code coverage on `src/` modules
- [ ] Can successfully import: `from src.indicators import Indicator`
- [ ] Can successfully import: `from src.backtester import backtest_signal`

---

## Validation Checklist

Before moving to Task 1.1, verify:

```bash
# 1. All tests pass
pytest tests/test_setup.py -v

# 2. Coverage is 100%
pytest tests/ --cov=src

# 3. Imports work
python -c "from src.indicators import Indicator; print('✓ Indicator imported')"
python -c "from src.backtester import backtest_signal; print('✓ backtest_signal imported')"

# 4. Base class prevents instantiation
python -c "from src.indicators import Indicator; Indicator(20)"
# Should fail with: TypeError: Can't instantiate abstract class...

# 5. Dummy indicator works
python << 'EOF'
import numpy as np
from src.indicators import Indicator

class TestInd(Indicator):
    def calculate(self, data):
        return data

ind = TestInd(period=5)
prices = np.array([1,2,3,4,5,6,7,8,9,10])
result = ind(prices)
print(f"✓ Dummy indicator works: {result}")
EOF
```

---

## Common Mistakes to Avoid

❌ **Mistake 1**: Forgetting `__init__.py` files
✅ **Fix**: Create empty `__init__.py` in each package directory

❌ **Mistake 2**: Not validating input in base class
✅ **Fix**: Use `_validate_input()` before calculation

❌ **Mistake 3**: Making Indicator concrete (not abstract)
✅ **Fix**: Use `@abstractmethod` on `calculate()`

❌ **Mistake 4**: Backtester not handling NaN values
✅ **Fix**: Use `np.full()` to pre-fill with NaN

❌ **Mistake 5**: Test fixtures not parameterized
✅ **Fix**: Use pytest fixtures, not hardcoded data

---

## Resources

- [Python ABC Documentation](https://docs.python.org/3/library/abc.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Professional Python Package Structure](https://packaging.python.org/tutorials/packaging-projects/)

---

## Time Estimate

⏱️ **3-4 hours** total

- 1 hour: Create directory structure + base files
- 1 hour: Implement Indicator base class
- 1 hour: Implement Backtester
- 30 min: Write conftest fixtures
- 1 hour: Write and validate setup tests

---

## Next Steps

Once Task 0 is complete:
- ✅ All 5 Phase 1 indicators (Tasks 1.1-1.5) can build on this foundation
- ✅ Each indicator simply inherits from `Indicator` and implements `calculate()`
- ✅ Backtester is ready to validate signals
- ✅ Test fixtures enable quick testing

**Ready to proceed to Task 1.1 (SMA) once this is complete!**
