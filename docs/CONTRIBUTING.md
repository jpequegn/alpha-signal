# Contributing Guide

Thank you for your interest in contributing to AlphaSignal! This guide explains how to contribute code, documentation, and improvements.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Adding New Indicators](#adding-new-indicators)
- [Testing Requirements](#testing-requirements)
- [Documentation Requirements](#documentation-requirements)
- [Pull Request Process](#pull-request-process)
- [Code Review Guidelines](#code-review-guidelines)

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Basic understanding of technical indicators
- Familiarity with NumPy

### Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/alpha-signal.git
cd alpha-signal

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/alpha-signal.git
```

## Development Setup

### Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n alpha-signal python=3.12
conda activate alpha-signal
```

### Install Dependencies

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing, linting, etc.
```

### Verify Setup

```bash
# Run tests
pytest tests/ -v

# Check code quality
flake8 src/ tests/
mypy src/

# Expected output: All tests pass, no lint errors
```

## Coding Standards

### Code Style

**Python Style Guide:** PEP 8

```python
# ✅ Good
def calculate(self, data: np.ndarray) -> np.ndarray:
    """Calculate indicator values."""
    if len(data) < self.period:
        return self._create_output(len(data))
    return self._compute_values(data)


# ❌ Bad
def calculate(self,data:np.ndarray)->np.ndarray:
    if len(data)<self.period:
        return self._create_output(len(data))
    return self._compute_values(data)
```

### Naming Conventions

```python
# Constants - UPPER_CASE
DEFAULT_PERIOD = 20
NUM_STANDARD_DEVS = 2.0

# Classes - CapitalCase
class SimpleMovingAverage:
    pass

# Functions/Methods - snake_case
def calculate_indicator(prices):
    pass

# Private - leading underscore
def _validate_input(self, data):
    pass

# Protected - leading underscore
self._period = period
```

### Type Hints

```python
# ✅ Always use type hints
def calculate(self, data: np.ndarray) -> np.ndarray:
    """Calculate indicator.

    Args:
        data: Price array

    Returns:
        Indicator values
    """

# ❌ Never omit type hints
def calculate(self, data):
    """Calculate indicator."""
```

### Docstrings

**Format:** Google-style docstrings

```python
def calculate(self, data: np.ndarray) -> np.ndarray:
    """Calculate SMA values.

    Formula:
        SMA(t) = (P(t) + P(t-1) + ... + P(t-n+1)) / n

    Args:
        data: Price array, shape (N,)

    Returns:
        SMA values, shape (N,) with first (period-1) as NaN

    Raises:
        TypeError: If data not numpy array
        ValueError: If data not 1D

    Example:
        >>> prices = np.array([100, 101, 102])
        >>> sma = SMA(period=2)
        >>> result = sma(prices)
    """
```

### Import Organization

```python
# Standard library
import abc
from abc import ABC, abstractmethod

# Third-party
import numpy as np
import pytest

# Local
from src.indicators.base import Indicator

# Within each section: alphabetical order
```

## Adding New Indicators

### Step 1: Create Indicator File

```bash
# For phase 2 advanced indicators
touch src/indicators/advanced.py
```

### Step 2: Implement Indicator Class

```python
"""Advanced indicators (ATR, Stochastic, etc.)."""

import numpy as np
from .base import Indicator


class ATR(Indicator):
    """
    Average True Range (ATR) - Volatility indicator.

    Mathematical Definition:
        TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        ATR = SMA(TR, period)

    Key Properties:
        - Measures volatility in price units
        - Not bounded (depends on price level)
        - Higher ATR = more volatility
        - First (period-1) values are NaN

    Example:
        >>> high = np.array([101, 102, 103])
        >>> low = np.array([99, 100, 101])
        >>> close = np.array([100, 101, 102])
        >>> atr = ATR(period=14)
        >>> result = atr(high, low, close)
    """

    def __init__(self, period: int = 14):
        """Initialize ATR."""
        super().__init__(period=period)

    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate ATR.

        Args:
            high: High prices, shape (N,)
            low: Low prices, shape (N,)
            close: Close prices, shape (N,)

        Returns:
            ATR values, shape (N,) with first (period-1) as NaN

        Raises:
            TypeError: If inputs not numpy arrays
            ValueError: If not same length or 1D
        """
        # Validate all inputs
        for data in [high, low, close]:
            if not isinstance(data, np.ndarray):
                raise TypeError("All inputs must be numpy arrays")
            if data.ndim != 1:
                raise ValueError("All inputs must be 1D")

        if len(high) != len(low) or len(low) != len(close):
            raise ValueError("All inputs must have same length")

        # Create output
        output = self._create_output(len(high))

        if len(high) < self.period:
            return output

        # Calculate true range
        tr = np.zeros(len(high) - 1)
        for i in range(1, len(high)):
            tr_val = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr[i-1] = tr_val

        # Calculate ATR as SMA of TR
        atr_vals = np.convolve(tr, np.ones(self.period) / self.period, mode='valid')
        output[self.period:] = atr_vals

        return output
```

### Step 3: Export in __init__.py

```python
# src/indicators/__init__.py
from .advanced import ATR

__all__ = ["Indicator", "SMA", "EMA", "RSI", "MACD", "BollingerBands", "ATR"]
```

### Step 4: Add Tests

```python
# tests/test_indicators.py
class TestATR:
    """Test cases for ATR indicator."""

    def test_atr_basic_calculation(self):
        """Test ATR with known values."""
        high = np.array([101, 102, 103])
        low = np.array([99, 100, 101])
        close = np.array([100, 101, 102])

        atr = ATR(period=1)
        result = atr(high, low, close)

        # First value is NaN (initialization)
        assert np.isnan(result[0])

        # Check calculation
        assert len(result) == 3

    def test_atr_high_volatility(self):
        """Test ATR increases with volatility."""
        # Low volatility
        high_calm = np.array([100.1, 100.2, 100.3])
        low_calm = np.array([99.9, 100.0, 100.1])

        # High volatility
        high_vol = np.array([105, 110, 115])
        low_vol = np.array([95, 100, 105])

        atr = ATR(period=2)
        result_calm = atr(high_calm, low_calm, np.array([100, 100.1, 100.2]))
        result_vol = atr(high_vol, low_vol, np.array([100, 105, 110]))

        # High volatility should have higher ATR
        assert np.nanmean(result_vol) > np.nanmean(result_calm)

    # Add more tests...
```

## Testing Requirements

### Test Coverage

**Minimum: 80% coverage, Target: 90%+**

```bash
# Run tests with coverage
pytest tests/ --cov=src/indicators --cov-report=term-missing

# Generate HTML report
pytest tests/ --cov=src/indicators --cov-report=html
# Open htmlcov/index.html
```

### Test Structure

```python
class TestNewIndicator:
    """Test cases for new indicator."""

    def test_basic_calculation(self):
        """Test basic functionality."""
        # Arrange
        prices = np.array([100, 101, 102])
        indicator = NewIndicator(period=2)

        # Act
        result = indicator(prices)

        # Assert
        assert len(result) == 3
        assert np.isnan(result[0])
        assert not np.isnan(result[1])

    def test_with_fixture(self, sample_prices):
        """Test with provided fixture."""
        indicator = NewIndicator()
        result = indicator(sample_prices)
        assert len(result) == len(sample_prices)

    def test_edge_case_constant_prices(self):
        """Test edge case: constant prices."""
        prices = np.full(50, 100.0)
        indicator = NewIndicator()
        result = indicator(prices)
        # Verify expected behavior...

    def test_error_handling(self):
        """Test error handling."""
        prices = [100, 101, 102]  # List, not array
        indicator = NewIndicator()

        with pytest.raises(TypeError):
            indicator(prices)

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            NewIndicator(period=-1)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test class
pytest tests/test_indicators.py::TestNewIndicator

# Run specific test
pytest tests/test_indicators.py::TestNewIndicator::test_basic_calculation

# Run with verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x

# Run with coverage
pytest tests/ --cov=src/indicators
```

## Documentation Requirements

### Code Documentation

**Every public method needs:**
- Docstring with description
- Args section with type and description
- Returns section with type and description
- Raises section if applicable
- Example section for non-trivial methods

```python
def get_signals(self, rsi_values: np.ndarray) -> np.ndarray:
    """
    Generate trading signals.

    Args:
        rsi_values: RSI values from calculate()

    Returns:
        Signal array: 1 (overbought), -1 (oversold), 0 (neutral)

    Example:
        rsi = RSI()
        rsi_values = rsi(prices)
        signals = rsi.get_signals(rsi_values)
    """
```

### Documentation Files

**New Indicator Documentation:**

Create `docs/phase2/TASK_X_INDICATOR_NAME.md`:

```markdown
# Task X: Implement [Indicator Name]

## What is [Indicator]?
- Purpose and use case
- Mathematical formula
- Key properties

## Calculation
- Step-by-step process
- Edge cases
- Efficiency considerations

## Trading Applications
- Common strategies
- Signal interpretation
- Pros and cons

## Implementation Details
- Algorithm choice rationale
- Optimization techniques
- Test coverage strategy

## References
- Academic papers
- External resources
```

## Pull Request Process

### 1. Create Feature Branch

```bash
# Sync with main
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feat/add-atr-indicator
# Or for fixes: git checkout -b fix/rsi-edge-case
```

### 2. Implement Changes

```bash
# Make changes
# Update tests
# Update documentation
# Commit regularly with clear messages
git add src/ tests/ docs/
git commit -m "Add ATR indicator implementation

- Implement ATR class with period validation
- Add get_signals() for volatility interpretation
- 8 unit tests covering all code paths
- Update API reference documentation

Closes #123"
```

### 3. Keep Branch Updated

```bash
# Fetch latest changes
git fetch upstream

# Rebase (preferred) or merge
git rebase upstream/main
# Or: git merge upstream/main
```

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feat/add-atr-indicator

# Create PR on GitHub
# Write clear PR description (see PR Template below)
```

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] New indicator
- [ ] Bug fix
- [ ] Documentation
- [ ] Performance improvement

## Changes
- List specific changes
- Each bullet point is one change

## Testing
- [ ] Added unit tests (minimum 80% coverage)
- [ ] All existing tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Updated docstrings
- [ ] Updated API reference
- [ ] Added task documentation
- [ ] Updated README if needed

## Checklist
- [ ] Code follows PEP 8
- [ ] Type hints added
- [ ] Error handling complete
- [ ] No breaking changes
```

## Code Review Guidelines

### What Reviewers Look For

**Functionality:**
- Does it solve the problem?
- Are edge cases handled?
- Is error handling appropriate?

**Code Quality:**
- Does it follow PEP 8?
- Are type hints present?
- Are docstrings complete?
- Is code readable?

**Testing:**
- Is coverage sufficient (80%+)?
- Are tests meaningful?
- Do tests cover edge cases?
- Are all assertions clear?

**Performance:**
- Is algorithmic complexity reasonable?
- Are NumPy operations used efficiently?
- Are there unnecessary copies?

**Documentation:**
- Are docstrings clear?
- Is mathematical formula documented?
- Are examples provided?
- Is user guide updated?

### Responding to Review Comments

**Do:**
- Thank reviewers for feedback
- Ask clarifying questions
- Make suggested changes
- Explain your reasoning if you disagree

**Don't:**
- Be defensive
- Ignore comments
- Make unrelated changes
- Request approval before addressing feedback

## Questions?

- Check existing documentation
- Review similar implementations
- Open an issue with your question
- Ask on discussions

## Contributor Recognition

Contributors are recognized in:
- Git commit history
- CONTRIBUTORS.md file
- GitHub contributor graphs
- Release notes

---

Thank you for contributing to AlphaSignal! 🙏
