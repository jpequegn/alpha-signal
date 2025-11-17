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


def test_indicator_create_output():
    """Test output array creation."""

    class DummyIndicator(Indicator):
        def calculate(self, data):
            return self._create_output(len(data))

    ind = DummyIndicator(period=5)
    prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = ind(prices)

    assert len(result) == 10
    assert np.all(np.isnan(result))


def test_backtester_edge_cases():
    """Test backtester with edge cases."""
    from src.backtester import backtest_signal

    # No trades
    prices = np.array([100, 101, 102, 103])
    signals = np.array([0, 0, 0, 0])
    result = backtest_signal(prices, signals)
    assert result.num_trades == 0

    # Single trade
    prices = np.array([100, 105, 110])
    signals = np.array([1, -1, 0])
    result = backtest_signal(prices, signals)
    assert result.num_trades == 1


def test_backtester_sharpe_calculation():
    """Test Sharpe ratio with multiple trades."""
    from src.backtester import backtest_signal

    prices = np.array([100, 105, 100, 110, 105, 115, 110, 120])
    signals = np.array([1, -1, 1, -1, 1, -1, 0, 0])

    result = backtest_signal(prices, signals)
    assert result.num_trades == 3
    assert isinstance(result.sharpe_ratio, float)
