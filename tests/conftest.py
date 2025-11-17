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
