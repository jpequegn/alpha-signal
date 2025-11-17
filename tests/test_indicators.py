"""Unit tests for technical indicators."""

import pytest
import numpy as np
from src.indicators import SMA


class TestSMA:
    """Test cases for Simple Moving Average indicator."""

    def test_sma_basic_calculation(self):
        """Test SMA with simple arithmetic."""
        # Test case: prices = [100, 101, 102, 103, 104, 105]
        # SMA(3) should be: [NaN, NaN, 101, 102, 103, 104]
        prices = np.array([100, 101, 102, 103, 104, 105], dtype=float)
        sma = SMA(period=3)
        result = sma(prices)

        # First 2 values should be NaN (period-1)
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # SMA(3) at index 2 = (100+101+102)/3 = 101
        assert np.isclose(result[2], 101.0)

        # SMA(3) at index 3 = (101+102+103)/3 = 102
        assert np.isclose(result[3], 102.0)

        # SMA(3) at index 4 = (102+103+104)/3 = 103
        assert np.isclose(result[4], 103.0)

        # SMA(3) at index 5 = (103+104+105)/3 = 104
        assert np.isclose(result[5], 104.0)

    def test_sma_with_constant_values(self):
        """Test SMA with constant prices (edge case)."""
        # If all prices are the same, SMA should equal that price
        prices = np.full(50, 100.0)
        sma = SMA(period=20)
        result = sma(prices)

        # First 19 values should be NaN
        assert np.all(np.isnan(result[:19]))

        # All SMA values should equal 100.0
        assert np.all(np.isclose(result[19:], 100.0))

    def test_sma_insufficient_data(self):
        """Test SMA when data length < period."""
        prices = np.array([100, 101, 102])
        sma = SMA(period=20)
        result = sma(prices)

        # All values should be NaN (insufficient data)
        assert np.all(np.isnan(result))

    def test_sma_minimal_data(self):
        """Test SMA with minimum required data."""
        # period=5 requires 5 values minimum
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = SMA(period=5)
        result = sma(prices)

        # First 4 values are NaN
        assert np.all(np.isnan(result[:4]))

        # SMA at index 4 = (1+2+3+4+5)/5 = 3.0
        assert np.isclose(result[4], 3.0)

    def test_sma_different_periods(self):
        """Test SMA with different period values."""
        prices = np.linspace(100, 150, 50)  # 50 values from 100 to 150

        # Test period=5
        sma5 = SMA(period=5)
        result5 = sma5(prices)
        assert np.sum(np.isnan(result5)) == 4  # First 4 are NaN
        assert len(result5) == 50

        # Test period=10
        sma10 = SMA(period=10)
        result10 = sma10(prices)
        assert np.sum(np.isnan(result10)) == 9  # First 9 are NaN
        assert len(result10) == 50

        # Test period=20
        sma20 = SMA(period=20)
        result20 = sma20(prices)
        assert np.sum(np.isnan(result20)) == 19  # First 19 are NaN
        assert len(result20) == 50

    def test_sma_realistic_data(self, sample_prices):
        """Test SMA with realistic price data."""
        sma = SMA(period=20)
        result = sma(sample_prices)

        # Check shape and NaN count
        assert len(result) == len(sample_prices)
        assert np.sum(np.isnan(result)) == 19  # First 19 should be NaN

        # Check that SMA values are reasonable (between min and max of data)
        valid_result = result[~np.isnan(result)]
        assert np.all(valid_result >= np.nanmin(sample_prices))
        assert np.all(valid_result <= np.nanmax(sample_prices))

        # Check that SMA is a smooth version of prices
        # (SMA should reduce variance compared to raw prices)
        price_variance = np.var(sample_prices)
        sma_variance = np.var(valid_result)
        assert sma_variance < price_variance  # SMA should be smoother


class TestSMAValidation:
    """Test SMA validation and error handling."""

    def test_sma_with_uptrend(self, uptrend_prices):
        """Test SMA during uptrend."""
        sma = SMA(period=10)
        result = sma(uptrend_prices)

        # In uptrend, prices should be above SMA
        valid_result = result[~np.isnan(result)]
        prices_after_nan = uptrend_prices[len(uptrend_prices) - len(valid_result):]

        # Most prices in uptrend should be above SMA
        above_sma = np.sum(prices_after_nan >= valid_result)
        assert above_sma > len(valid_result) * 0.8  # At least 80% above SMA

    def test_sma_with_downtrend(self, downtrend_prices):
        """Test SMA during downtrend."""
        sma = SMA(period=10)
        result = sma(downtrend_prices)

        # In downtrend, prices should be below SMA
        valid_result = result[~np.isnan(result)]
        prices_after_nan = downtrend_prices[len(downtrend_prices) - len(valid_result):]

        # Most prices in downtrend should be below SMA
        below_sma = np.sum(prices_after_nan <= valid_result)
        assert below_sma > len(valid_result) * 0.8  # At least 80% below SMA
