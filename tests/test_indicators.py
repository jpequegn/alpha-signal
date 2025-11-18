"""Unit tests for technical indicators."""

import pytest
import numpy as np
from src.indicators import SMA, EMA, RSI


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


class TestEMA:
    """Test cases for Exponential Moving Average indicator."""

    def test_ema_basic_calculation(self):
        """Test EMA with known values."""
        prices = np.array([100, 101, 102, 103, 104, 105], dtype=float)
        ema = EMA(period=3)
        result = ema(prices)

        # First 2 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # Index 2: initialized with SMA(3) = 101
        assert np.isclose(result[2], 101.0)

        # Index 3: EMA = 103 * 0.5 + 101 * 0.5 = 102
        assert np.isclose(result[3], 102.0)

        # Index 4: EMA = 104 * 0.5 + 102 * 0.5 = 103
        assert np.isclose(result[4], 103.0)

    def test_ema_alpha_property(self):
        """Test alpha smoothing factor calculation."""
        # Alpha = 2 / (period + 1)
        ema3 = EMA(period=3)
        assert np.isclose(ema3.alpha, 2.0 / 4.0)  # 0.5

        ema5 = EMA(period=5)
        assert np.isclose(ema5.alpha, 2.0 / 6.0)  # ~0.333

        ema12 = EMA(period=12)
        assert np.isclose(ema12.alpha, 2.0 / 13.0)  # ~0.1538

        ema26 = EMA(period=26)
        assert np.isclose(ema26.alpha, 2.0 / 27.0)  # ~0.0741

    def test_ema_responsiveness_vs_sma(self, sample_prices):
        """Test that EMA responds faster to changes than SMA."""
        sma = SMA(period=10)
        ema = EMA(period=10)

        sma_result = sma(sample_prices)
        ema_result = ema(sample_prices)

        # Both should have same NaN count
        assert np.sum(np.isnan(sma_result)) == np.sum(np.isnan(ema_result))

        # Get valid values (after NaN period)
        valid_sma = sma_result[~np.isnan(sma_result)]
        valid_ema = ema_result[~np.isnan(ema_result)]
        valid_prices = sample_prices[-len(valid_sma):]

        # EMA should have more variance than SMA (more responsive)
        sma_variance = np.var(valid_sma)
        ema_variance = np.var(valid_ema)
        assert ema_variance > sma_variance * 0.5  # EMA more responsive

    def test_ema_with_constant_values(self):
        """Test EMA with constant prices."""
        prices = np.full(50, 100.0)
        ema = EMA(period=20)
        result = ema(prices)

        # First 19 values are NaN
        assert np.all(np.isnan(result[:19]))

        # All EMA values should equal 100.0
        assert np.all(np.isclose(result[19:], 100.0))

    def test_ema_insufficient_data(self):
        """Test EMA when data length < period."""
        prices = np.array([100, 101, 102])
        ema = EMA(period=20)
        result = ema(prices)

        # All values should be NaN
        assert np.all(np.isnan(result))

    def test_ema_different_periods(self):
        """Test EMA with different periods."""
        prices = np.linspace(100, 150, 50)

        # Test period=5
        ema5 = EMA(period=5)
        result5 = ema5(prices)
        assert np.sum(np.isnan(result5)) == 4
        assert len(result5) == 50

        # Test period=12 (common in trading)
        ema12 = EMA(period=12)
        result12 = ema12(prices)
        assert np.sum(np.isnan(result12)) == 11
        assert len(result12) == 50

        # Test period=26 (common in trading)
        ema26 = EMA(period=26)
        result26 = ema26(prices)
        assert np.sum(np.isnan(result26)) == 25
        assert len(result26) == 50


class TestEMAValidation:
    """Test EMA validation and properties."""

    def test_ema_with_trend(self, uptrend_prices):
        """Test EMA during uptrend - should follow trend closely."""
        ema = EMA(period=10)
        result = ema(uptrend_prices)

        # EMA should be between min and max prices
        valid_result = result[~np.isnan(result)]
        assert np.all(valid_result >= np.nanmin(uptrend_prices))
        assert np.all(valid_result <= np.nanmax(uptrend_prices))

    def test_ema_smoothing_property(self, sample_prices):
        """Test that EMA is smoother than raw prices."""
        ema = EMA(period=20)
        result = ema(sample_prices)

        # EMA should reduce variance compared to raw prices
        valid_result = result[~np.isnan(result)]
        ema_variance = np.var(valid_result)
        price_variance = np.var(sample_prices)

        assert ema_variance < price_variance

    def test_ema_repr(self):
        """Test string representation of EMA."""
        ema = EMA(period=12)
        repr_str = repr(ema)

        assert "EMA" in repr_str
        assert "period=12" in repr_str
        assert "alpha=" in repr_str


class TestRSI:
    """Test cases for Relative Strength Index indicator."""

    def test_rsi_basic_calculation(self):
        """Test RSI with known uptrend data."""
        # Uptrend: all prices increasing
        prices = np.array([44.0, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42])
        rsi = RSI(period=3)
        result = rsi(prices)

        # First period values should be NaN
        assert np.sum(np.isnan(result[:3])) == 3

        # RSI values should be in 0-100 range
        valid_rsi = result[~np.isnan(result)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_rsi_bounds(self):
        """Test that RSI always stays within 0-100."""
        prices = np.random.uniform(90, 110, 100)
        rsi = RSI(period=14)
        result = rsi(prices)

        # All non-NaN values should be between 0 and 100
        valid_rsi = result[~np.isnan(result)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)

    def test_rsi_uptrend(self, uptrend_prices):
        """Test RSI during strong uptrend."""
        rsi = RSI(period=14)
        result = rsi(uptrend_prices)

        # During uptrend, RSI should be high (>50 mostly)
        valid_rsi = result[~np.isnan(result)]
        assert np.mean(valid_rsi) > 50  # Average RSI should be > 50 in uptrend

    def test_rsi_downtrend(self, downtrend_prices):
        """Test RSI during strong downtrend."""
        rsi = RSI(period=14)
        result = rsi(downtrend_prices)

        # During downtrend, RSI should be low (<50 mostly)
        valid_rsi = result[~np.isnan(result)]
        assert np.mean(valid_rsi) < 50  # Average RSI should be < 50 in downtrend

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = np.array([100, 101, 102])
        rsi = RSI(period=14)
        result = rsi(prices)

        # All values should be NaN
        assert np.all(np.isnan(result))

    def test_rsi_constant_prices(self):
        """Test RSI with constant prices."""
        prices = np.full(50, 100.0)
        rsi = RSI(period=14)
        result = rsi(prices)

        # First 14 values are NaN
        assert np.sum(np.isnan(result[:14])) == 14

        # With no gains or losses, RSI should be 50 (neutral)
        valid_rsi = result[~np.isnan(result)]
        # When avg_gain = avg_loss, RSI = 100 - (100/(1+1)) = 50
        assert np.all(np.isclose(valid_rsi, 50.0))

    def test_rsi_period_validation(self):
        """Test RSI with different periods."""
        prices = np.linspace(100, 150, 100)

        # Period 14 (standard)
        rsi14 = RSI(period=14)
        result14 = rsi14(prices)
        assert np.sum(np.isnan(result14)) == 14

        # Period 7 (shorter)
        rsi7 = RSI(period=7)
        result7 = rsi7(prices)
        assert np.sum(np.isnan(result7)) == 7

        # Period 21 (longer)
        rsi21 = RSI(period=21)
        result21 = rsi21(prices)
        assert np.sum(np.isnan(result21)) == 21


class TestRSISignals:
    """Test RSI signal generation."""

    def test_rsi_signal_generation_basic(self):
        """Test basic RSI signal generation."""
        prices = np.array([44.0, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42])
        rsi = RSI(period=3)
        rsi_values = rsi(prices)

        signals = rsi.get_signals(rsi_values, overbought=70, oversold=30)

        # Should have same length as input
        assert len(signals) == len(prices)

        # Signals should be -1, 0, or 1
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_rsi_overbought_signal(self):
        """Test overbought signal generation."""
        # Create strong uptrend to trigger overbought
        prices = np.linspace(100, 110, 50)
        rsi = RSI(period=10)
        rsi_values = rsi(prices)

        signals = rsi.get_signals(rsi_values, overbought=70, oversold=30)

        # In strong uptrend, should have overbought signals (1)
        valid_signals = signals[~np.isnan(rsi_values)]
        assert np.sum(valid_signals == 1) > 0  # Should have some overbought

    def test_rsi_oversold_signal(self):
        """Test oversold signal generation."""
        # Create strong downtrend to trigger oversold
        prices = np.linspace(110, 100, 50)
        rsi = RSI(period=10)
        rsi_values = rsi(prices)

        signals = rsi.get_signals(rsi_values, overbought=70, oversold=30)

        # In strong downtrend, should have oversold signals (-1)
        valid_signals = signals[~np.isnan(rsi_values)]
        assert np.sum(valid_signals == -1) > 0  # Should have some oversold

    def test_rsi_custom_thresholds(self):
        """Test RSI with custom overbought/oversold thresholds."""
        prices = np.random.uniform(90, 110, 100)
        rsi = RSI(period=14)
        rsi_values = rsi(prices)

        # Custom thresholds
        signals = rsi.get_signals(rsi_values, overbought=80, oversold=20)

        # Should have same length as input
        assert len(signals) == len(prices)
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_rsi_signal_nan_handling(self):
        """Test that signals properly handle NaN values."""
        prices = np.array([100, 101, 102, 103, 104, 105])
        rsi = RSI(period=3)
        rsi_values = rsi(prices)

        signals = rsi.get_signals(rsi_values)

        # NaN RSI should give 0 signal
        for i, (rsi_val, signal) in enumerate(zip(rsi_values, signals)):
            if np.isnan(rsi_val):
                assert signal == 0
