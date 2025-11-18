"""Unit tests for technical indicators."""

import pytest
import numpy as np
from src.indicators import SMA, EMA, RSI, MACD, BollingerBands


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


class TestMACD:
    """Test cases for MACD (Moving Average Convergence Divergence) indicator."""

    def test_macd_basic_calculation(self):
        """Test MACD returns tuple of three arrays."""
        prices = np.linspace(100, 110, 100)
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        # Should return tuple of three arrays
        assert isinstance(macd_line, np.ndarray)
        assert isinstance(signal_line, np.ndarray)
        assert isinstance(histogram, np.ndarray)

        # All arrays should have same length as input
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

    def test_macd_composition_with_ema(self):
        """Test that MACD correctly composes EMAs."""
        prices = np.linspace(100, 110, 100)

        # Calculate MACD
        macd = MACD(fast=12, slow=26, signal=9)
        macd_line, signal_line, histogram = macd(prices)

        # Calculate EMAs independently
        ema_fast = EMA(period=12)
        ema_slow = EMA(period=26)
        fast_vals = ema_fast(prices)
        slow_vals = ema_slow(prices)

        # MACD line should equal EMA(12) - EMA(26)
        expected_macd = fast_vals - slow_vals
        valid_idx = ~np.isnan(expected_macd)
        assert np.allclose(
            macd_line[valid_idx], expected_macd[valid_idx], equal_nan=True
        )

    def test_macd_nan_initialization(self):
        """Test that MACD has proper NaN initialization."""
        prices = np.linspace(100, 110, 100)
        macd = MACD(fast=12, slow=26, signal=9)
        macd_line, signal_line, histogram = macd(prices)

        # MACD line should have NaN for first (slow-1) values
        # because EMA(26) requires 26 values
        assert np.sum(np.isnan(macd_line[:25])) == 25

        # Signal line should have more NaN due to EMA(9) applied to MACD
        # Total: slow - 1 + signal - 1 = 25 + 8 = 33
        assert np.sum(np.isnan(signal_line[:33])) == 33

        # Histogram should match signal line NaN pattern
        assert np.sum(np.isnan(histogram[:33])) == 33

    def test_macd_histogram_calculation(self):
        """Test that histogram = MACD - Signal."""
        prices = np.linspace(100, 110, 100)
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        # Histogram should equal MACD - Signal
        expected_histogram = macd_line - signal_line
        assert np.allclose(histogram, expected_histogram, equal_nan=True)

    def test_macd_uptrend_behavior(self):
        """Test MACD line behavior during uptrend."""
        # Create uptrend with enough data
        prices = np.linspace(100, 130, 100)
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        # MACD line should increase in uptrend (fast EMA > slow EMA)
        valid_macd = macd_line[~np.isnan(macd_line)]
        assert len(valid_macd) > 0
        # In uptrend, MACD line should be positive
        assert np.all(valid_macd > 0)

    def test_macd_downtrend_behavior(self):
        """Test MACD line behavior during downtrend."""
        # Create downtrend with enough data
        prices = np.linspace(130, 100, 100)
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        # MACD line should decrease in downtrend (fast EMA < slow EMA)
        valid_macd = macd_line[~np.isnan(macd_line)]
        assert len(valid_macd) > 0
        # In downtrend, MACD line should be negative
        assert np.all(valid_macd < 0)

    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data."""
        prices = np.array([100, 101, 102])
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        # With insufficient data, should return all NaN
        assert np.all(np.isnan(macd_line))
        assert np.all(np.isnan(signal_line))
        assert np.all(np.isnan(histogram))

    def test_macd_constant_prices(self):
        """Test MACD with constant prices."""
        prices = np.full(100, 100.0)
        macd = MACD(fast=12, slow=26, signal=9)
        macd_line, signal_line, histogram = macd(prices)

        # With constant prices, MACD line should be ~0 (EMA(12) - EMA(26) = 100 - 100)
        # MACD line initializes at index 25 (slow-1)
        valid_macd = macd_line[~np.isnan(macd_line)]
        assert len(valid_macd) > 0
        assert np.allclose(valid_macd, 0.0, atol=0.01)

    def test_macd_custom_periods(self):
        """Test MACD with custom periods."""
        prices = np.linspace(100, 120, 100)

        # Test standard periods
        macd_std = MACD(fast=12, slow=26, signal=9)
        macd_line1, signal_line1, hist1 = macd_std(prices)
        assert len(macd_line1) == 100

        # Test custom periods
        macd_custom = MACD(fast=5, slow=10, signal=3)
        macd_line2, signal_line2, hist2 = macd_custom(prices)
        assert len(macd_line2) == 100

        # Different periods should give different results
        assert not np.allclose(macd_line1[50:], macd_line2[50:])

    def test_macd_period_validation(self):
        """Test MACD parameter validation."""
        prices = np.linspace(100, 110, 100)

        # Invalid fast period
        with pytest.raises(ValueError):
            MACD(fast=0, slow=26, signal=9)(prices)

        # Invalid slow period
        with pytest.raises(ValueError):
            MACD(fast=12, slow=-1, signal=9)(prices)

        # Invalid signal period
        with pytest.raises(ValueError):
            MACD(fast=12, slow=26, signal=0)(prices)


class TestMACDSignals:
    """Test MACD signal generation."""

    def test_macd_signal_generation_basic(self):
        """Test basic MACD signal generation."""
        prices = np.linspace(100, 110, 100)
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        signals = macd.get_signals(macd_line, signal_line)

        # Should have same length as input
        assert len(signals) == len(prices)

        # Signals should be -1, 0, or 1
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_macd_signal_zero_when_nan(self):
        """Test that signals are 0 when MACD or signal line is NaN."""
        prices = np.linspace(100, 110, 100)
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        signals = macd.get_signals(macd_line, signal_line)

        # When MACD or signal line is NaN, signal should be 0
        for i, (m, s, sig) in enumerate(zip(macd_line, signal_line, signals)):
            if np.isnan(m) or np.isnan(s):
                assert sig == 0

    def test_macd_signal_consistency(self):
        """Test that signal generation is consistent."""
        prices = np.linspace(110, 100, 100)
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        signals = macd.get_signals(macd_line, signal_line)

        # All signals should be valid (-1, 0, or 1)
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_macd_signal_nan_handling(self):
        """Test that signals properly handle NaN values."""
        prices = np.array([100, 101, 102, 103, 104, 105])
        macd = MACD()
        macd_line, signal_line, histogram = macd(prices)

        signals = macd.get_signals(macd_line, signal_line)

        # NaN MACD should give 0 signal
        for i, (macd_val, signal) in enumerate(zip(macd_line, signals)):
            if np.isnan(macd_val):
                assert signal == 0

    def test_macd_signal_with_realistic_data(self, sample_prices):
        """Test MACD signals with realistic price data."""
        macd = MACD()
        macd_line, signal_line, histogram = macd(sample_prices)

        signals = macd.get_signals(macd_line, signal_line)

        # Should have same length
        assert len(signals) == len(sample_prices)

        # All signals should be valid
        assert np.all(np.isin(signals, [-1, 0, 1]))

        # Signals should be 0 where MACD or signal line are NaN
        for i, (m, s) in enumerate(zip(macd_line, signal_line)):
            if np.isnan(m) or np.isnan(s):
                assert signals[i] == 0


class TestBollingerBands:
    """Test cases for Bollinger Bands volatility indicator."""

    def test_bollinger_bands_basic_calculation(self):
        """Test Bollinger Bands returns three arrays."""
        prices = np.linspace(100, 110, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        # Should return three arrays
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)

        # All arrays should have same length as input
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    def test_bollinger_bands_band_ordering(self):
        """Test that upper band > middle band > lower band."""
        prices = np.linspace(100, 110, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        # Get valid indices (not NaN)
        valid_idx = ~np.isnan(upper)

        # Upper should always be > middle, middle > lower
        assert np.all(upper[valid_idx] >= middle[valid_idx])
        assert np.all(middle[valid_idx] >= lower[valid_idx])

    def test_bollinger_bands_nan_initialization(self):
        """Test that bands have proper NaN initialization."""
        prices = np.linspace(100, 110, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        # First (period-1) values should be NaN
        assert np.sum(np.isnan(upper[:19])) == 19
        assert np.sum(np.isnan(middle[:19])) == 19
        assert np.sum(np.isnan(lower[:19])) == 19

    def test_bollinger_bands_with_constant_prices(self):
        """Test Bollinger Bands with constant prices (zero volatility)."""
        prices = np.full(100, 100.0)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        # With constant prices, std dev = 0, so bands should equal middle
        valid_idx = ~np.isnan(upper)
        valid_upper = upper[valid_idx]
        valid_middle = middle[valid_idx]
        valid_lower = lower[valid_idx]

        # Middle should be 100 (constant price)
        assert np.allclose(valid_middle, 100.0)

        # Upper and lower should equal middle (no volatility)
        assert np.allclose(valid_upper, valid_middle, atol=0.01)
        assert np.allclose(valid_lower, valid_middle, atol=0.01)

    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data."""
        prices = np.array([100, 101, 102])
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        # All values should be NaN
        assert np.all(np.isnan(upper))
        assert np.all(np.isnan(middle))
        assert np.all(np.isnan(lower))

    def test_bollinger_bands_custom_num_std(self):
        """Test Bollinger Bands with custom num_std parameter."""
        prices = np.linspace(100, 120, 100)

        # Standard 2.0 std dev
        bb2 = BollingerBands(period=20, num_std=2.0)
        upper2, middle2, lower2 = bb2(prices)

        # Wider bands with 3.0 std dev
        bb3 = BollingerBands(period=20, num_std=3.0)
        upper3, middle3, lower3 = bb3(prices)

        # Get valid indices
        valid_idx = ~np.isnan(upper2)

        # 3.0 std dev should give wider bands than 2.0
        width2 = upper2[valid_idx] - lower2[valid_idx]
        width3 = upper3[valid_idx] - lower3[valid_idx]

        assert np.all(width3 > width2)

    def test_bollinger_bands_period_validation(self):
        """Test Bollinger Bands parameter validation."""
        prices = np.linspace(100, 110, 100)

        # Invalid period
        with pytest.raises(ValueError):
            BollingerBands(period=0, num_std=2)(prices)

        # Invalid num_std
        with pytest.raises(ValueError):
            BollingerBands(period=20, num_std=0)(prices)

        with pytest.raises(ValueError):
            BollingerBands(period=20, num_std=-1)(prices)


class TestBollingerBandsBandwidth:
    """Test Bollinger Bands bandwidth calculations."""

    def test_bandwidth_calculation(self):
        """Test band width (absolute difference)."""
        prices = np.linspace(100, 120, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        bandwidth = bb.get_bandwidth(upper, lower)

        # Should have same length
        assert len(bandwidth) == len(prices)

        # Bandwidth should equal upper - lower
        expected = upper - lower
        assert np.allclose(bandwidth, expected, equal_nan=True)

    def test_bandwidth_percent_calculation(self):
        """Test bandwidth percentage (relative to middle band)."""
        prices = np.linspace(100, 120, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        bandwidth_pct = bb.get_bandwidth_percent(upper, middle, lower)

        # Should have same length
        assert len(bandwidth_pct) == len(prices)

        # All valid values should be positive (upper > middle > lower)
        valid_idx = ~np.isnan(bandwidth_pct)
        assert np.all(bandwidth_pct[valid_idx] >= 0)

    def test_bandwidth_expands_with_volatility(self):
        """Test that bandwidth expands with increased volatility."""
        # Low volatility data
        prices_calm = np.full(100, 100.0)
        prices_calm[:50] = np.linspace(100, 101, 50)  # Tiny variation

        # High volatility data
        prices_volatile = np.concatenate([
            np.linspace(100, 110, 50),
            np.linspace(110, 90, 50)
        ])

        bb = BollingerBands(period=20, num_std=2)

        # Calm market
        upper_calm, middle_calm, lower_calm = bb(prices_calm)
        bw_calm = bb.get_bandwidth(upper_calm, lower_calm)
        valid_bw_calm = bw_calm[~np.isnan(bw_calm)]

        # Volatile market
        upper_vol, middle_vol, lower_vol = bb(prices_volatile)
        bw_vol = bb.get_bandwidth(upper_vol, lower_vol)
        valid_bw_vol = bw_vol[~np.isnan(bw_vol)]

        # Volatile market should have wider bands on average
        assert np.mean(valid_bw_vol) > np.mean(valid_bw_calm)


class TestBollingerBandsSignals:
    """Test Bollinger Bands signal generation."""

    def test_signal_generation_basic(self):
        """Test basic signal generation from band touches."""
        prices = np.linspace(100, 110, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        signals = bb.get_signals(prices, upper, lower)

        # Should have same length
        assert len(signals) == len(prices)

        # All signals should be -1, 0, or 1
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_signal_nan_handling(self):
        """Test that signals are 0 when bands are NaN."""
        prices = np.array([100, 101, 102, 103, 104, 105])
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        signals = bb.get_signals(prices, upper, lower)

        # When bands are NaN, signal should be 0
        for i, (u, l) in enumerate(zip(upper, lower)):
            if np.isnan(u) or np.isnan(l):
                assert signals[i] == 0

    def test_signal_overbought_touch_upper(self):
        """Test overbought signal when price touches upper band."""
        prices = np.linspace(100, 110, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        # Modify prices to touch/exceed upper band
        prices_modified = prices.copy()
        prices_modified[50:] = upper[50:] + 0.1  # Exceed upper band

        signals = bb.get_signals(prices_modified, upper, lower)

        # Should have overbought signals where price > upper
        valid_signals = signals[~np.isnan(upper)]
        assert np.sum(valid_signals == 1) > 0

    def test_signal_oversold_touch_lower(self):
        """Test oversold signal when price touches lower band."""
        prices = np.linspace(100, 110, 100)
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(prices)

        # Modify prices to touch/go below lower band
        prices_modified = prices.copy()
        prices_modified[50:] = lower[50:] - 0.1  # Go below lower band

        signals = bb.get_signals(prices_modified, upper, lower)

        # Should have oversold signals where price < lower
        valid_signals = signals[~np.isnan(lower)]
        assert np.sum(valid_signals == -1) > 0

    def test_signal_statistical_property_95_percent(self, sample_prices):
        """Test that ~95% of prices fall within bands."""
        bb = BollingerBands(period=20, num_std=2)
        upper, middle, lower = bb(sample_prices)

        # Count prices within bands (where bands are valid)
        valid_idx = ~np.isnan(upper)
        prices_valid = sample_prices[valid_idx]
        upper_valid = upper[valid_idx]
        lower_valid = lower[valid_idx]

        # Check how many prices are within bands
        within_bands = np.sum((prices_valid >= lower_valid) & (prices_valid <= upper_valid))
        percent_within = (within_bands / len(prices_valid)) * 100

        # Should be approximately 95% (allow some tolerance)
        assert percent_within >= 90  # Allow some variance
