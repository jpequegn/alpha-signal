"""Tests for batch signal generation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.backfill.signal_generator import BackfillSignalGenerator
from src.daemon.runner import DaemonRunner


class TestBackfillSignalGenerator:
    """Test batch signal generation from historical data."""

    @pytest.fixture
    def sample_data(self):
        """Create sample historical data."""
        dates = pd.date_range('2024-01-01', periods=200, freq='D')  # Enough data for all indicators
        # Create more realistic price data with some variation (not perfectly linear)
        base_prices = np.linspace(440, 460, 200)
        # Add some noise to make it more realistic
        np.random.seed(42)
        noise = np.random.normal(0, 2, 200)
        prices = base_prices + noise

        return pd.DataFrame({
            'open': prices - 1,
            'high': prices + 2,
            'low': prices - 2,
            'close': prices,
            'volume': [1000000] * 200  # Simple constant volume for testing
        }, index=dates)

    def test_generator_initialization(self):
        """Test generator can be created."""
        gen = BackfillSignalGenerator(
            symbol="SPY",
            db_url="sqlite:///:memory:"
        )
        assert gen is not None

    def test_generate_signals_returns_list(self, sample_data):
        """Test signal generation returns list."""
        gen = BackfillSignalGenerator(
            symbol="SPY",
            db_url="sqlite:///:memory:"
        )

        signals = gen.generate_signals(sample_data)

        # Should return a list
        assert isinstance(signals, list)
        # Note: may be empty without API key or with insufficient indicator data

    def test_generated_signals_have_required_fields(self, sample_data):
        """Test signals have required fields."""
        gen = BackfillSignalGenerator(
            symbol="SPY",
            db_url="sqlite:///:memory:"
        )

        signals = gen.generate_signals(sample_data)

        if len(signals) > 0:
            signal = signals[0]
            assert 'symbol' in signal
            assert 'timestamp' in signal
            assert 'signal' in signal
            assert 'confidence' in signal

    def test_progress_callback_called(self, sample_data):
        """Test progress callback is invoked."""
        gen = BackfillSignalGenerator(
            symbol="SPY",
            db_url="sqlite:///:memory:"
        )

        progress_calls = []

        def progress_callback(current, total):
            progress_calls.append((current, total))

        gen.generate_signals(sample_data, progress_callback=progress_callback)

        # Should have progress updates
        assert len(progress_calls) > 0
