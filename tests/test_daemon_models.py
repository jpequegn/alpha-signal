"""Tests for daemon database models."""

import pytest
from datetime import datetime

from src.daemon.models import Signal, IndicatorSnapshot, ReasoningStep
from src.daemon.db import Base, get_db_session


class TestSignalModel:
    """Test Signal model."""

    def test_signal_creation(self):
        """Test Signal instance creation."""
        signal = Signal(
            symbol='SPY',
            timestamp=datetime(2025, 1, 15, 10, 30),
            signal='BUY',
            confidence=0.78,
            key_factors=['SMA uptrend', 'RSI bullish'],
            contradictions=[],
            final_reasoning='Multiple indicators align on bullish thesis.'
        )

        assert signal.symbol == 'SPY'
        assert signal.signal == 'BUY'
        assert signal.confidence == 0.78
        assert len(signal.key_factors) == 2

    def test_indicator_snapshot_creation(self):
        """Test IndicatorSnapshot creation."""
        snapshot = IndicatorSnapshot(
            signal_id=1,
            sma_20=450.5,
            ema_12=451.2,
            rsi_14=65.0,
            macd_line=2.3,
            macd_signal=1.8,
            macd_histogram=0.5,
            bb_upper=455.0,
            bb_middle=450.0,
            bb_lower=445.0,
            bb_bandwidth_pct=2.2
        )

        assert snapshot.signal_id == 1
        assert snapshot.sma_20 == 450.5
        assert snapshot.rsi_14 == 65.0

    def test_reasoning_step_creation(self):
        """Test ReasoningStep creation."""
        step = ReasoningStep(
            signal_id=1,
            step_order=1,
            indicator_group='TREND',
            analysis='SMA shows clear uptrend with price above 20-day average.'
        )

        assert step.signal_id == 1
        assert step.step_order == 1
        assert step.indicator_group == 'TREND'
        assert 'uptrend' in step.analysis
