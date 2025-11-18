"""Tests for daemon runner."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.daemon.runner import DaemonRunner


class TestDaemonRunner:
    """Test daemon runner."""

    def test_runner_initialization(self):
        """Test DaemonRunner can be created."""
        runner = DaemonRunner(
            symbol="SPY",
            db_url="sqlite:///:memory:",  # Use in-memory SQLite for testing
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 6, 30)
        )
        assert runner is not None
        assert runner.symbol == "SPY"

    def test_runner_has_required_methods(self):
        """Test runner has execution methods."""
        runner = DaemonRunner(
            symbol="SPY",
            db_url="sqlite:///:memory:",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 6, 30)
        )

        assert hasattr(runner, 'run')
        assert hasattr(runner, 'generate_signal_for_date')
        assert hasattr(runner, 'save_signal')
