"""Integration tests for full daemon pipeline."""

import pytest
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import logging

from src.daemon.runner import DaemonRunner
from src.daemon.db import get_db_session, init_db
from src.daemon.models import Signal, Base

logger = logging.getLogger(__name__)


@pytest.fixture
def test_db():
    """Create temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db_url = f"sqlite:///{db_path}"
        init_db(db_url)

        yield db_url

        # Cleanup
        session = get_db_session()
        Base.metadata.drop_all(bind=session.bind)
        session.close()


class TestDaemonIntegration:
    """Integration tests for daemon."""

    def test_daemon_generates_signals(self, test_db):
        """Test daemon generates signals from price data."""
        # Create sample price data (100 days)
        prices = np.linspace(440, 460, 100)  # Rising trend
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(100)]

        runner = DaemonRunner(
            symbol="SPY",
            db_url=test_db,
            start_date=datetime(2025, 1, 30),  # After warmup period
            end_date=datetime(2025, 4, 10)
        )

        # Run daemon
        signal_count = runner.run(prices, dates)

        # Verify signals were generated
        assert signal_count > 0
        logger.info(f"Generated {signal_count} signals")

    def test_signals_have_complete_reasoning(self, test_db):
        """Test generated signals have complete reasoning chain."""
        prices = np.linspace(440, 460, 100)
        dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(100)]

        runner = DaemonRunner(
            symbol="SPY",
            db_url=test_db,
            start_date=datetime(2025, 1, 30),
            end_date=datetime(2025, 4, 10)
        )

        signal_count = runner.run(prices, dates)

        if signal_count > 0:
            session = get_db_session()
            signal = session.query(Signal).first()

            assert signal is not None
            assert signal.signal in ['BUY', 'SELL', 'HOLD']
            assert signal.confidence > 0
            assert signal.final_reasoning is not None
            session.close()
