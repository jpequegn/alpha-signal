"""Batch signal generation for historical backfill."""

import logging
from datetime import datetime
from typing import Callable, List, Optional
import pandas as pd
import numpy as np

from src.daemon.runner import DaemonRunner
from src.daemon.db import init_db, create_tables, get_db_session
from src.daemon.models import Signal, IndicatorSnapshot, ReasoningStep

logger = logging.getLogger(__name__)


class BackfillSignalGenerator:
    """Generate signals for historical date range."""

    def __init__(
        self,
        symbol: str,
        db_url: str,
        batch_size: int = 100
    ):
        """Initialize signal generator.

        Args:
            symbol: Stock symbol
            db_url: Database URL
            batch_size: Minimum data points for indicator calculation
        """
        self.symbol = symbol
        self.db_url = db_url
        self.batch_size = batch_size

        # Initialize database
        init_db(db_url)
        create_tables()

    def generate_signals(
        self,
        data: pd.DataFrame,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[dict]:
        """Generate signals for historical data.

        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
            progress_callback: Function(current, total) for progress updates

        Returns:
            List of signal dicts
        """
        logger.info(f"Starting backfill for {self.symbol}")
        logger.info(f"Processing {len(data)} dates")

        signals = []
        prices = data['close'].values
        dates = data.index.to_list()

        # Check if we have enough data
        if len(dates) <= self.batch_size:
            logger.warning(f"Not enough data for signal generation: {len(dates)} <= {self.batch_size}")
            return signals

        # Create daemon runner
        runner = DaemonRunner(
            symbol=self.symbol,
            db_url=self.db_url,
            start_date=dates[self.batch_size],
            end_date=dates[-1]
        )

        # Generate signals for each date
        for i in range(self.batch_size, len(dates)):
            date = dates[i]
            current_price = float(prices[i])

            if progress_callback:
                progress_callback(i - self.batch_size, len(dates) - self.batch_size)

            try:
                signal = runner.generate_signal_for_date(
                    prices=prices[:i+1],
                    date=pd.Timestamp(date),
                    current_price=current_price
                )

                if signal:
                    signals.append(signal)
                    logger.info(f"{date.date()}: {signal['signal']} (confidence: {signal['confidence']:.2f})")

            except Exception as e:
                logger.error(f"Error generating signal for {date}: {e}")
                continue

        logger.info(f"Backfill complete. Generated {len(signals)} signals.")
        return signals
