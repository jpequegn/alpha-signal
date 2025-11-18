"""Main daemon runner for signal generation."""

import logging
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

from src.daemon.data_loader import HistoricalDataLoader
from src.daemon.agent import run_signal_agent, AgentState
from src.daemon.models import Signal, IndicatorSnapshot, ReasoningStep
from src.daemon.db import init_db, get_db_session, create_tables

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DaemonRunner:
    """Main daemon runner for signal generation."""

    def __init__(
        self,
        symbol: str,
        db_url: str,
        start_date: datetime,
        end_date: datetime,
        batch_size: int = 100
    ):
        """Initialize daemon runner.

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            db_url: Database URL (e.g., postgresql://user:pass@localhost/db)
            start_date: Start date for signal generation
            end_date: End date for signal generation
            batch_size: Number of price points for indicator calculation
        """
        self.symbol = symbol
        self.db_url = db_url
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        self.data_loader = HistoricalDataLoader()

        # Initialize database
        init_db(db_url)
        create_tables()

    def run(self, price_data: np.ndarray, dates: list) -> int:
        """Run daemon on historical data.

        Args:
            price_data: Array of closing prices
            dates: List of corresponding dates

        Returns:
            Number of signals generated
        """
        logger.info(f"Starting daemon for {self.symbol}")
        logger.info(f"Processing {len(price_data)} price points")

        signal_count = 0

        # Process dates
        for i in range(self.batch_size, len(dates)):
            date = dates[i]

            # Skip if outside date range
            if date < self.start_date or date > self.end_date:
                continue

            try:
                signal = self.generate_signal_for_date(
                    prices=price_data[:i+1],
                    date=date,
                    current_price=float(price_data[i])
                )

                if signal:
                    self.save_signal(signal)
                    signal_count += 1
                    logger.info(f"{date}: {signal['signal']} (confidence: {signal['confidence']:.2f})")

            except Exception as e:
                logger.error(f"Error processing {date}: {e}", exc_info=True)
                continue

        logger.info(f"Daemon completed. Generated {signal_count} signals.")
        return signal_count

    def generate_signal_for_date(
        self,
        prices: np.ndarray,
        date: datetime,
        current_price: float
    ) -> Optional[dict]:
        """Generate signal for a specific date.

        Args:
            prices: Price array up to this date
            date: Current date
            current_price: Current close price

        Returns:
            Signal dict, or None if generation failed
        """
        # Calculate indicators
        indicators = self.data_loader.calculate_indicators(
            closes=prices,
            symbol=self.symbol,
            timestamp=date
        )

        # Check for invalid indicators
        if any(v is None for v in indicators.values() if v not in [self.symbol, date]):
            logger.warning(f"Insufficient data for {date}")
            return None

        # Build agent state
        state = AgentState(
            symbol=self.symbol,
            timestamp=date,
            current_price=current_price,
            closes=prices,
            indicator_state={
                'sma_20': indicators.get('sma_20'),
                'ema_12': indicators.get('ema_12'),
                'ema_26': indicators.get('ema_26'),
                'rsi_14': indicators.get('rsi_14'),
                'macd_line': indicators.get('macd_line'),
                'macd_signal': indicators.get('macd_signal'),
                'macd_histogram': indicators.get('macd_histogram'),
                'bb_upper': indicators.get('bb_upper'),
                'bb_middle': indicators.get('bb_middle'),
                'bb_lower': indicators.get('bb_lower'),
                'bb_bandwidth_pct': indicators.get('bb_bandwidth_pct'),
            },
            reasoning_steps=[],
            final_signal=None
        )

        # Run agent
        try:
            result = run_signal_agent(state)

            # Extract final_signal and add missing fields
            final_signal = result.get('final_signal')
            if final_signal:
                # Add indicator_state and reasoning_steps to the signal_data
                final_signal['indicator_state'] = result.get('indicator_state', {})
                final_signal['reasoning_steps'] = result.get('reasoning_steps', [])

            return final_signal
        except Exception as e:
            logger.error(f"Agent error for {date}: {e}", exc_info=True)
            return None

    def save_signal(self, signal_data: dict) -> None:
        """Save signal and reasoning to database.

        Args:
            signal_data: Signal dict with signal, confidence, reasoning, etc.
        """
        session = get_db_session()

        try:
            # Create signal
            signal = Signal(
                symbol=signal_data['symbol'],
                timestamp=signal_data['timestamp'],
                signal=signal_data['signal'],
                confidence=signal_data['confidence'],
                key_factors=signal_data.get('key_factors', []),
                contradictions=signal_data.get('contradictions', []),
                final_reasoning=signal_data.get('final_reasoning', '')
            )
            session.add(signal)
            session.flush()  # Get signal ID

            # Create indicator snapshot
            ind = signal_data.get('indicator_state', {})
            snapshot = IndicatorSnapshot(
                signal_id=signal.id,
                sma_20=ind.get('sma_20'),
                ema_12=ind.get('ema_12'),
                ema_26=ind.get('ema_26'),
                rsi_14=ind.get('rsi_14'),
                macd_line=ind.get('macd_line'),
                macd_signal=ind.get('macd_signal'),
                macd_histogram=ind.get('macd_histogram'),
                bb_upper=ind.get('bb_upper'),
                bb_middle=ind.get('bb_middle'),
                bb_lower=ind.get('bb_lower'),
                bb_bandwidth_pct=ind.get('bb_bandwidth_pct')
            )
            session.add(snapshot)

            # Create reasoning steps
            for step in signal_data.get('reasoning_steps', []):
                reasoning = ReasoningStep(
                    signal_id=signal.id,
                    step_order=step.get('step_order', 0),
                    indicator_group=step.get('indicator_group', 'UNKNOWN'),
                    analysis=step.get('analysis', '')
                )
                session.add(reasoning)

            session.commit()
            logger.debug(f"Saved signal {signal.id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save signal: {e}", exc_info=True)
            raise
        finally:
            session.close()


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Load sample data
    df = pd.read_csv("data/spy_historical.csv")  # You need to provide this
    prices = df['close'].values
    dates = pd.to_datetime(df['date']).tolist()

    # Run daemon
    runner = DaemonRunner(
        symbol="SPY",
        db_url="postgresql://user:pass@localhost/alpha_signal",
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 30)
    )

    runner.run(prices, dates)
