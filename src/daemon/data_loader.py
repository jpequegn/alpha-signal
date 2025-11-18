"""Historical data loader for daemon."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.indicators import SMA, EMA, RSI, MACD, BollingerBands


class HistoricalDataLoader:
    """Load and process historical OHLCV data."""

    def __init__(self):
        """Initialize data loader."""
        self.indicators_cache = {}

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load OHLCV data from CSV file.

        Args:
            csv_path: Path to CSV with columns: date, open, high, low, close, volume

        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df

    def calculate_indicators(
        self,
        closes: np.ndarray,
        symbol: str,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Calculate all indicators for a price series.

        Args:
            closes: Array of close prices
            symbol: Stock symbol
            timestamp: Current timestamp

        Returns:
            Dict with indicator values at current timestamp
        """
        if len(closes) < 26:
            # Not enough data for MACD (slow=26)
            return self._empty_indicators()

        # Calculate all indicators
        sma = SMA(period=20)
        ema12 = EMA(period=12)
        ema26 = EMA(period=26)
        rsi = RSI(period=14)
        macd = MACD(fast=12, slow=26, signal=9)
        bb = BollingerBands(period=20, num_std=2.0)

        sma_vals = sma(closes)
        ema12_vals = ema12(closes)
        ema26_vals = ema26(closes)
        rsi_vals = rsi(closes)
        macd_line, signal_line, histogram = macd(closes)
        bb_upper, bb_middle, bb_lower = bb(closes)

        # Get bandwidth percentage
        bb_bandwidth_pct = bb.get_bandwidth_percent(bb_upper, bb_middle, bb_lower)

        # Extract latest valid values
        return {
            'symbol': symbol,
            'timestamp': timestamp,
            'sma_20': float(sma_vals[-1]) if not np.isnan(sma_vals[-1]) else None,
            'ema_12': float(ema12_vals[-1]) if not np.isnan(ema12_vals[-1]) else None,
            'ema_26': float(ema26_vals[-1]) if not np.isnan(ema26_vals[-1]) else None,
            'rsi_14': float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else None,
            'macd_line': float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None,
            'macd_signal': float(signal_line[-1]) if not np.isnan(signal_line[-1]) else None,
            'macd_histogram': float(histogram[-1]) if not np.isnan(histogram[-1]) else None,
            'bb_upper': float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None,
            'bb_middle': float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None,
            'bb_lower': float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None,
            'bb_bandwidth_pct': float(bb_bandwidth_pct[-1]) if not np.isnan(bb_bandwidth_pct[-1]) else None,
        }

    def batch_calculate_indicators(
        self,
        closes: np.ndarray,
        symbol: str,
        start_date: datetime,
        frequency: str = 'D'
    ) -> List[Dict]:
        """Calculate indicators for multiple dates.

        Args:
            closes: Array of close prices
            symbol: Stock symbol
            start_date: Start date for calculations
            frequency: 'D' for daily, 'H' for hourly, etc.

        Returns:
            List of indicator dicts, one per date
        """
        results = []
        min_data = 26  # Minimum for MACD

        for i in range(min_data, len(closes)):
            # Calculate indicators using data up to index i
            window_closes = closes[:i + 1]
            timestamp = start_date + timedelta(days=i)

            indicators = self.calculate_indicators(
                closes=window_closes,
                symbol=symbol,
                timestamp=timestamp
            )
            results.append(indicators)

        return results

    def _empty_indicators(self) -> Dict[str, Optional[float]]:
        """Return empty indicator dict."""
        return {
            'symbol': None,
            'timestamp': None,
            'sma_20': None,
            'ema_12': None,
            'ema_26': None,
            'rsi_14': None,
            'macd_line': None,
            'macd_signal': None,
            'macd_histogram': None,
            'bb_upper': None,
            'bb_middle': None,
            'bb_lower': None,
            'bb_bandwidth_pct': None,
        }
