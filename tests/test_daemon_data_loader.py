"""Tests for daemon data loader."""

import numpy as np
import pytest
from datetime import datetime, timedelta

from src.daemon.data_loader import HistoricalDataLoader


class TestHistoricalDataLoader:
    """Test historical data loading and indicator calculation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        base_price = 100.0
        for i in range(100):
            dates.append(datetime(2025, 1, 1) + timedelta(days=i))
            close = base_price + np.sin(i / 10) * 5
            opens.append(close - 0.5)
            highs.append(close + 1.0)
            lows.append(close - 1.0)
            closes.append(close)
            volumes.append(1000000 + i * 1000)

        return {
            "dates": dates,
            "opens": np.array(opens),
            "highs": np.array(highs),
            "lows": np.array(lows),
            "closes": np.array(closes),
            "volumes": np.array(volumes)
        }

    def test_loader_initialization(self):
        """Test loader can be initialized."""
        loader = HistoricalDataLoader()
        assert loader is not None

    def test_load_csv_creates_dataframe(self, sample_data, tmp_path):
        """Test loading CSV data."""
        # Create sample CSV
        import csv
        csv_path = tmp_path / "test_data.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
            for i in range(len(sample_data['dates'])):
                writer.writerow([
                    sample_data['dates'][i].isoformat(),
                    sample_data['opens'][i],
                    sample_data['highs'][i],
                    sample_data['lows'][i],
                    sample_data['closes'][i],
                    sample_data['volumes'][i]
                ])

        loader = HistoricalDataLoader()
        df = loader.load_csv(str(csv_path))

        assert df is not None
        assert len(df) == 100
        assert 'close' in df.columns

    def test_calculate_indicators(self, sample_data):
        """Test indicator calculation."""
        loader = HistoricalDataLoader()
        indicators = loader.calculate_indicators(
            closes=sample_data['closes'],
            symbol='SPY',
            timestamp=datetime(2025, 1, 1)
        )

        assert 'sma_20' in indicators
        assert 'ema_12' in indicators
        assert 'rsi_14' in indicators
        assert 'macd_line' in indicators
        assert 'bb_upper' in indicators

        # Check that values are reasonable
        assert isinstance(indicators['sma_20'], float)
        assert 95 < indicators['sma_20'] < 110  # Should be near price range

    def test_batch_calculate_indicators(self, sample_data):
        """Test batch indicator calculation across multiple dates."""
        loader = HistoricalDataLoader()
        indicators_list = loader.batch_calculate_indicators(
            closes=sample_data['closes'],
            symbol='SPY',
            start_date=datetime(2025, 1, 1),
            frequency='D'
        )

        # Should have indicators for each day after initialization period
        assert len(indicators_list) > 0
        assert all('sma_20' in ind for ind in indicators_list)
