"""Tests for historical data fetcher."""

import pytest
import pandas as pd
from datetime import datetime
from src.backfill.data_fetcher import HistoricalDataFetcher


class TestHistoricalDataFetcher:
    """Test historical data loading."""

    def test_fetcher_initialization(self):
        """Test fetcher can be initialized."""
        fetcher = HistoricalDataFetcher()
        assert fetcher is not None

    def test_fetch_spy_data(self):
        """Test fetching SPY data."""
        fetcher = HistoricalDataFetcher()
        df = fetcher.fetch_spy(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31)
        )

        assert df is not None
        assert len(df) > 0
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns

    def test_fetch_returns_sorted_chronological(self):
        """Test data is sorted oldest to newest."""
        fetcher = HistoricalDataFetcher()
        df = fetcher.fetch_spy(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )

        assert df.index[0] < df.index[-1]  # Chronological order

    def test_fetch_full_decade(self):
        """Test fetching full 2015-2024 dataset."""
        fetcher = HistoricalDataFetcher()
        df = fetcher.fetch_spy(
            start_date=datetime(2015, 1, 1),
            end_date=datetime(2024, 12, 31)
        )

        # Should have ~2500 trading days in 10 years
        assert len(df) > 2000
        assert len(df) < 3000
