"""Historical data fetcher for backfill."""

from datetime import datetime
import pandas as pd
import yfinance as yf


class HistoricalDataFetcher:
    """Fetch historical market data from yfinance."""

    def __init__(self, symbol: str = "SPY"):
        """Initialize fetcher.

        Args:
            symbol: Stock symbol to fetch (default: SPY)
        """
        self.symbol = symbol

    def fetch_spy(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        df = yf.download(
            self.symbol,
            start=start_date,
            end=end_date,
            progress=False
        )

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure columns are lowercase
        df.columns = df.columns.str.lower()

        # Sort by date (oldest first)
        df = df.sort_index()

        return df

    def fetch_decade(self) -> pd.DataFrame:
        """Fetch full 2015-2024 SPY data.

        Returns:
            DataFrame with 10 years of data
        """
        return self.fetch_spy(
            start_date=datetime(2015, 1, 1),
            end_date=datetime(2024, 12, 31)
        )
