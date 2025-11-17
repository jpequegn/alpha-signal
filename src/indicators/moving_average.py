"""Moving average indicators (SMA, EMA)."""

import numpy as np
from .base import Indicator


class SMA(Indicator):
    """
    Simple Moving Average (SMA) indicator.

    The SMA is the average of closing prices over a fixed period.
    It smooths price data to identify trends.

    Mathematical Definition:
        SMA(t) = (P(t) + P(t-1) + ... + P(t-n+1)) / n

    where:
        - P(t) is the price at time t
        - n is the period (lookback window)

    Properties:
        - First (period-1) values are NaN (insufficient data)
        - Output has same length as input
        - Values are calculated using np.convolve() for efficiency
        - Fully vectorized (no loops)

    Example:
        >>> import numpy as np
        >>> prices = np.array([100, 101, 102, 103, 104, 105])
        >>> sma = SMA(period=3)
        >>> result = sma(prices)
        >>> result
        array([nan, nan, 101., 102., 103., 104.])
    """

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate SMA using efficient convolution.

        Args:
            data: Array of prices, shape (N,)

        Returns:
            Array of SMA values, shape (N,)
            First (period-1) values are NaN
        """
        # Validate input type and shape (but allow shorter lengths)
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(data)}")

        if data.ndim != 1:
            raise ValueError(f"data must be 1D array, got shape {data.shape}")

        # Create output array (will be filled with NaN initially)
        output = self._create_output(len(data))

        # If data is too short, return all NaN
        if len(data) < self.period:
            return output

        # Use convolution for efficient rolling average
        # Create kernel of size period with values 1/period
        kernel = np.ones(self.period) / self.period

        # convolve(data, kernel, mode='valid') gives us only fully-covered windows
        # This produces period-1 fewer values than input
        convolved = np.convolve(data, kernel, mode='valid')

        # Place convolved values starting at index (period-1)
        output[self.period - 1:] = convolved

        return output

    def __repr__(self) -> str:
        """String representation of SMA."""
        return f"SMA(period={self.period})"


class EMA(Indicator):
    """
    Exponential Moving Average (EMA) indicator.

    EMA is a weighted moving average that emphasizes recent prices more heavily
    than older prices. It responds faster to price changes than SMA.

    Mathematical Definition:
        EMA(t) = Price(t) × α + EMA(t-1) × (1 - α)

    where:
        - α (alpha) = 2 / (period + 1)
        - EMA is initialized with SMA at index (period-1)

    Properties:
        - First (period-1) values are NaN (initialization period)
        - Output has same length as input
        - Uses recursive calculation (each value depends on previous)
        - More responsive to recent price changes than SMA
        - Exponential decay of older prices

    Relationship to SMA:
        - EMA responds faster during trends
        - SMA is smoother but lags more
        - EMA good for short-term trading signals
        - Both useful for different strategies

    Example:
        >>> import numpy as np
        >>> prices = np.array([100, 101, 102, 103, 104, 105])
        >>> ema = EMA(period=3)
        >>> result = ema(prices)
        >>> ema.alpha
        0.5
    """

    @property
    def alpha(self) -> float:
        """
        Smoothing factor for EMA.

        Formula: α = 2 / (period + 1)

        Higher α → more weight on recent prices → more responsive
        Lower α → more weight on historical prices → smoother
        """
        return 2.0 / (self.period + 1)

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate EMA using recursive formula.

        Args:
            data: Array of prices, shape (N,)

        Returns:
            Array of EMA values, shape (N,)
            First (period-1) values are NaN
        """
        # Validate input type and shape
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(data)}")

        if data.ndim != 1:
            raise ValueError(f"data must be 1D array, got shape {data.shape}")

        # Create output array
        output = self._create_output(len(data))

        # If data is too short, return all NaN
        if len(data) < self.period:
            return output

        # Initialize EMA with SMA at index (period-1)
        sma = SMA(period=self.period)
        sma_values = sma.calculate(data)

        # EMA starts at index (period-1) with SMA value
        output[self.period - 1] = sma_values[self.period - 1]

        # Recursive calculation: EMA(t) = Price(t) × α + EMA(t-1) × (1-α)
        alpha = self.alpha
        for i in range(self.period, len(data)):
            output[i] = data[i] * alpha + output[i - 1] * (1 - alpha)

        return output

    def __repr__(self) -> str:
        """String representation of EMA."""
        return f"EMA(period={self.period}, alpha={self.alpha:.4f})"
