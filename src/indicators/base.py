"""Base class for all technical indicators."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.

    All indicators inherit from this class and must implement:
    1. __init__(self, ...) - Store parameters
    2. calculate(self, data) - Compute the indicator

    Example usage:
        sma = SMA(period=20)
        result = sma.calculate(prices)  # or sma(prices)
    """

    def __init__(self, period: int):
        """
        Initialize indicator.

        Args:
            period: Lookback period for calculation

        Raises:
            ValueError: If period < 1
        """
        if not isinstance(period, int) or period < 1:
            raise ValueError(f"period must be positive integer, got {period}")

        self.period = period
        self._last_result = None

    @abstractmethod
    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the indicator on input data.

        Args:
            data: Array of prices/values, shape (N,)

        Returns:
            Array of indicator values, shape (N,)
            First (period-1) values typically NaN

        Raises:
            ValueError: If data is invalid
            TypeError: If data type is wrong
        """
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Allow indicator to be called as function: indicator(data)"""
        return self.calculate(data)

    def _validate_input(self, data: np.ndarray, min_length: Optional[int] = None) -> None:
        """
        Validate input data before calculation.

        Args:
            data: Input array to validate
            min_length: Minimum required length (defaults to period)

        Raises:
            TypeError: If data is not numpy array
            ValueError: If data shape is invalid
            ValueError: If data has insufficient length
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(data)}")

        if data.ndim != 1:
            raise ValueError(f"data must be 1D array, got shape {data.shape}")

        min_len = min_length or self.period
        if len(data) < min_len:
            raise ValueError(
                f"data length {len(data)} < minimum required {min_len}"
            )

    def _create_output(self, length: int) -> np.ndarray:
        """Create output array initialized with NaN."""
        return np.full(length, np.nan, dtype=np.float64)
