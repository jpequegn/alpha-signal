"""Volatility indicators (Bollinger Bands)."""

import numpy as np
from .base import Indicator
from .moving_average import SMA


class BollingerBands(Indicator):
    """
    Bollinger Bands - A volatility indicator that shows upper/lower price bands.

    Bollinger Bands use standard deviation to create dynamic bands around a moving average.
    Approximately 95% of prices fall within the bands, making them useful for volatility
    and mean reversion trading strategies.

    Mathematical Definition:
        Middle Band = SMA(period)
        StdDev = Standard deviation of prices over period
        Upper Band = Middle Band + (num_std × StdDev)
        Lower Band = Middle Band - (num_std × StdDev)

    where num_std is typically 2 (covers ~95% of normally distributed data)

    Key Properties:
        - Middle band: Simple moving average (trend)
        - Band width: Measures volatility
        - Prices outside bands: Potential reversal signals
        - Band squeeze: Low volatility, possible breakout coming
        - Band expansion: High volatility, potential trend
        - First (period-1) values are NaN (insufficient data)

    Standard Parameters:
        - Period: 20 (number of bars for SMA and StdDev)
        - NumStd: 2 (standard deviations, ~95% coverage)

    Calculation Steps:
        1. Calculate SMA(20) as middle band
        2. Calculate standard deviation over same 20 periods
        3. Upper band = Middle + (2 × StdDev)
        4. Lower band = Middle - (2 × StdDev)

    Trading Signals:
        - Price touches upper band: Overbought (potential sell)
        - Price touches lower band: Oversold (potential buy)
        - Narrow bands: Consolidation/low volatility (squeeze)
        - Wide bands: High volatility/strong trend

    Example:
        >>> import numpy as np
        >>> prices = np.random.normal(100, 2, 100)
        >>> bb = BollingerBands(period=20, num_std=2)
        >>> upper, middle, lower = bb(prices)
        >>> # upper, middle, lower are numpy arrays with band values
    """

    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        Initialize Bollinger Bands with configurable parameters.

        Args:
            period: Period for SMA and standard deviation (default 20)
            num_std: Number of standard deviations for band width (default 2.0)

        Raises:
            ValueError: If period < 1 or num_std <= 0
        """
        super().__init__(period=period)

        if not isinstance(num_std, (int, float)) or num_std <= 0:
            raise ValueError(f"num_std must be positive number, got {num_std}")

        self.num_std = num_std

    def calculate(self, data: np.ndarray) -> tuple:
        """
        Calculate Bollinger Bands.

        Args:
            data: Array of prices, shape (N,)

        Returns:
            Tuple of three arrays (upper_band, middle_band, lower_band)
            All arrays have shape (N,) with first (period-1) values as NaN
        """
        # Validate input type and shape
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(data)}")

        if data.ndim != 1:
            raise ValueError(f"data must be 1D array, got shape {data.shape}")

        # Create output arrays
        upper_band = self._create_output(len(data))
        middle_band = self._create_output(len(data))
        lower_band = self._create_output(len(data))

        # Need at least period values
        if len(data) < self.period:
            return upper_band, middle_band, lower_band

        # Step 1: Calculate middle band (SMA)
        sma = SMA(period=self.period)
        middle_band = sma.calculate(data)

        # Step 2: Calculate standard deviation using rolling window
        # We need to calculate std for each window
        std_devs = np.full(len(data), np.nan)

        for i in range(self.period - 1, len(data)):
            window = data[i - self.period + 1 : i + 1]
            std_devs[i] = np.std(window)

        # Step 3: Calculate upper and lower bands
        upper_band = middle_band + (self.num_std * std_devs)
        lower_band = middle_band - (self.num_std * std_devs)

        return upper_band, middle_band, lower_band

    def get_bandwidth(self, upper_band: np.ndarray, lower_band: np.ndarray) -> np.ndarray:
        """
        Calculate band width (absolute difference between bands).

        Band width measures the absolute volatility.

        Args:
            upper_band: Upper band values from calculate()
            lower_band: Lower band values from calculate()

        Returns:
            Array of band width values, shape (N,)
        """
        return upper_band - lower_band

    def get_bandwidth_percent(
        self, upper_band: np.ndarray, middle_band: np.ndarray, lower_band: np.ndarray
    ) -> np.ndarray:
        """
        Calculate bandwidth percentage (band width relative to middle band).

        Bandwidth % = (upper - lower) / middle × 100
        Measures relative volatility (independent of price level).

        Args:
            upper_band: Upper band values from calculate()
            middle_band: Middle band values from calculate()
            lower_band: Lower band values from calculate()

        Returns:
            Array of bandwidth percentage values, shape (N,)
            Values represent volatility as % of middle band
        """
        bandwidth = upper_band - lower_band
        bandwidth_pct = np.zeros(len(middle_band))

        # Avoid division by zero
        for i in range(len(middle_band)):
            if not np.isnan(middle_band[i]) and middle_band[i] != 0:
                bandwidth_pct[i] = (bandwidth[i] / middle_band[i]) * 100
            else:
                bandwidth_pct[i] = np.nan

        return bandwidth_pct

    def get_signals(
        self, prices: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray
    ) -> np.ndarray:
        """
        Generate trading signals from band touches.

        Args:
            prices: Original price array
            upper_band: Upper band values from calculate()
            lower_band: Lower band values from calculate()

        Returns:
            Array of signals: 1 (price at upper band), -1 (price at lower band), 0 (neutral)
            Signal generated when price touches or crosses bands
        """
        signals = np.zeros(len(prices), dtype=int)

        for i in range(len(prices)):
            # Skip if bands are not yet calculated
            if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
                signals[i] = 0
            # Check if price touches or exceeds upper band
            elif prices[i] >= upper_band[i]:
                signals[i] = 1  # Overbought
            # Check if price touches or goes below lower band
            elif prices[i] <= lower_band[i]:
                signals[i] = -1  # Oversold
            else:
                signals[i] = 0  # Neutral, within bands

        return signals

    def __repr__(self) -> str:
        """String representation of Bollinger Bands."""
        return f"BollingerBands(period={self.period}, num_std={self.num_std})"
