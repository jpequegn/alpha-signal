"""Momentum indicators (RSI, MACD)."""

import numpy as np
from .base import Indicator
from .moving_average import SMA, EMA


class RSI(Indicator):
    """
    Relative Strength Index (RSI) momentum indicator.

    RSI measures momentum on a 0-100 scale, identifying overbought and oversold
    conditions. It's one of the most widely used momentum indicators.

    Mathematical Definition:
        RSI = 100 - (100 / (1 + RS))
        where RS = AvgGain / AvgLoss

    Key Properties:
        - Values range from 0 to 100
        - RSI > 70: Overbought (potential sell signal)
        - RSI < 30: Oversold (potential buy signal)
        - RSI = 50: Neutral, no strong momentum
        - Uses Wilder's smoothing (not EMA)
        - First (period) values are NaN (initialization)

    Calculation Steps:
        1. Calculate price changes: change = P(t) - P(t-1)
        2. Separate gains and losses
        3. Initialize with simple average
        4. Apply Wilder's smoothing: (prev × (n-1) + current) / n
        5. Calculate RS = AvgGain / AvgLoss
        6. Calculate RSI = 100 - (100 / (1 + RS))

    Common Uses:
        - Identify trend reversals (extreme RSI)
        - Confirm trends (RSI in 40-60 range for sideways)
        - Divergence trading (price makes new high but RSI doesn't)

    Example:
        >>> import numpy as np
        >>> prices = np.array([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42])
        >>> rsi = RSI(period=3)
        >>> result = rsi(prices)
        # Returns RSI values (0-100 scale)
    """

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate RSI using Wilder's smoothing.

        Args:
            data: Array of prices, shape (N,)

        Returns:
            Array of RSI values (0-100), shape (N,)
            First (period) values are NaN
        """
        # Validate input type and shape
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(data)}")

        if data.ndim != 1:
            raise ValueError(f"data must be 1D array, got shape {data.shape}")

        # Create output array
        output = self._create_output(len(data))

        # Need at least period+1 values (period changes to calculate)
        if len(data) < self.period + 1:
            return output

        # Step 1: Calculate price changes
        changes = np.diff(data)

        # Step 2: Separate gains and losses
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        # Step 3: Initialize with simple averages
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])

        # Create arrays to store smoothed values
        smoothed_gains = np.zeros(len(gains))
        smoothed_losses = np.zeros(len(losses))

        smoothed_gains[self.period - 1] = avg_gain
        smoothed_losses[self.period - 1] = avg_loss

        # Step 4: Apply Wilder's smoothing
        # Smoothed = (prev_smoothed × (n-1) + current) / n
        for i in range(self.period, len(gains)):
            avg_gain = (smoothed_gains[i - 1] * (self.period - 1) + gains[i]) / self.period
            avg_loss = (smoothed_losses[i - 1] * (self.period - 1) + losses[i]) / self.period

            smoothed_gains[i] = avg_gain
            smoothed_losses[i] = avg_loss

        # Step 5 & 6: Calculate RSI
        # RSI = 100 - (100 / (1 + RS)) where RS = AvgGain / AvgLoss
        # Output starts at index (period) since we need period changes
        for i in range(self.period, len(data)):
            gain_idx = i - 1  # gains array is len(data)-1
            avg_gain = smoothed_gains[gain_idx]
            avg_loss = smoothed_losses[gain_idx]

            # Handle edge cases
            if avg_gain == 0 and avg_loss == 0:
                # No gains or losses (constant prices)
                rsi_value = 50.0
            elif avg_loss == 0:
                # Only gains, no losses (pure uptrend)
                rsi_value = 100.0
            elif avg_gain == 0:
                # Only losses, no gains (pure downtrend)
                rsi_value = 0.0
            else:
                # Normal case
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))

            output[i] = np.clip(rsi_value, 0, 100)  # Ensure 0-100 bounds

        return output

    def get_signals(self, rsi_values: np.ndarray,
                   overbought: float = 70, oversold: float = 30) -> np.ndarray:
        """
        Generate trading signals from RSI values.

        Args:
            rsi_values: Array of RSI values from calculate()
            overbought: Threshold for overbought (default 70)
            oversold: Threshold for oversold (default 30)

        Returns:
            Array of signals: 1 (overbought), -1 (oversold), 0 (neutral)
        """
        signals = np.zeros(len(rsi_values), dtype=int)

        for i, rsi in enumerate(rsi_values):
            if np.isnan(rsi):
                signals[i] = 0
            elif rsi > overbought:
                signals[i] = 1  # Overbought
            elif rsi < oversold:
                signals[i] = -1  # Oversold
            else:
                signals[i] = 0  # Neutral

        return signals

    def __repr__(self) -> str:
        """String representation of RSI."""
        return f"RSI(period={self.period})"
