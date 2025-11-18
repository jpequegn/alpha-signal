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


class MACD(Indicator):
    """
    MACD (Moving Average Convergence Divergence) indicator.

    MACD is a trend-following momentum indicator that shows the relationship
    between two exponential moving averages. This indicator demonstrates
    the power of composition - it's built entirely from EMAs.

    Mathematical Definition:
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line

    Trading Signals:
        - MACD > Signal: Bullish (BUY signal)
        - MACD < Signal: Bearish (SELL signal)
        - Histogram changes sign: Momentum shift
        - Divergence: Price high but MACD low = reversal signal

    Key Properties:
        - Uses 12-period and 26-period EMAs (standard)
        - Signal line is 9-period EMA of MACD
        - Returns tuple: (macd_line, signal_line, histogram)
        - First 25 values are NaN (EMA(26) initialization)
        - Demonstrates indicator composition (reuses EMA class)

    Why Composition Matters:
        - No code duplication
        - Changes to EMA automatically propagate to MACD
        - Easier to test and maintain
        - Clear separation of concerns

    Example:
        >>> import numpy as np
        >>> prices = np.linspace(100, 110, 100)
        >>> macd = MACD()
        >>> macd_line, signal_line, histogram = macd(prices)
        >>> # All three arrays returned as tuple
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Initialize MACD with configurable periods.

        Args:
            fast: Period for fast EMA (default 12)
            slow: Period for slow EMA (default 26)
            signal: Period for signal line EMA (default 9)
        """
        # Use slow period as the "period" for Indicator base class
        super().__init__(period=slow)

        if not isinstance(fast, int) or fast < 1:
            raise ValueError(f"fast period must be positive integer, got {fast}")
        if not isinstance(slow, int) or slow < 1:
            raise ValueError(f"slow period must be positive integer, got {slow}")
        if not isinstance(signal, int) or signal < 1:
            raise ValueError(f"signal period must be positive integer, got {signal}")

        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate(self, data: np.ndarray) -> tuple:
        """
        Calculate MACD using composition of EMAs.

        Args:
            data: Array of prices, shape (N,)

        Returns:
            Tuple of three arrays (macd_line, signal_line, histogram)
            All arrays have shape (N,) with first (slow-1) values as NaN
        """
        # Validate input type and shape
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be numpy array, got {type(data)}")

        if data.ndim != 1:
            raise ValueError(f"data must be 1D array, got shape {data.shape}")

        # Create output arrays
        macd_line = self._create_output(len(data))
        signal_line = self._create_output(len(data))
        histogram = self._create_output(len(data))

        # Need at least slow+1 values
        if len(data) < self.slow + 1:
            return macd_line, signal_line, histogram

        # Step 1: Calculate MACD Line = EMA(12) - EMA(26)
        ema_fast = EMA(period=self.fast)
        ema_slow = EMA(period=self.slow)

        fast_values = ema_fast.calculate(data)
        slow_values = ema_slow.calculate(data)

        # MACD line = fast EMA - slow EMA
        macd_line = fast_values - slow_values

        # Step 2: Calculate Signal Line = EMA(9) of MACD Line
        # Only apply EMA to non-NaN MACD values
        ema_signal = EMA(period=self.signal)
        signal_line = ema_signal.calculate(macd_line)

        # Step 3: Calculate Histogram = MACD - Signal
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def get_signals(self, macd_line: np.ndarray, signal_line: np.ndarray) -> np.ndarray:
        """
        Generate trading signals from MACD crossovers.

        Args:
            macd_line: MACD line values from calculate()
            signal_line: Signal line values from calculate()

        Returns:
            Array of signals: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        signals = np.zeros(len(macd_line), dtype=int)

        for i, (macd, signal) in enumerate(zip(macd_line, signal_line)):
            if np.isnan(macd) or np.isnan(signal):
                signals[i] = 0
            elif macd > signal:
                signals[i] = 1  # Bullish
            elif macd < signal:
                signals[i] = -1  # Bearish
            else:
                signals[i] = 0  # Neutral

        return signals

    def __repr__(self) -> str:
        """String representation of MACD."""
        return f"MACD(fast={self.fast}, slow={self.slow}, signal={self.signal})"
