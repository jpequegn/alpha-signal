# Task 1.5: Bollinger Bands Indicator

**Difficulty**: ⭐⭐ (Intermediate)
**Estimated Time**: 3 hours
**GitHub Issue**: #5
**Prerequisite**: Task 1.1 (SMA)

## Overview

Implement Bollinger Bands - a volatility indicator showing upper/lower price bands around a moving average.

## What You'll Learn

- Standard deviation and volatility
- Mean reversion concept
- Band-based trading signals
- How volatility changes with market conditions

## Mathematical Definition

```
Middle Band = SMA(period) [typically 20]
StdDev = Standard deviation of prices over period
Upper Band = Middle + (StdDev × 2)
Lower Band = Middle - (StdDev × 2)

Interpretation:
- Price near Upper Band: Potential overbought (mean reversion)
- Price near Lower Band: Potential oversold (mean reversion)
- Band width: Volatility (wide = high volatility, narrow = low volatility)
- 95% of prices should fall within bands (statistical property)
```

## Key Concept: Mean Reversion

When price touches the upper band, it's statistically extreme and tends to revert toward the mean (middle band). Same for lower band.

This is different from trend-following indicators (SMA, EMA, MACD).

## Implementation

### 1. Implement Bollinger Bands Class

**File**: `src/indicators/volatility.py` (create new file)

```python
import numpy as np
from typing import Tuple
from .base import Indicator
from .moving_average import SMA

class BollingerBands(Indicator):
    """Bollinger Bands volatility indicator."""

    def __init__(self, period: int = 20, num_std_dev: float = 2.0):
        """
        Initialize Bollinger Bands.

        Args:
            period: SMA period (typically 20)
            num_std_dev: Number of standard deviations (typically 2)
        """
        self.period = period
        self.num_std_dev = num_std_dev

    def calculate(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price data

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Middle band = SMA
        sma = SMA(self.period)(data)

        # Calculate rolling standard deviation
        std_dev = np.full_like(data, np.nan, dtype=float)

        for i in range(self.period - 1, len(data)):
            window = data[i - self.period + 1 : i + 1]
            std_dev[i] = np.std(window)  # ddof=0 for population std dev

        # Bands
        upper_band = sma + (std_dev * self.num_std_dev)
        lower_band = sma - (std_dev * self.num_std_dev)

        return upper_band, sma, lower_band

    def get_band_width(
        self,
        upper_band: np.ndarray,
        lower_band: np.ndarray
    ) -> np.ndarray:
        """
        Calculate band width (measure of volatility).

        Wider bands = higher volatility
        Narrower bands = lower volatility

        Args:
            upper_band: Upper Bollinger Band
            lower_band: Lower Bollinger Band

        Returns:
            Array of band widths
        """
        return upper_band - lower_band

    def get_signals(
        self,
        prices: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray
    ) -> np.ndarray:
        """
        Generate mean-reversion signals based on band touches.

        Args:
            prices: Price data
            upper_band: Upper band
            lower_band: Lower band

        Returns:
            Signals: 1 (buy at oversold), -1 (sell at overbought), 0 (hold)
        """
        signals = np.zeros_like(prices)

        for i in range(1, len(prices)):
            if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
                continue

            # Oversold: price touches lower band → BUY (mean reversion)
            if prices[i-1] > lower_band[i-1] and prices[i] <= lower_band[i]:
                signals[i] = 1

            # Overbought: price touches upper band → SELL (mean reversion)
            elif prices[i-1] < upper_band[i-1] and prices[i] >= upper_band[i]:
                signals[i] = -1

        return signals

    def get_bandwidth_percent(
        self,
        prices: np.ndarray,
        upper_band: np.ndarray,
        middle_band: np.ndarray,
        lower_band: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Bollinger Bandwidth % (volatility indicator).

        When bandwidth contracts significantly, breakout often follows.

        Args:
            prices: Price data
            upper_band: Upper band
            middle_band: Middle band (SMA)
            lower_band: Lower band

        Returns:
            Bandwidth as % of middle band (0 = at middle, 100 = at band)
        """
        band_width = upper_band - lower_band

        # Position of price within bands (0-100%)
        position = (prices - lower_band) / band_width * 100
        position = np.clip(position, 0, 100)  # Constrain to 0-100

        return position
```

### 2. Unit Tests

```python
class TestBollingerBands:

    def test_bollinger_bands_basic(self):
        """Test Bollinger Bands calculation."""
        data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108],
                        dtype=float)
        bb = BollingerBands(period=2, num_std_dev=1.0)
        upper, middle, lower = bb(data)

        # Middle should be SMA
        sma = SMA(2)(data)
        assert np.allclose(middle[~np.isnan(middle)],
                          sma[~np.isnan(middle)])

        # Upper > Middle > Lower
        valid = ~np.isnan(upper)
        assert np.all(upper[valid] >= middle[valid])
        assert np.all(middle[valid] >= lower[valid])

    def test_bollinger_bands_bounds(self):
        """Test bands correctly bound prices."""
        data = np.random.uniform(90, 110, size=100)
        bb = BollingerBands(period=20, num_std_dev=2.0)
        upper, middle, lower = bb(data)

        # ~95% of prices should be within bands (statistical)
        within_bands = (data >= lower) & (data <= upper)
        pct_within = np.sum(within_bands[~np.isnan(upper)]) / np.sum(~np.isnan(upper))
        assert pct_within > 0.85  # Allow some deviation

    def test_bollinger_bands_volatility_extreme(self):
        """Test bands expand during high volatility."""
        # Low volatility
        data_low_vol = np.array([100.0] * 20 + [100.5, 99.5] * 5, dtype=float)

        # High volatility
        data_high_vol = np.concatenate([
            np.arange(100, 110, 0.5),
            np.arange(110, 90, -0.5)
        ])

        bb = BollingerBands(period=20, num_std_dev=2.0)

        upper_low, _, lower_low = bb(data_low_vol)
        upper_high, _, lower_high = bb(data_high_vol)

        # Band width should be larger in high volatility
        width_low = np.nanmean(upper_low - lower_low)
        width_high = np.nanmean(upper_high - lower_high)
        assert width_high > width_low

    def test_bollinger_signals(self):
        """Test signal generation from band touches."""
        data = np.array([100, 95, 96, 97, 105, 104, 103], dtype=float)
        bb = BollingerBands(period=2, num_std_dev=1.0)
        upper, middle, lower = bb(data)
        signals = bb.get_signals(data, upper, lower)

        # Should generate some signals
        assert np.any(signals != 0)

    def test_bollinger_bandwidth(self):
        """Test band width calculation."""
        data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108],
                        dtype=float)
        bb = BollingerBands(period=2, num_std_dev=1.0)
        upper, middle, lower = bb(data)
        width = bb.get_band_width(upper, lower)

        # Width should be positive and non-NaN where bands exist
        valid = ~np.isnan(width)
        assert np.all(width[valid] > 0)

    def test_bollinger_bandwidth_percent(self):
        """Test bandwidth percent (position within bands)."""
        data = np.array([100, 102, 101, 103, 105, 104, 106, 105, 107, 108],
                        dtype=float)
        bb = BollingerBands(period=2, num_std_dev=2.0)
        upper, middle, lower = bb(data)
        bw_pct = bb.get_bandwidth_percent(data, upper, middle, lower)

        # Should be between 0 and 100
        valid = ~np.isnan(bw_pct)
        assert np.all((bw_pct[valid] >= 0) & (bw_pct[valid] <= 100))
```

### 3. Volatility Analysis

Bollinger Bands teach important lesson about volatility:

```python
def analyze_volatility(data: np.ndarray):
    """Demonstrate volatility changes."""
    bb = BollingerBands(period=20, num_std_dev=2.0)
    upper, middle, lower = bb(data)

    band_width = bb.get_band_width(upper, lower)

    # Narrow bands (squeeze) often precede big moves
    squeeze = band_width < np.nanpercentile(band_width, 25)
    print(f"Squeeze periods: {np.sum(squeeze)}")

    # Expanding bands show increasing volatility
    bb_expansion = np.diff(band_width) > 0
    print(f"Expanding bands: {np.sum(bb_expansion)}")
```

## Validation

### Statistical Property

Bollinger Bands with 2 standard deviations should contain ~95% of prices (normally distributed data):

```python
data = np.random.normal(loc=100, scale=2, size=500)
bb = BollingerBands(period=20, num_std_dev=2.0)
upper, middle, lower = bb(data)

within = (data >= lower) & (data <= upper)
pct = np.sum(within) / len(within)
print(f"% within bands: {pct:.1%}")  # Should be ~95%
```

### Against TradingView
1. Add Bollinger Bands(20, 2) to chart
2. Verify:
   - Bands widen during volatile periods
   - Bands narrow during quiet periods
   - Price rarely extends beyond bands
   - Band touches often precede reversals

## Run Tests

```bash
pytest tests/test_indicators.py::TestBollingerBands -v
pytest tests/test_indicators.py::TestBollingerBands --cov=src/indicators
```

## Success Criteria

- [ ] BollingerBands class implemented with correct bands
- [ ] Returns tuple of (upper_band, middle_band, lower_band)
- [ ] Band width calculation correct
- [ ] Bandwidth percentage calculation correct
- [ ] All 6 unit tests pass
- [ ] 100% code coverage
- [ ] Signal generation works (band touches)
- [ ] Statistical property validated (~95% within bands)

## Key Insights to Document

1. **Mean reversion**: How Bollinger Bands differ from trend-following
2. **Volatility**: How band width represents volatility
3. **Squeeze & breakout**: What narrow bands predict
4. **Signal vs noise**: When band touches are valid signals vs noise

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using sample std dev (ddof=1) | Use population std dev (ddof=0) |
| Bands reversed (lower > upper) | Must be: upper = SMA + (2 × std), lower = SMA - (2 × std) |
| Not handling edges | First period-1 values should be NaN |
| Wrong period | Standard is period=20 |
| Band touches → immediate trade | Band touch is signal, still needs confirmation |

## Trading Concepts Learned

**Mean Reversion vs Trend Following**:
- Bollinger Bands (mean reversion): Buy oversold, sell overbought
- SMA/EMA/MACD (trend following): Buy uptrend, sell downtrend
- Different tools for different market conditions!

## Next Steps

Once complete:
1. ✅ Document volatility insights
2. ✅ Understand mean reversion vs trending
3. ✅ **PHASE 1 COMPLETE!**
4. ✅ Move to Phase 2: LLM Daemon Integration

---

**Final stretch! 🚀 You've built 5 indicators from scratch!**
