"""Synthetic risk data generator for testing bubble detection.

This module generates realistic synthetic market data for 2015-2024 (10 years, ~2500 trading days)
with temporal correlation and mean-reversion properties. Enables testing without real market APIs.

Generates 4 risk factor inputs:
1. P/E Ratio: 15-40 range with mean reversion toward 22
2. VIX: 10-50 range with jump diffusion and clustering
3. Market Breadth: 20-90% with regime-switching and lagged correlation
4. Momentum: -15% to +15% weekly change derived from price series
"""

from typing import Dict, Optional
import numpy as np


class SyntheticRiskDataGenerator:
    """Generate realistic synthetic market data for risk assessment testing."""

    def __init__(
        self,
        seed: Optional[int] = None,
        start_year: int = 2015,
        end_year: int = 2024
    ):
        """Initialize synthetic data generator.

        Args:
            seed: Random seed for reproducibility (default: None for random)
            start_year: Start year for data generation (default: 2015)
            end_year: End year for data generation (default: 2024)
        """
        self.seed = seed
        self.start_year = start_year
        self.end_year = end_year
        self.num_years = end_year - start_year + 1
        self.trading_days_per_year = 252
        self.num_days = self.num_years * self.trading_days_per_year

        # Create random number generator for reproducibility
        self.rng = np.random.default_rng(seed)

    def generate_pe_ratio(self, num_days: Optional[int] = None) -> np.ndarray:
        """Generate P/E ratio series with mean reversion using Ornstein-Uhlenbeck process.

        Args:
            num_days: Number of trading days to generate (default: self.num_days)

        Returns:
            Array of P/E ratios in range 15-40 with mean reversion toward 22

        Properties:
            - Mean reversion toward 22
            - Bubble spikes to 35-40 (2021, 2000-era behavior)
            - Crash drops to 15-18 (2008, 2020-era behavior)
            - Daily random walk with drift
        """
        if num_days is None:
            num_days = self.num_days

        # Ornstein-Uhlenbeck parameters
        mean = 22.0  # Long-term average P/E
        theta = 0.01  # Mean reversion speed (slow)
        sigma = 3.0  # Volatility
        x0 = 20.0  # Initial P/E

        # Generate base OU process
        pe_series = self._ou_process(mean, theta, sigma, x0, num_days)

        # Only add structural events if series is long enough
        if num_days >= 500:
            # Add structural bubbles and crashes
            # 2008-style crash (around day 750 for 2018)
            crash_day = int(num_days * 0.3)  # ~30% through
            crash_len = min(60, num_days - crash_day)
            recovery_len = min(60, num_days - crash_day - crash_len)
            if crash_len > 0:
                pe_series[crash_day:crash_day+crash_len] -= np.linspace(0, 7, crash_len)
            if recovery_len > 0:
                pe_series[crash_day+crash_len:crash_day+crash_len+recovery_len] += np.linspace(0, 5, recovery_len)

            # 2020-style volatility (around day 1250 for 2020)
            covid_day = int(num_days * 0.5)
            covid_drop_len = min(30, num_days - covid_day)
            covid_recovery_len = min(60, num_days - covid_day - covid_drop_len)
            if covid_drop_len > 0:
                pe_series[covid_day:covid_day+covid_drop_len] -= np.linspace(0, 6, covid_drop_len)
            if covid_recovery_len > 0:
                pe_series[covid_day+covid_drop_len:covid_day+covid_drop_len+covid_recovery_len] += np.linspace(0, 8, covid_recovery_len)

            # 2021-style bubble (around day 1500 for 2021)
            bubble_day = int(num_days * 0.6)
            bubble_rise_len = min(200, num_days - bubble_day)
            bubble_deflate_len = min(100, num_days - bubble_day - bubble_rise_len)
            if bubble_rise_len > 0:
                pe_series[bubble_day:bubble_day+bubble_rise_len] += np.linspace(0, 10, bubble_rise_len)
            if bubble_deflate_len > 0:
                pe_series[bubble_day+bubble_rise_len:bubble_day+bubble_rise_len+bubble_deflate_len] -= np.linspace(0, 5, bubble_deflate_len)

        # Clamp to realistic range [10, 50]
        pe_series = np.clip(pe_series, 10.0, 50.0)

        return pe_series

    def generate_vix(self, num_days: Optional[int] = None) -> np.ndarray:
        """Generate VIX series with jumps and clustering using jump-diffusion process.

        Args:
            num_days: Number of trading days to generate (default: self.num_days)

        Returns:
            Array of VIX values in range 10-50 with jump clustering

        Properties:
            - Base level 15-20 (normal markets)
            - Cluster during stress periods (jumps to 40+, stays elevated)
            - Spikes to 50+ during crashes
            - Jump diffusion (sudden spikes)
        """
        if num_days is None:
            num_days = self.num_days

        # Base mean-reverting process
        mean = 16.0  # Normal VIX level
        theta = 0.02  # Faster mean reversion than P/E
        sigma = 2.0  # Base volatility
        x0 = 15.0  # Initial VIX

        # Generate base OU process
        vix_base = self._ou_process(mean, theta, sigma, x0, num_days)

        # Add jump diffusion component
        jump_prob = 0.015  # 1.5% daily probability of jump
        jump_std = 8.0  # Jump size standard deviation
        vix_series = self._jump_diffusion(vix_base, jump_prob, jump_std)

        # Only add structural events if series is long enough
        if num_days >= 500:
            # Add structural stress periods
            # 2018 crash spike
            crash_day = int(num_days * 0.3)
            crash_spike_len = min(30, num_days - crash_day)
            crash_decay_len = min(60, num_days - crash_day - crash_spike_len)
            if crash_spike_len > 0:
                vix_series[crash_day:crash_day+crash_spike_len] += np.linspace(0, 25, crash_spike_len)
            if crash_decay_len > 0:
                vix_series[crash_day+crash_spike_len:crash_day+crash_spike_len+crash_decay_len] -= np.linspace(0, 20, crash_decay_len)

            # 2020 COVID panic
            covid_day = int(num_days * 0.5)
            covid_spike_len = min(20, num_days - covid_day)
            covid_decay_len = min(100, num_days - covid_day - covid_spike_len)
            if covid_spike_len > 0:
                vix_series[covid_day:covid_day+covid_spike_len] += np.linspace(0, 35, covid_spike_len)
            if covid_decay_len > 0:
                vix_series[covid_day+covid_spike_len:covid_day+covid_spike_len+covid_decay_len] -= np.linspace(0, 30, covid_decay_len)

        # Clamp to realistic range [5, 80]
        vix_series = np.clip(vix_series, 5.0, 80.0)

        return vix_series

    def generate_breadth(
        self,
        num_days: Optional[int] = None,
        price_series: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate market breadth series with regime-switching and lagged correlation.

        Args:
            num_days: Number of trading days to generate (default: self.num_days)
            price_series: Optional price series to correlate with (generates internal if None)

        Returns:
            Array of breadth percentages (0-100) representing % stocks above 200-day MA

        Properties:
            - Correlated with price (uptrend → high breadth)
            - Mean reversion toward 50%
            - Lags price by 5-10 days
            - Ranges 20-30% during downtrends, 70-90% during uptrends
        """
        if num_days is None:
            num_days = self.num_days

        # Generate price series if not provided
        if price_series is None:
            price_series = self._create_price_series(num_days)

        # Calculate price trend (smoothed momentum over 20 days)
        trend = np.convolve(
            np.diff(price_series, prepend=price_series[0]),
            np.ones(20) / 20,
            mode='same'
        )

        # Initialize breadth array
        breadth = np.zeros(num_days)
        breadth[0] = 50.0  # Start at neutral

        # Regime-switching based on lagged price trend
        lag = 7  # Breadth lags price by ~7 days
        mean_reversion_speed = 0.05

        for i in range(1, num_days):
            # Determine regime from lagged trend
            lagged_trend = trend[max(0, i - lag)]

            if lagged_trend > 0:
                # Uptrend regime: target 70-85%
                target = 75.0 + self.rng.standard_normal() * 5
            else:
                # Downtrend regime: target 25-40%
                target = 35.0 + self.rng.standard_normal() * 5

            # Mean reversion toward target
            breadth[i] = breadth[i-1] + mean_reversion_speed * (target - breadth[i-1])

            # Add noise
            breadth[i] += self.rng.standard_normal() * 1.5

        # Clamp to valid range [0, 100]
        breadth = np.clip(breadth, 0.0, 100.0)

        return breadth

    def generate_momentum(
        self,
        price_series: Optional[np.ndarray] = None,
        num_days: Optional[int] = None
    ) -> np.ndarray:
        """Calculate weekly momentum from price series.

        Args:
            price_series: Price series to calculate momentum from (generates if None)
            num_days: Number of trading days (default: self.num_days)

        Returns:
            Array of weekly percentage changes in range -15% to +15%

        Properties:
            - Derived from price data (actual weekly returns)
            - Normal distribution centered at 0
            - Std deviation 3-5% (volatile periods use 8-10%)
            - Autocorrelation (momentum persistence)
        """
        if num_days is None:
            num_days = self.num_days

        # Generate price series if not provided
        if price_series is None:
            price_series = self._create_price_series(num_days)

        # Calculate weekly returns (5-day lookback for trading week)
        lookback = 5
        momentum = np.zeros(num_days)

        for i in range(lookback, num_days):
            momentum[i] = (price_series[i] - price_series[i - lookback]) / price_series[i - lookback] * 100

        # Add volatility regime changes
        # Higher volatility during crashes and bubbles
        crash_day = int(num_days * 0.3)
        covid_day = int(num_days * 0.5)

        # Increase volatility during stress periods
        for day in [crash_day, covid_day]:
            momentum[day:day+60] *= 1.5  # 50% more volatile

        # Clamp to reasonable range [-20, 20]
        momentum = np.clip(momentum, -20.0, 20.0)

        return momentum

    def generate_market_scenario(
        self,
        scenario: str = "normal",
        num_days: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Generate pre-built market scenarios for testing.

        Args:
            scenario: Scenario type - "normal", "bubble", "crash", "recovery"
            num_days: Number of trading days (default: self.num_days)

        Returns:
            Dict containing:
                - pe_ratio: P/E series
                - vix: VIX series
                - breadth: Market breadth series
                - momentum: Momentum series
                - price: Price series

        Raises:
            ValueError: If scenario is not recognized
        """
        valid_scenarios = ["normal", "bubble", "crash", "recovery"]
        if scenario not in valid_scenarios:
            raise ValueError(f"Invalid scenario '{scenario}'. Must be one of {valid_scenarios}")

        if num_days is None:
            num_days = self.num_days

        # Generate base data
        price_series = self._create_price_series(num_days)
        pe_base = self.generate_pe_ratio(num_days)
        vix_base = self.generate_vix(num_days)
        breadth_base = self.generate_breadth(num_days, price_series)
        momentum_base = self.generate_momentum(price_series, num_days)

        # Modify based on scenario
        if scenario == "normal":
            # 70% of days in normal range
            # P/E: 18-25, VIX: 12-20, Breadth: 45-65%, Momentum: -3% to +3%
            pe = np.clip(pe_base * 0.9 + 20, 18, 25)
            vix = np.clip(vix_base * 0.5 + 15, 12, 20)
            breadth = np.clip(breadth_base * 0.4 + 55, 45, 65)
            momentum = np.clip(momentum_base * 0.5, -3, 3)

        elif scenario == "bubble":
            # 30% bubble characteristics
            # P/E: 35-40, VIX: 10-15 (complacency), Breadth: 80%+, Momentum: +5-10%
            pe = np.clip(pe_base * 1.3 + 10, 35, 40)
            vix = np.clip(vix_base * 0.3 + 12, 10, 15)
            breadth = np.clip(breadth_base * 0.2 + 80, 75, 95)
            momentum = np.clip(momentum_base * 1.5 + 7, 5, 10)

        elif scenario == "crash":
            # 30% crash characteristics
            # P/E: 15-18, VIX: 45-50, Breadth: 20-30%, Momentum: -10% to -15%
            pe = np.clip(pe_base * 0.7 + 5, 15, 18)
            vix = np.clip(vix_base * 1.5 + 30, 45, 50)
            breadth = np.clip(breadth_base * 0.3 + 10, 20, 30)
            momentum = np.clip(momentum_base * 2.0 - 12, -15, -10)

        else:  # recovery
            # Mixed recovery (VIX dropping, breadth improving)
            # P/E: 20-28, VIX: 25-35 (elevated but declining), Breadth: 40-60% (improving), Momentum: -2% to +5%
            pe = np.clip(pe_base * 1.0 + 2, 20, 28)
            vix = np.clip(vix_base * 1.2 + 10, 25, 35)
            # Create improving breadth trend
            breadth_trend = np.linspace(40, 60, num_days)
            breadth = np.clip(breadth_base * 0.3 + breadth_trend, 40, 60)
            momentum = np.clip(momentum_base * 0.8 + 1.5, -2, 5)

        return {
            'pe_ratio': pe,
            'vix': vix,
            'breadth': breadth,
            'momentum': momentum,
            'price': price_series
        }

    # Private helper methods

    def _ou_process(
        self,
        mean: float,
        theta: float,
        sigma: float,
        x0: float,
        num_steps: int
    ) -> np.ndarray:
        """Generate Ornstein-Uhlenbeck mean-reverting process.

        Args:
            mean: Long-term mean to revert to
            theta: Mean reversion speed (higher = faster reversion)
            sigma: Volatility
            x0: Initial value
            num_steps: Number of time steps

        Returns:
            Array of values following OU process
        """
        if num_steps == 0:
            return np.array([])

        dt = 1.0  # Daily timestep
        x = np.zeros(num_steps)
        x[0] = x0

        for t in range(1, num_steps):
            dx = theta * (mean - x[t-1]) * dt + sigma * np.sqrt(dt) * self.rng.standard_normal()
            x[t] = x[t-1] + dx

        return x

    def _jump_diffusion(
        self,
        base_process: np.ndarray,
        jump_prob: float,
        jump_std: float
    ) -> np.ndarray:
        """Add jump component to base process.

        Args:
            base_process: Base diffusion process
            jump_prob: Daily probability of jump (0-1)
            jump_std: Standard deviation of jump size

        Returns:
            Process with jumps added
        """
        num_steps = len(base_process)
        process = base_process.copy()

        # Generate jump events
        jumps = self.rng.random(num_steps) < jump_prob
        jump_indices = np.where(jumps)[0]

        # Add positive jumps (volatility spikes are upward)
        for idx in jump_indices:
            jump_size = abs(self.rng.standard_normal() * jump_std)
            process[idx] += jump_size

            # Jump clustering: elevated base for next 3-5 days
            cluster_days = self.rng.integers(3, 6)
            process[idx:min(idx+cluster_days, num_steps)] += jump_size * 0.3

        return process

    def _create_price_series(self, num_days: int) -> np.ndarray:
        """Generate realistic price series for SPY-like instrument.

        Args:
            num_days: Number of trading days

        Returns:
            Realistic price series with trends and volatility
        """
        # Start at 200 (typical SPY-like level in 2015)
        price = np.zeros(num_days)
        price[0] = 200.0

        # Generate returns with trend and volatility regimes
        base_drift = 0.0003  # ~7.5% annual return
        base_vol = 0.01  # ~15% annual volatility

        for i in range(1, num_days):
            # Add structural trends
            year_frac = i / self.trading_days_per_year

            # 2018 crash (year ~3)
            if 2.8 < year_frac < 3.2:
                drift = -0.001
                vol = 0.02
            # 2020 COVID (year ~5)
            elif 4.8 < year_frac < 5.3:
                drift = -0.002 if i < int(num_days * 0.51) else 0.002  # V-recovery
                vol = 0.03
            # 2021 bubble (year ~6)
            elif 5.5 < year_frac < 6.5:
                drift = 0.001
                vol = 0.008
            else:
                drift = base_drift
                vol = base_vol

            # Generate return
            ret = drift + vol * self.rng.standard_normal()
            price[i] = price[i-1] * (1 + ret)

        return price
