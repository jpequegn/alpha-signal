"""Risk factor calculator for bubble detection.

This module computes 4 independent risk factors on a 0.0-0.8 scale:
1. Valuation Risk (P/E ratio historical context)
2. Volatility Risk (VIX as proxy for fear)
3. Market Breadth Risk (% of stocks above 200-day MA)
4. Momentum Risk (rate of price change)
"""

from typing import Dict


class RiskFactorCalculator:
    """Calculate independent risk factors for bubble detection."""

    def __init__(self):
        """Initialize risk factor calculator."""
        pass

    def calculate_valuation_risk(
        self,
        pe_ratio: float,
        pe_percentiles: Dict[str, float]
    ) -> float:
        """Calculate valuation risk based on P/E ratio historical context.

        Args:
            pe_ratio: Current P/E ratio
            pe_percentiles: Dict with 'p25', 'p75', 'p90' percentile values

        Returns:
            Risk score 0.0-0.8 where:
                - 0.0-0.2: Normal valuation (below 75th percentile)
                - 0.3-0.6: Moderate valuation (75th-90th percentile)
                - 0.6-0.8: Elevated valuation (above 90th percentile)

        Raises:
            ValueError: If pe_ratio <= 0 or required percentiles missing
        """
        if pe_ratio <= 0:
            raise ValueError(f"P/E ratio must be positive, got {pe_ratio}")

        required_keys = {'p25', 'p75', 'p90'}
        if not required_keys.issubset(pe_percentiles.keys()):
            raise ValueError(f"Missing required percentiles: {required_keys - pe_percentiles.keys()}")

        p25 = pe_percentiles['p25']
        p75 = pe_percentiles['p75']
        p90 = pe_percentiles['p90']

        # Validate percentiles are in order
        if not (p25 < p75 < p90):
            raise ValueError(f"Percentiles must be in ascending order: p25={p25}, p75={p75}, p90={p90}")

        # Below 75th percentile: 0.0-0.2 (normal)
        if pe_ratio <= p75:
            # Linear interpolation from p25 to p75 → 0.0 to 0.2
            if pe_ratio <= p25:
                return 0.0
            ratio = (pe_ratio - p25) / (p75 - p25)
            return 0.0 + ratio * 0.2

        # Between 75th and 90th percentile: 0.3-0.6 (moderate)
        elif pe_ratio <= p90:
            ratio = (pe_ratio - p75) / (p90 - p75)
            return 0.3 + ratio * 0.3

        # Above 90th percentile: 0.6-0.8 (elevated)
        else:
            # Cap at 0.8 for extremely high P/E
            # Assume 1.5x p90 = 0.8
            max_pe = p90 * 1.5
            if pe_ratio >= max_pe:
                return 0.8
            ratio = (pe_ratio - p90) / (max_pe - p90)
            return 0.6 + ratio * 0.2

    def calculate_volatility_risk(self, vix: float) -> float:
        """Calculate volatility risk based on VIX fear index.

        Args:
            vix: VIX value (typically 10-50 range)

        Returns:
            Risk score 0.0-0.8 where:
                - 0.0-0.1: VIX < 15 (very low fear)
                - 0.1-0.4: VIX 15-30 (normal)
                - 0.4-0.6: VIX 30-40 (elevated)
                - 0.6-0.8: VIX > 40 (extreme fear/greed)

        Raises:
            ValueError: If vix < 0
        """
        if vix < 0:
            raise ValueError(f"VIX must be non-negative, got {vix}")

        # VIX < 15: Very low (0.0-0.1)
        if vix < 15:
            return min(0.1, vix / 15 * 0.1)

        # VIX 15-30: Normal (0.1-0.4)
        elif vix < 30:
            ratio = (vix - 15) / 15
            return 0.1 + ratio * 0.3

        # VIX 30-40: Elevated (0.4-0.6)
        elif vix < 40:
            ratio = (vix - 30) / 10
            return 0.4 + ratio * 0.2

        # VIX >= 40: Extreme (0.6-0.8)
        else:
            # Cap at 0.8 for VIX >= 50
            if vix >= 50:
                return 0.8
            ratio = (vix - 40) / 10
            return 0.6 + ratio * 0.2

    def calculate_breadth_risk(self, breadth_percentage: float) -> float:
        """Calculate market breadth risk (inverse relationship).

        Args:
            breadth_percentage: % of stocks above 200-day MA (0-100)

        Returns:
            Risk score 0.0-0.8 where:
                - 0.0-0.2: Breadth > 70% (healthy participation)
                - 0.2-0.4: Breadth 50-70% (moderate)
                - 0.4-0.6: Breadth 30-50% (declining)
                - 0.6-0.8: Breadth < 30% (severe decline)

        Raises:
            ValueError: If breadth not in 0-100 range
        """
        if not 0 <= breadth_percentage <= 100:
            raise ValueError(f"Breadth must be 0-100%, got {breadth_percentage}")

        # Breadth > 70%: Healthy (0.0-0.2)
        if breadth_percentage > 70:
            # Higher breadth = lower risk (inverse)
            # 100% → 0.0, 70% → 0.2
            ratio = (100 - breadth_percentage) / 30
            return 0.0 + ratio * 0.2

        # Breadth 50-70%: Moderate (0.2-0.4)
        elif breadth_percentage >= 50:
            ratio = (70 - breadth_percentage) / 20
            return 0.2 + ratio * 0.2

        # Breadth 30-50%: Declining (0.4-0.6)
        elif breadth_percentage >= 30:
            ratio = (50 - breadth_percentage) / 20
            return 0.4 + ratio * 0.2

        # Breadth < 30%: Severe (0.6-0.8)
        else:
            # 30% → 0.6, 0% → 0.8
            ratio = (30 - breadth_percentage) / 30
            return 0.6 + ratio * 0.2

    def calculate_momentum_risk(self, price_change_pct: float) -> float:
        """Calculate momentum risk based on rate of price change.

        Args:
            price_change_pct: Weekly percentage change in price (-100 to +100+)

        Returns:
            Risk score 0.0-0.8 where:
                - 0.5-0.7: < -10% (panic selling)
                - 0.3-0.5: -10% to -5% (strong downside)
                - 0.0-0.2: -5% to +5% (normal)
                - 0.1-0.3: +5% to +10% (strong upside)
                - 0.3-0.5: > +10% (parabolic move)

        Raises:
            ValueError: If price_change_pct is unreasonably extreme (< -100%)
        """
        if price_change_pct < -100:
            raise ValueError(f"Price change cannot be less than -100%, got {price_change_pct}%")

        # Panic selling: < -10% (0.5-0.7)
        if price_change_pct < -10:
            # -15% → 0.7, -10% → 0.5
            # Cap at -15% for 0.7
            if price_change_pct <= -15:
                return 0.7
            ratio = (-10 - price_change_pct) / 5
            return 0.5 + ratio * 0.2

        # Strong downside: -10% to -5% (0.3-0.5)
        elif price_change_pct < -5:
            ratio = (-5 - price_change_pct) / 5
            return 0.3 + ratio * 0.2

        # Normal: -5% to +5% (0.0-0.2)
        elif price_change_pct <= 5:
            # Map -5 to +5 → 0.0 to 0.2
            ratio = (price_change_pct + 5) / 10
            return 0.0 + ratio * 0.2

        # Strong upside: +5% to +10% (0.1-0.3)
        elif price_change_pct <= 10:
            ratio = (price_change_pct - 5) / 5
            return 0.1 + ratio * 0.2

        # Parabolic: > +10% (0.3-0.5)
        else:
            # Cap at +15% for 0.5
            if price_change_pct >= 15:
                return 0.5
            ratio = (price_change_pct - 10) / 5
            return 0.3 + ratio * 0.2

    def aggregate_risks(
        self,
        valuation_risk: float,
        volatility_risk: float,
        breadth_risk: float,
        momentum_risk: float
    ) -> Dict[str, float]:
        """Aggregate all risk factors into a comprehensive risk profile.

        Args:
            valuation_risk: Valuation risk score (0.0-0.8)
            volatility_risk: Volatility risk score (0.0-0.8)
            breadth_risk: Market breadth risk score (0.0-0.8)
            momentum_risk: Momentum risk score (0.0-0.8)

        Returns:
            Dict containing:
                - valuation_risk: Input valuation risk
                - volatility_risk: Input volatility risk
                - breadth_risk: Input breadth risk
                - momentum_risk: Input momentum risk
                - average_risk: Simple average of all 4 factors
                - max_risk: Maximum of all 4 factors
                - min_risk: Minimum of all 4 factors

        Raises:
            ValueError: If any risk score is not in 0.0-0.8 range
        """
        risks = [valuation_risk, volatility_risk, breadth_risk, momentum_risk]
        risk_names = ['valuation_risk', 'volatility_risk', 'breadth_risk', 'momentum_risk']

        # Validate all inputs are in valid range
        for name, risk in zip(risk_names, risks):
            if not 0.0 <= risk <= 0.8:
                raise ValueError(f"{name} must be in 0.0-0.8 range, got {risk}")

        return {
            'valuation_risk': valuation_risk,
            'volatility_risk': volatility_risk,
            'breadth_risk': breadth_risk,
            'momentum_risk': momentum_risk,
            'average_risk': sum(risks) / len(risks),
            'max_risk': max(risks),
            'min_risk': min(risks),
        }
