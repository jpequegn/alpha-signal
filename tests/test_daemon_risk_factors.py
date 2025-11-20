"""Tests for risk factor calculator."""

import pytest
from src.daemon.risk_factors import RiskFactorCalculator


@pytest.fixture
def calculator():
    """Create calculator instance."""
    return RiskFactorCalculator()


@pytest.fixture
def sample_percentiles():
    """Sample P/E percentiles for testing."""
    return {
        'p25': 18.0,
        'p75': 28.0,
        'p90': 35.0
    }


class TestValuationRisk:
    """Test valuation risk calculation."""

    def test_normal_valuation_low(self, calculator, sample_percentiles):
        """Test low normal valuation (below p25)."""
        risk = calculator.calculate_valuation_risk(15.0, sample_percentiles)
        assert 0.0 <= risk <= 0.2
        assert risk == 0.0  # Below p25

    def test_normal_valuation_mid(self, calculator, sample_percentiles):
        """Test mid normal valuation (between p25 and p75)."""
        risk = calculator.calculate_valuation_risk(23.0, sample_percentiles)  # Midpoint
        assert 0.0 <= risk <= 0.2
        assert 0.05 <= risk <= 0.15  # Should be around 0.1

    def test_moderate_valuation_low(self, calculator, sample_percentiles):
        """Test low moderate valuation (just above p75)."""
        risk = calculator.calculate_valuation_risk(28.1, sample_percentiles)
        assert 0.3 <= risk <= 0.6

    def test_moderate_valuation_high(self, calculator, sample_percentiles):
        """Test high moderate valuation (near p90)."""
        risk = calculator.calculate_valuation_risk(34.0, sample_percentiles)
        assert 0.3 <= risk <= 0.6

    def test_elevated_valuation_low(self, calculator, sample_percentiles):
        """Test low elevated valuation (just above p90)."""
        risk = calculator.calculate_valuation_risk(36.0, sample_percentiles)
        assert 0.6 <= risk <= 0.8

    def test_elevated_valuation_extreme(self, calculator, sample_percentiles):
        """Test extreme elevated valuation (>> p90)."""
        risk = calculator.calculate_valuation_risk(60.0, sample_percentiles)
        assert risk == 0.8  # Capped

    def test_invalid_pe_negative(self, calculator, sample_percentiles):
        """Test negative P/E raises error."""
        with pytest.raises(ValueError, match="P/E ratio must be positive"):
            calculator.calculate_valuation_risk(-5.0, sample_percentiles)

    def test_invalid_pe_zero(self, calculator, sample_percentiles):
        """Test zero P/E raises error."""
        with pytest.raises(ValueError, match="P/E ratio must be positive"):
            calculator.calculate_valuation_risk(0.0, sample_percentiles)

    def test_missing_percentiles(self, calculator):
        """Test missing percentiles raises error."""
        with pytest.raises(ValueError, match="Missing required percentiles"):
            calculator.calculate_valuation_risk(25.0, {'p25': 18.0})

    def test_invalid_percentile_order(self, calculator):
        """Test invalid percentile order raises error."""
        with pytest.raises(ValueError, match="must be in ascending order"):
            calculator.calculate_valuation_risk(25.0, {'p25': 35.0, 'p75': 28.0, 'p90': 18.0})

    @pytest.mark.parametrize("pe_ratio,expected_low,expected_high", [
        (15.0, 0.0, 0.05),    # Well below p25
        (18.0, 0.0, 0.05),    # At p25
        (23.0, 0.08, 0.12),   # Mid p25-p75
        (28.0, 0.18, 0.22),   # At p75 (boundary)
        (31.5, 0.4, 0.5),     # Mid p75-p90
        (35.0, 0.58, 0.62),   # At p90 (boundary)
        (40.0, 0.63, 0.69),   # Above p90 (40-35)/(52.5-35) = 5/17.5 = 0.286 → 0.6+0.057 = 0.657
        (52.5, 0.78, 0.8),    # Near cap (1.5x p90)
        (100.0, 0.8, 0.8),    # Well above cap
    ])
    def test_valuation_boundaries(self, calculator, sample_percentiles, pe_ratio, expected_low, expected_high):
        """Test valuation risk at various P/E boundaries."""
        risk = calculator.calculate_valuation_risk(pe_ratio, sample_percentiles)
        assert expected_low <= risk <= expected_high


class TestVolatilityRisk:
    """Test volatility risk calculation."""

    def test_very_low_vix(self, calculator):
        """Test very low VIX (< 15)."""
        risk = calculator.calculate_volatility_risk(10.0)
        assert 0.0 <= risk <= 0.1

    def test_normal_vix_low(self, calculator):
        """Test normal VIX (15-30) - low end."""
        risk = calculator.calculate_volatility_risk(20.0)
        assert 0.1 <= risk <= 0.4

    def test_normal_vix_high(self, calculator):
        """Test normal VIX (15-30) - high end."""
        risk = calculator.calculate_volatility_risk(28.0)
        assert 0.1 <= risk <= 0.4

    def test_elevated_vix(self, calculator):
        """Test elevated VIX (30-40)."""
        risk = calculator.calculate_volatility_risk(35.0)
        assert 0.4 <= risk <= 0.6

    def test_extreme_vix_low(self, calculator):
        """Test extreme VIX (40-50)."""
        risk = calculator.calculate_volatility_risk(45.0)
        assert 0.6 <= risk <= 0.8

    def test_extreme_vix_capped(self, calculator):
        """Test VIX above cap (>= 50)."""
        risk = calculator.calculate_volatility_risk(60.0)
        assert risk == 0.8

    def test_invalid_vix_negative(self, calculator):
        """Test negative VIX raises error."""
        with pytest.raises(ValueError, match="VIX must be non-negative"):
            calculator.calculate_volatility_risk(-5.0)

    @pytest.mark.parametrize("vix,expected_low,expected_high", [
        (0.0, 0.0, 0.0),      # Zero VIX
        (10.0, 0.0, 0.1),     # Very low
        (15.0, 0.1, 0.12),    # Normal low (boundary)
        (22.5, 0.2, 0.3),     # Normal mid
        (30.0, 0.38, 0.42),   # Elevated low (boundary)
        (35.0, 0.48, 0.52),   # Elevated mid
        (40.0, 0.58, 0.62),   # Extreme low (boundary)
        (45.0, 0.68, 0.72),   # Extreme mid
        (50.0, 0.78, 0.8),    # Extreme high (boundary)
        (100.0, 0.8, 0.8),    # Way above cap
    ])
    def test_volatility_boundaries(self, calculator, vix, expected_low, expected_high):
        """Test volatility risk at various VIX levels."""
        risk = calculator.calculate_volatility_risk(vix)
        assert expected_low <= risk <= expected_high


class TestBreadthRisk:
    """Test market breadth risk calculation."""

    def test_healthy_breadth_high(self, calculator):
        """Test healthy breadth (> 70%) - high end."""
        risk = calculator.calculate_breadth_risk(90.0)
        assert 0.0 <= risk <= 0.2

    def test_healthy_breadth_low(self, calculator):
        """Test healthy breadth (> 70%) - low end."""
        risk = calculator.calculate_breadth_risk(75.0)
        assert 0.0 <= risk <= 0.2

    def test_moderate_breadth(self, calculator):
        """Test moderate breadth (50-70%)."""
        risk = calculator.calculate_breadth_risk(60.0)
        assert 0.2 <= risk <= 0.4

    def test_declining_breadth(self, calculator):
        """Test declining breadth (30-50%)."""
        risk = calculator.calculate_breadth_risk(40.0)
        assert 0.4 <= risk <= 0.6

    def test_severe_breadth_high(self, calculator):
        """Test severe breadth (< 30%) - high end."""
        risk = calculator.calculate_breadth_risk(25.0)
        assert 0.6 <= risk <= 0.8

    def test_severe_breadth_zero(self, calculator):
        """Test zero breadth (worst case)."""
        risk = calculator.calculate_breadth_risk(0.0)
        assert risk == 0.8

    def test_invalid_breadth_negative(self, calculator):
        """Test negative breadth raises error."""
        with pytest.raises(ValueError, match="Breadth must be 0-100%"):
            calculator.calculate_breadth_risk(-10.0)

    def test_invalid_breadth_over_100(self, calculator):
        """Test > 100% breadth raises error."""
        with pytest.raises(ValueError, match="Breadth must be 0-100%"):
            calculator.calculate_breadth_risk(105.0)

    @pytest.mark.parametrize("breadth,expected_low,expected_high", [
        (100.0, 0.0, 0.05),   # Perfect breadth
        (85.0, 0.08, 0.12),   # Healthy high
        (70.0, 0.18, 0.22),   # Healthy low (boundary)
        (60.0, 0.28, 0.32),   # Moderate mid
        (50.0, 0.38, 0.42),   # Moderate low (boundary)
        (40.0, 0.48, 0.52),   # Declining mid
        (30.0, 0.58, 0.62),   # Declining low (boundary)
        (15.0, 0.68, 0.72),   # Severe mid
        (0.0, 0.78, 0.8),     # Zero breadth (worst)
    ])
    def test_breadth_boundaries(self, calculator, breadth, expected_low, expected_high):
        """Test breadth risk at various levels."""
        risk = calculator.calculate_breadth_risk(breadth)
        assert expected_low <= risk <= expected_high


class TestMomentumRisk:
    """Test momentum risk calculation."""

    def test_panic_selling_extreme(self, calculator):
        """Test extreme panic selling (< -10%)."""
        risk = calculator.calculate_momentum_risk(-20.0)
        assert risk == 0.7  # Capped

    def test_panic_selling_moderate(self, calculator):
        """Test moderate panic selling."""
        risk = calculator.calculate_momentum_risk(-12.0)
        assert 0.5 <= risk <= 0.7

    def test_strong_downside(self, calculator):
        """Test strong downside (-10% to -5%)."""
        risk = calculator.calculate_momentum_risk(-7.5)
        assert 0.3 <= risk <= 0.5

    def test_normal_negative(self, calculator):
        """Test normal negative movement (-5% to 0%)."""
        risk = calculator.calculate_momentum_risk(-2.0)
        assert 0.0 <= risk <= 0.2

    def test_normal_positive(self, calculator):
        """Test normal positive movement (0% to +5%)."""
        risk = calculator.calculate_momentum_risk(3.0)
        assert 0.0 <= risk <= 0.2

    def test_strong_upside(self, calculator):
        """Test strong upside (+5% to +10%)."""
        risk = calculator.calculate_momentum_risk(7.5)
        assert 0.1 <= risk <= 0.3

    def test_parabolic_moderate(self, calculator):
        """Test moderate parabolic move (> +10%)."""
        risk = calculator.calculate_momentum_risk(12.0)
        assert 0.3 <= risk <= 0.5

    def test_parabolic_extreme(self, calculator):
        """Test extreme parabolic move."""
        risk = calculator.calculate_momentum_risk(20.0)
        assert risk == 0.5  # Capped

    def test_invalid_momentum_extreme_negative(self, calculator):
        """Test momentum < -100% raises error."""
        with pytest.raises(ValueError, match="cannot be less than -100%"):
            calculator.calculate_momentum_risk(-150.0)

    @pytest.mark.parametrize("price_change,expected_low,expected_high", [
        (-20.0, 0.68, 0.7),    # Panic extreme (capped)
        (-15.0, 0.68, 0.7),    # Panic cap boundary
        (-12.0, 0.58, 0.62),   # Panic moderate
        (-10.0, 0.48, 0.52),   # Strong down boundary
        (-7.5, 0.38, 0.42),    # Strong down mid
        (-5.0, 0.0, 0.02),     # Normal boundary (negative) - at -5%: 0.0
        (0.0, 0.08, 0.12),     # Normal mid
        (5.0, 0.18, 0.22),     # Normal boundary (positive)
        (7.5, 0.18, 0.22),     # Strong up mid
        (10.0, 0.28, 0.32),    # Strong up boundary
        (12.0, 0.38, 0.42),    # Parabolic moderate
        (15.0, 0.48, 0.5),     # Parabolic cap boundary
        (20.0, 0.48, 0.5),     # Parabolic extreme (capped)
    ])
    def test_momentum_boundaries(self, calculator, price_change, expected_low, expected_high):
        """Test momentum risk at various price changes."""
        risk = calculator.calculate_momentum_risk(price_change)
        assert expected_low <= risk <= expected_high


class TestAggregateRisks:
    """Test risk aggregation."""

    def test_aggregate_all_low(self, calculator):
        """Test aggregation with all low risks."""
        result = calculator.aggregate_risks(0.1, 0.1, 0.1, 0.1)

        assert result['valuation_risk'] == 0.1
        assert result['volatility_risk'] == 0.1
        assert result['breadth_risk'] == 0.1
        assert result['momentum_risk'] == 0.1
        assert result['average_risk'] == 0.1
        assert result['max_risk'] == 0.1
        assert result['min_risk'] == 0.1

    def test_aggregate_all_high(self, calculator):
        """Test aggregation with all high risks."""
        result = calculator.aggregate_risks(0.7, 0.7, 0.7, 0.7)

        assert result['average_risk'] == 0.7
        assert result['max_risk'] == 0.7
        assert result['min_risk'] == 0.7

    def test_aggregate_mixed(self, calculator):
        """Test aggregation with mixed risks."""
        result = calculator.aggregate_risks(0.2, 0.5, 0.3, 0.6)

        assert result['valuation_risk'] == 0.2
        assert result['volatility_risk'] == 0.5
        assert result['breadth_risk'] == 0.3
        assert result['momentum_risk'] == 0.6
        assert result['average_risk'] == 0.4  # (0.2 + 0.5 + 0.3 + 0.6) / 4
        assert result['max_risk'] == 0.6
        assert result['min_risk'] == 0.2

    def test_aggregate_extreme_variance(self, calculator):
        """Test aggregation with extreme variance."""
        result = calculator.aggregate_risks(0.0, 0.8, 0.1, 0.7)

        assert result['average_risk'] == 0.4  # (0.0 + 0.8 + 0.1 + 0.7) / 4
        assert result['max_risk'] == 0.8
        assert result['min_risk'] == 0.0

    def test_aggregate_invalid_valuation_low(self, calculator):
        """Test invalid valuation (< 0) raises error."""
        with pytest.raises(ValueError, match="valuation_risk must be in 0.0-0.8 range"):
            calculator.aggregate_risks(-0.1, 0.5, 0.3, 0.4)

    def test_aggregate_invalid_valuation_high(self, calculator):
        """Test invalid valuation (> 0.8) raises error."""
        with pytest.raises(ValueError, match="valuation_risk must be in 0.0-0.8 range"):
            calculator.aggregate_risks(0.9, 0.5, 0.3, 0.4)

    def test_aggregate_invalid_volatility(self, calculator):
        """Test invalid volatility raises error."""
        with pytest.raises(ValueError, match="volatility_risk must be in 0.0-0.8 range"):
            calculator.aggregate_risks(0.2, 1.0, 0.3, 0.4)

    def test_aggregate_invalid_breadth(self, calculator):
        """Test invalid breadth raises error."""
        with pytest.raises(ValueError, match="breadth_risk must be in 0.0-0.8 range"):
            calculator.aggregate_risks(0.2, 0.5, -0.5, 0.4)

    def test_aggregate_invalid_momentum(self, calculator):
        """Test invalid momentum raises error."""
        with pytest.raises(ValueError, match="momentum_risk must be in 0.0-0.8 range"):
            calculator.aggregate_risks(0.2, 0.5, 0.3, 0.85)

    @pytest.mark.parametrize("val,vol,breadth,mom,expected_avg", [
        (0.0, 0.0, 0.0, 0.0, 0.0),      # All zero
        (0.2, 0.2, 0.2, 0.2, 0.2),      # All low
        (0.4, 0.4, 0.4, 0.4, 0.4),      # All moderate
        (0.6, 0.6, 0.6, 0.6, 0.6),      # All elevated
        (0.8, 0.8, 0.8, 0.8, 0.8),      # All max
        (0.1, 0.3, 0.5, 0.7, 0.4),      # Linear progression
        (0.0, 0.4, 0.4, 0.0, 0.2),      # Balanced extremes
    ])
    def test_aggregate_average_calculation(self, calculator, val, vol, breadth, mom, expected_avg):
        """Test average calculation is correct."""
        result = calculator.aggregate_risks(val, vol, breadth, mom)
        assert abs(result['average_risk'] - expected_avg) < 0.001  # Floating point tolerance


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_normal_market_conditions(self, calculator, sample_percentiles):
        """Test normal market conditions (all low risk)."""
        # Normal conditions: P/E at 20, VIX at 15, breadth at 80%, momentum at +2%
        val_risk = calculator.calculate_valuation_risk(20.0, sample_percentiles)
        vol_risk = calculator.calculate_volatility_risk(15.0)
        breadth_risk = calculator.calculate_breadth_risk(80.0)
        mom_risk = calculator.calculate_momentum_risk(2.0)

        result = calculator.aggregate_risks(val_risk, vol_risk, breadth_risk, mom_risk)

        # All should be low
        assert result['average_risk'] < 0.3
        assert result['max_risk'] < 0.4

    def test_bubble_conditions(self, calculator, sample_percentiles):
        """Test bubble conditions (all high risk)."""
        # Bubble: P/E at 45, VIX at 12 (complacency), breadth at 95%, momentum at +12%
        val_risk = calculator.calculate_valuation_risk(45.0, sample_percentiles)
        vol_risk = calculator.calculate_volatility_risk(12.0)  # Low fear (risk)
        breadth_risk = calculator.calculate_breadth_risk(95.0)  # Low risk
        mom_risk = calculator.calculate_momentum_risk(12.0)  # Parabolic

        result = calculator.aggregate_risks(val_risk, vol_risk, breadth_risk, mom_risk)

        # Valuation and momentum should be elevated
        assert result['valuation_risk'] > 0.6  # High P/E
        assert result['momentum_risk'] > 0.3  # Parabolic

    def test_crash_conditions(self, calculator, sample_percentiles):
        """Test crash conditions (panic selling)."""
        # Crash: P/E at 15, VIX at 50, breadth at 10%, momentum at -15%
        val_risk = calculator.calculate_valuation_risk(15.0, sample_percentiles)
        vol_risk = calculator.calculate_volatility_risk(50.0)
        breadth_risk = calculator.calculate_breadth_risk(10.0)
        mom_risk = calculator.calculate_momentum_risk(-15.0)

        result = calculator.aggregate_risks(val_risk, vol_risk, breadth_risk, mom_risk)

        # Volatility, breadth, and momentum should be high
        assert result['volatility_risk'] >= 0.6  # Extreme VIX
        assert result['breadth_risk'] >= 0.6  # Low breadth
        assert result['momentum_risk'] >= 0.5  # Panic selling
        assert result['average_risk'] > 0.4  # Overall high risk

    def test_recovery_conditions(self, calculator, sample_percentiles):
        """Test recovery conditions (improving from lows)."""
        # Recovery: P/E at 22, VIX at 25, breadth at 55%, momentum at +6%
        val_risk = calculator.calculate_valuation_risk(22.0, sample_percentiles)
        vol_risk = calculator.calculate_volatility_risk(25.0)
        breadth_risk = calculator.calculate_breadth_risk(55.0)
        mom_risk = calculator.calculate_momentum_risk(6.0)

        result = calculator.aggregate_risks(val_risk, vol_risk, breadth_risk, mom_risk)

        # Should be low to moderate across the board
        # val_risk ~0.08, vol_risk ~0.3, breadth_risk ~0.35, mom_risk ~0.14
        assert 0.15 <= result['average_risk'] <= 0.3
        assert all(0.0 <= v <= 0.5 for v in [val_risk, vol_risk, breadth_risk, mom_risk])
