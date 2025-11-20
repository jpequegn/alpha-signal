"""Tests for ConfidenceAdjuster risk-based confidence adjustment.

Tests cover basic adjustment, signal enrichment, edge cases, and integration.
"""

import pytest

from src.daemon.confidence_adjuster import ConfidenceAdjuster


class TestConfidenceAdjusterInitialization:
    """Test ConfidenceAdjuster initialization."""

    def test_init_succeeds(self):
        """Test initialization works."""
        adjuster = ConfidenceAdjuster()
        assert adjuster is not None


class TestBasicAdjustment:
    """Test basic confidence adjustment formula."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    def test_no_bubble_no_adjustment(self, adjuster):
        """Test no adjustment when no bubble (probability=0.0)."""
        adjusted = adjuster.adjust(0.8, 0.0)
        assert adjusted == 0.8

    def test_full_bubble_zero_confidence(self, adjuster):
        """Test zero confidence when extreme bubble (probability=1.0)."""
        adjusted = adjuster.adjust(0.8, 1.0)
        assert adjusted == 0.0

    def test_moderate_bubble(self, adjuster):
        """Test moderate adjustment with bubble_prob=0.5."""
        adjusted = adjuster.adjust(0.8, 0.5)
        assert adjusted == 0.4

    def test_adjustment_formula_examples(self, adjuster):
        """Test documented examples from docstring."""
        # Normal market (bubble=0.2): 0.78 × 0.8 = 0.624
        result = adjuster.adjust(0.78, 0.2)
        assert 0.62 <= result <= 0.63

        # Moderate bubble (bubble=0.5): 0.78 × 0.5 = 0.39
        result = adjuster.adjust(0.78, 0.5)
        assert 0.38 <= result <= 0.40

        # Extreme bubble (bubble=0.8): 0.78 × 0.2 = 0.156
        result = adjuster.adjust(0.78, 0.8)
        assert 0.15 <= result <= 0.17

    def test_adjustment_always_nonnegative(self, adjuster):
        """Test adjustment never produces negative confidence."""
        for conf in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for bubble in [0.0, 0.2, 0.5, 0.8, 1.0]:
                result = adjuster.adjust(conf, bubble)
                assert result >= 0.0

    def test_adjustment_always_clamped(self, adjuster):
        """Test adjustment output is always in 0.0-1.0 range."""
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for bubble in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = adjuster.adjust(conf, bubble)
                assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("conf,bubble,expected_approx", [
        (0.0, 0.5, 0.0),        # Low confidence stays low
        (1.0, 0.0, 1.0),        # High confidence, no bubble
        (1.0, 0.5, 0.5),        # High confidence, moderate bubble
        (0.5, 0.5, 0.25),       # Medium × medium
        (0.9, 0.1, 0.81),       # Minimal reduction
    ])
    def test_parametric_adjustment(self, adjuster, conf, bubble, expected_approx):
        """Test adjustment with parametric inputs."""
        result = adjuster.adjust(conf, bubble)
        assert abs(result - expected_approx) < 0.01


class TestInputValidation:
    """Test input validation."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    def test_confidence_too_high(self, adjuster):
        """Test validation fails for confidence > 1.0."""
        with pytest.raises(ValueError, match="0.0-1.0"):
            adjuster.adjust(1.1, 0.5)

    def test_confidence_negative(self, adjuster):
        """Test validation fails for negative confidence."""
        with pytest.raises(ValueError, match="0.0-1.0"):
            adjuster.adjust(-0.1, 0.5)

    def test_bubble_too_high(self, adjuster):
        """Test validation fails for bubble probability > 1.0."""
        with pytest.raises(ValueError, match="0.0-1.0"):
            adjuster.adjust(0.8, 1.1)

    def test_bubble_negative(self, adjuster):
        """Test validation fails for negative bubble probability."""
        with pytest.raises(ValueError, match="0.0-1.0"):
            adjuster.adjust(0.8, -0.1)

    def test_confidence_not_numeric(self, adjuster):
        """Test validation fails for non-numeric confidence."""
        with pytest.raises(ValueError, match="numeric"):
            adjuster.adjust("high", 0.5)

    def test_bubble_not_numeric(self, adjuster):
        """Test validation fails for non-numeric bubble probability."""
        with pytest.raises(ValueError, match="numeric"):
            adjuster.adjust(0.8, "moderate")

    def test_both_invalid(self, adjuster):
        """Test validation catches first invalid input."""
        with pytest.raises(ValueError):
            adjuster.adjust(1.5, 1.5)


class TestSignalAdjustment:
    """Test signal enrichment and adjustment."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    @pytest.fixture
    def sample_signal(self):
        """Sample Phase 2 signal."""
        return {
            "signal": "BUY",
            "confidence": 0.75,
            "reasoning": "SMA uptrend, RSI bullish",
            "timestamp": "2025-01-15",
            "key_factors": ["SMA uptrend", "RSI at 65"],
        }

    def test_adjust_signal_normal_market(self, adjuster, sample_signal):
        """Test signal adjustment in normal market."""
        result = adjuster.adjust_signal(sample_signal, 0.2)

        # Check signal preserved
        assert result["signal"] == "BUY"
        assert result["timestamp"] == "2025-01-15"
        assert "key_factors" in result

        # Check confidence adjusted
        assert result["original_confidence"] == 0.75
        assert result["adjusted_confidence"] == pytest.approx(0.6)  # 0.75 × 0.8
        assert result["confidence"] == pytest.approx(0.6)  # Main confidence updated
        assert result["bubble_probability"] == 0.2

    def test_adjust_signal_bubble_market(self, adjuster, sample_signal):
        """Test signal adjustment in bubble market."""
        result = adjuster.adjust_signal(sample_signal, 0.7)

        assert result["adjusted_confidence"] == pytest.approx(0.225)  # 0.75 × 0.3
        assert result["original_confidence"] == 0.75

    def test_adjust_signal_adds_risk_reasoning(self, adjuster, sample_signal):
        """Test that risk-adjusted reasoning is added."""
        result = adjuster.adjust_signal(sample_signal, 0.5)

        assert "risk_adjusted_reasoning" in result
        assert "Signal adjusted for" in result["risk_adjusted_reasoning"]
        assert sample_signal["reasoning"] in result["risk_adjusted_reasoning"]

    def test_adjust_signal_no_original_reasoning(self, adjuster):
        """Test signal adjustment works without original reasoning."""
        signal = {
            "signal": "SELL",
            "confidence": 0.6,
            "key_factors": ["Volume declining"]
        }

        result = adjuster.adjust_signal(signal, 0.3)

        assert result["adjusted_confidence"] == pytest.approx(0.42)  # 0.6 × 0.7
        assert "risk_adjusted_reasoning" in result
        assert ("Signal adjusted for" in result["risk_adjusted_reasoning"] or
                "WARNING" in result["risk_adjusted_reasoning"])

    def test_adjust_signal_missing_confidence(self, adjuster):
        """Test validation fails if signal missing confidence."""
        signal = {
            "signal": "BUY",
            "reasoning": "Test"
        }

        with pytest.raises(KeyError, match="confidence"):
            adjuster.adjust_signal(signal, 0.5)

    def test_adjust_signal_missing_signal_field(self, adjuster):
        """Test validation fails if signal missing signal field."""
        signal = {
            "confidence": 0.75,
            "reasoning": "Test"
        }

        with pytest.raises(KeyError, match="signal"):
            adjuster.adjust_signal(signal, 0.5)

    def test_adjust_signal_not_dict(self, adjuster):
        """Test validation fails if signal not dict."""
        with pytest.raises(ValueError, match="dict"):
            adjuster.adjust_signal("not a signal", 0.5)

    def test_adjust_signal_preserves_fields(self, adjuster):
        """Test adjustment preserves all original signal fields."""
        signal = {
            "signal": "HOLD",
            "confidence": 0.5,
            "reasoning": "Mixed signals",
            "timestamp": "2025-01-14",
            "key_factors": ["Factor1", "Factor2"],
            "contradictions": ["Contradiction1"],
            "custom_field": "custom_value"
        }

        result = adjuster.adjust_signal(signal, 0.4)

        # All original fields preserved
        assert result["signal"] == "HOLD"
        assert result["timestamp"] == "2025-01-14"
        assert result["key_factors"] == ["Factor1", "Factor2"]
        assert result["contradictions"] == ["Contradiction1"]
        assert result["custom_field"] == "custom_value"


class TestAdjustmentFactors:
    """Test utility methods for understanding adjustments."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    def test_confidence_adjustment_factor_no_bubble(self, adjuster):
        """Test adjustment factor with no bubble."""
        factor = adjuster.get_confidence_adjustment_factor(0.0)
        assert factor == 1.0

    def test_confidence_adjustment_factor_full_bubble(self, adjuster):
        """Test adjustment factor with full bubble."""
        factor = adjuster.get_confidence_adjustment_factor(1.0)
        assert factor == 0.0

    def test_confidence_adjustment_factor_moderate(self, adjuster):
        """Test adjustment factor with moderate bubble."""
        factor = adjuster.get_confidence_adjustment_factor(0.3)
        assert factor == 0.7

    def test_confidence_reduction_no_change(self, adjuster):
        """Test reduction when no change."""
        reduction = adjuster.get_confidence_reduction(0.8, 0.8)
        assert reduction == 0.0

    def test_confidence_reduction_full_loss(self, adjuster):
        """Test reduction when confidence goes to zero."""
        reduction = adjuster.get_confidence_reduction(0.8, 0.0)
        assert reduction == 0.8

    def test_confidence_reduction_partial(self, adjuster):
        """Test reduction with partial adjustment."""
        reduction = adjuster.get_confidence_reduction(0.8, 0.5)
        assert reduction == pytest.approx(0.3)

    def test_confidence_reduction_never_negative(self, adjuster):
        """Test reduction never negative (adjusted >= original is invalid but handled)."""
        # This shouldn't happen in normal use, but test the function handles it
        reduction = adjuster.get_confidence_reduction(0.5, 0.6)
        assert reduction >= 0.0


class TestSignalReliability:
    """Test signal reliability assessment."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    def test_very_high_reliability(self, adjuster):
        """Test very high reliability (>= 0.8)."""
        assert adjuster.assess_signal_reliability(0.9) == "Very High"
        assert adjuster.assess_signal_reliability(0.8) == "Very High"

    def test_high_reliability(self, adjuster):
        """Test high reliability (0.6-0.8)."""
        assert adjuster.assess_signal_reliability(0.7) == "High"
        assert adjuster.assess_signal_reliability(0.6) == "High"

    def test_moderate_reliability(self, adjuster):
        """Test moderate reliability (0.4-0.6)."""
        assert adjuster.assess_signal_reliability(0.5) == "Moderate"
        assert adjuster.assess_signal_reliability(0.4) == "Moderate"

    def test_low_reliability(self, adjuster):
        """Test low reliability (0.2-0.4)."""
        assert adjuster.assess_signal_reliability(0.3) == "Low"
        assert adjuster.assess_signal_reliability(0.2) == "Low"

    def test_very_low_reliability(self, adjuster):
        """Test very low reliability (< 0.2)."""
        assert adjuster.assess_signal_reliability(0.1) == "Very Low"
        assert adjuster.assess_signal_reliability(0.0) == "Very Low"

    @pytest.mark.parametrize("confidence,expected", [
        (1.0, "Very High"),
        (0.85, "Very High"),
        (0.8, "Very High"),
        (0.75, "High"),
        (0.65, "High"),
        (0.6, "High"),
        (0.55, "Moderate"),
        (0.45, "Moderate"),
        (0.4, "Moderate"),
        (0.35, "Low"),
        (0.25, "Low"),
        (0.2, "Low"),
        (0.15, "Very Low"),
        (0.05, "Very Low"),
        (0.0, "Very Low"),
    ])
    def test_reliability_parametric(self, adjuster, confidence, expected):
        """Test reliability assessment with parametric inputs."""
        assert adjuster.assess_signal_reliability(confidence) == expected


class TestRiskReasoning:
    """Test risk-adjusted reasoning generation."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    def test_reasoning_includes_original(self, adjuster):
        """Test risk reasoning includes original reasoning."""
        result = adjuster._build_risk_reasoning("Test reasoning", 0.3, 0.7)
        assert "Test reasoning" in result
        assert "Signal adjusted for" in result

    def test_reasoning_normal_market(self, adjuster):
        """Test reasoning for normal market (low bubble)."""
        result = adjuster._build_risk_reasoning("Test", 0.1, 0.9)
        assert "healthy" in result.lower()

    def test_reasoning_bubble_market(self, adjuster):
        """Test reasoning for bubble market (high bubble)."""
        result = adjuster._build_risk_reasoning("Test", 0.8, 0.2)
        assert "extreme" in result.lower()

    def test_reasoning_moderate_market(self, adjuster):
        """Test reasoning for moderate market."""
        result = adjuster._build_risk_reasoning("Test", 0.5, 0.5)
        assert "moderate" in result.lower()

    def test_reasoning_no_original(self, adjuster):
        """Test reasoning works with no original reasoning."""
        result = adjuster._build_risk_reasoning("", 0.3, 0.7)
        assert len(result) > 0
        assert "Signal adjusted for" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    def test_zero_confidence_stays_zero(self, adjuster):
        """Test that zero confidence stays zero regardless of bubble."""
        for bubble in [0.0, 0.3, 0.7, 1.0]:
            result = adjuster.adjust(0.0, bubble)
            assert result == 0.0

    def test_floating_point_precision(self, adjuster):
        """Test handling of floating point arithmetic."""
        result = adjuster.adjust(0.333333, 0.666667)
        assert 0.0 <= result <= 1.0
        assert isinstance(result, float)

    def test_very_small_values(self, adjuster):
        """Test with very small confidence values."""
        result = adjuster.adjust(0.001, 0.5)
        assert result == 0.0005

    def test_very_small_bubble(self, adjuster):
        """Test with very small bubble probability."""
        result = adjuster.adjust(0.8, 0.001)
        assert 0.799 <= result <= 0.801


class TestIntegration:
    """Integration tests with other components."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    def test_integration_with_risk_factors(self, adjuster):
        """Test confidence adjustment with realistic risk factor workflow."""
        from src.daemon.risk_factors import RiskFactorCalculator

        calc = RiskFactorCalculator()

        # Get risk factors for bubble scenario
        val_risk = calc.calculate_valuation_risk(38.5, {"p25": 18, "p75": 28, "p90": 35})
        vol_risk = calc.calculate_volatility_risk(42.0)
        breadth_risk = calc.calculate_breadth_risk(85.0)
        mom_risk = calc.calculate_momentum_risk(12.0)

        # Create sample signal
        signal = {
            "signal": "BUY",
            "confidence": 0.85,
            "reasoning": "Strong uptrend with high breadth",
        }

        # In real system, bubble_scorer would synthesize these risk factors
        # For now, use simple average as proxy
        bubble_prob = (val_risk + vol_risk + breadth_risk + mom_risk) / 4.0 / 0.8

        # Adjust signal
        adjusted = adjuster.adjust_signal(signal, bubble_prob)

        assert 0.0 <= adjusted["adjusted_confidence"] <= 0.85
        assert adjusted["original_confidence"] == 0.85
        assert adjusted["bubble_probability"] == bubble_prob

    def test_full_phase3_workflow(self, adjuster):
        """Test complete Phase 3 workflow: signal → risk factors → adjustment."""
        # This represents a complete Phase 3 workflow

        # 1. Original signal from Phase 2
        signal = {
            "signal": "SELL",
            "confidence": 0.72,
            "reasoning": "Bearish divergence on daily RSI",
            "timestamp": "2025-01-15",
        }

        # 2. Simulate bubble probability (would come from BubbleScorer)
        bubble_probability = 0.35  # Moderate market

        # 3. Adjust signal
        adjusted_signal = adjuster.adjust_signal(signal, bubble_probability)

        # Verify workflow
        assert adjusted_signal["signal"] == "SELL"  # Signal unchanged
        assert adjusted_signal["original_confidence"] == 0.72
        assert adjusted_signal["adjusted_confidence"] == pytest.approx(0.468)  # 0.72 × 0.65
        assert adjusted_signal["bubble_probability"] == 0.35
        assert "risk_adjusted_reasoning" in adjusted_signal

        # Verify signal quality
        reliability = adjuster.assess_signal_reliability(adjusted_signal["adjusted_confidence"])
        assert reliability in ["Very High", "High", "Moderate", "Low", "Very Low"]


class TestConsistency:
    """Test consistency of adjustments."""

    @pytest.fixture
    def adjuster(self):
        """Create adjuster instance for tests."""
        return ConfidenceAdjuster()

    def test_same_inputs_same_output(self, adjuster):
        """Test deterministic output for same inputs."""
        result1 = adjuster.adjust(0.75, 0.3)
        result2 = adjuster.adjust(0.75, 0.3)
        assert result1 == result2

    def test_adjustment_order_independent(self, adjuster):
        """Test multiple adjustments give consistent results."""
        # Single adjustment
        single = adjuster.adjust(0.8, 0.5)

        # Create adjusted signal and re-adjust
        signal = {"signal": "BUY", "confidence": 0.8}
        adjusted1 = adjuster.adjust_signal(signal, 0.5)
        assert adjusted1["adjusted_confidence"] == single

    def test_monotonic_with_bubble(self, adjuster):
        """Test that higher bubble probability gives lower confidence."""
        conf = 0.8
        results = [adjuster.adjust(conf, b) for b in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]

        # Each result should be >= next result (monotonic decreasing)
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1]
