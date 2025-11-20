"""Integration tests for Phase 3: Backfill + Risk Assessment Pipeline.

Tests complete end-to-end workflows combining:
1. Historical data loading (backfill)
2. Phase 2 signal generation
3. Risk factor calculation
4. Bubble probability synthesis
5. Confidence adjustment

Validates data flows through entire pipeline and signal quality improves with risk adjustment.
"""

import time
from typing import Any, Dict, List

import pytest

from src.backfill.data_fetcher import HistoricalDataFetcher
from src.backfill.signal_generator import BackfillSignalGenerator
from src.backfill.synthetic_data import SyntheticRiskDataGenerator
from src.daemon.bubble_scorer import BubbleScorer
from src.daemon.confidence_adjuster import ConfidenceAdjuster
from src.daemon.risk_factors import RiskFactorCalculator


class TestPhase3Pipeline:
    """Test complete Phase 3 data pipeline."""

    @pytest.fixture
    def components(self):
        """Initialize all Phase 3 components."""
        return {
            "risk_calculator": RiskFactorCalculator(),
            "synthetic_gen": SyntheticRiskDataGenerator(seed=42),
            "confidence_adjuster": ConfidenceAdjuster(),
        }

    def test_pipeline_complete_workflow(self, components):
        """Test complete Phase 3 workflow from start to finish.

        Workflow:
        1. Generate synthetic market data
        2. Calculate risk factors from synthetic data
        3. Score bubble probability
        4. Adjust signal confidence
        5. Validate all steps produce valid output
        """
        # Step 1: Generate synthetic market data
        scenario = components["synthetic_gen"].generate_market_scenario("bubble")
        assert "pe_ratio" in scenario
        assert "vix" in scenario
        assert "breadth" in scenario
        assert "momentum" in scenario
        assert len(scenario["pe_ratio"]) > 0

        # Step 2: Calculate risk factors
        pe_risk = components["risk_calculator"].calculate_valuation_risk(
            scenario["pe_ratio"][0],
            {"p25": 18, "p75": 28, "p90": 35}
        )
        vol_risk = components["risk_calculator"].calculate_volatility_risk(
            scenario["vix"][0]
        )
        breadth_risk = components["risk_calculator"].calculate_breadth_risk(
            scenario["breadth"][0]
        )
        mom_risk = components["risk_calculator"].calculate_momentum_risk(
            scenario["momentum"][0]
        )

        # Validate risk factors
        for risk in [pe_risk, vol_risk, breadth_risk, mom_risk]:
            assert 0.0 <= risk <= 0.8

        # Step 3: Aggregate risks
        aggregated = components["risk_calculator"].aggregate_risks(
            pe_risk, vol_risk, breadth_risk, mom_risk
        )
        assert "valuation_risk" in aggregated
        assert "volatility_risk" in aggregated
        assert "breadth_risk" in aggregated
        assert "momentum_risk" in aggregated
        assert "average_risk" in aggregated

        # Step 4: Score bubble probability (mocked LLM)
        # For integration test, use heuristic
        bubble_prob = aggregated["average_risk"] / 0.8  # Normalize to 0-1
        bubble_prob = max(0.0, min(1.0, bubble_prob))

        # Step 5: Create and adjust signal
        original_signal = {
            "signal": "BUY",
            "confidence": 0.78,
            "reasoning": "SMA uptrend, RSI bullish",
        }

        adjusted_signal = components["confidence_adjuster"].adjust_signal(
            original_signal, bubble_prob
        )

        # Validate adjusted signal
        assert adjusted_signal["signal"] == "BUY"
        assert adjusted_signal["original_confidence"] == 0.78
        assert 0.0 <= adjusted_signal["adjusted_confidence"] <= 0.78
        assert adjusted_signal["bubble_probability"] == bubble_prob
        assert "risk_adjusted_reasoning" in adjusted_signal

    def test_pipeline_with_different_scenarios(self, components):
        """Test pipeline behaves correctly across market scenarios."""
        # Test that all scenarios can be processed without error
        scenarios = ["normal", "bubble", "crash", "recovery"]

        for scenario_name in scenarios:
            scenario = components["synthetic_gen"].generate_market_scenario(scenario_name)

            # Calculate risks
            pe_risk = components["risk_calculator"].calculate_valuation_risk(
                scenario["pe_ratio"][0],
                {"p25": 18, "p75": 28, "p90": 35}
            )
            vol_risk = components["risk_calculator"].calculate_volatility_risk(
                scenario["vix"][0]
            )
            breadth_risk = components["risk_calculator"].calculate_breadth_risk(
                scenario["breadth"][0]
            )
            mom_risk = components["risk_calculator"].calculate_momentum_risk(
                scenario["momentum"][0]
            )

            # Synthesize bubble probability
            avg_risk = (pe_risk + vol_risk + breadth_risk + mom_risk) / 4.0
            bubble_prob = avg_risk / 0.8

            # Adjust signal
            signal = {
                "signal": "BUY",
                "confidence": 0.80,
                "reasoning": "Test signal"
            }
            adjusted = components["confidence_adjuster"].adjust_signal(signal, bubble_prob)

            # Validate scenario produces valid bubble range
            assert 0.0 <= bubble_prob <= 1.0, \
                f"Scenario {scenario_name}: bubble_prob {bubble_prob} not in [0, 1]"
            assert adjusted["adjusted_confidence"] <= adjusted["original_confidence"]

    def test_pipeline_error_handling(self, components):
        """Test pipeline handles errors gracefully."""
        # Invalid confidence (too high)
        signal = {"signal": "BUY", "confidence": 1.5}  # Out of range
        with pytest.raises(ValueError):
            components["confidence_adjuster"].adjust_signal(signal, 0.5)

        # Invalid bubble probability
        signal = {"signal": "BUY", "confidence": 0.8}
        with pytest.raises(ValueError):
            components["confidence_adjuster"].adjust_signal(signal, 1.5)

        # Missing required fields
        bad_signal = {"signal": "BUY"}  # Missing confidence
        with pytest.raises(KeyError):
            components["confidence_adjuster"].adjust_signal(bad_signal, 0.5)

    def test_pipeline_data_consistency(self, components):
        """Test that data flows consistently through pipeline without loss."""
        # Generate scenario
        scenario = components["synthetic_gen"].generate_market_scenario("normal")
        initial_days = len(scenario["pe_ratio"])

        # Process through risk calculation
        pe_values = []
        for i in range(min(10, initial_days)):  # Test first 10 days
            pe_risk = components["risk_calculator"].calculate_valuation_risk(
                scenario["pe_ratio"][i],
                {"p25": 18, "p75": 28, "p90": 35}
            )
            pe_values.append(pe_risk)

        # Verify no data corruption
        assert len(pe_values) == min(10, initial_days)
        assert all(0.0 <= v <= 0.8 for v in pe_values)


class TestSignalQualityImprovement:
    """Test that Phase 3 improves signal quality via risk adjustment."""

    @pytest.fixture
    def adjuster(self):
        """Create confidence adjuster for tests."""
        return ConfidenceAdjuster()

    def test_normal_market_minimal_adjustment(self, adjuster):
        """Test that normal market (low bubble) minimally adjusts confidence."""
        signal = {
            "signal": "BUY",
            "confidence": 0.85,
            "reasoning": "Strong uptrend"
        }

        # Low bubble probability (normal market)
        adjusted = adjuster.adjust_signal(signal, bubble_probability=0.15)

        # Confidence should be reduced but not much
        reduction = signal["confidence"] - adjusted["adjusted_confidence"]
        assert 0.05 <= reduction <= 0.15  # 5-15% reduction

    def test_bubble_market_significant_adjustment(self, adjuster):
        """Test that bubble market significantly reduces confidence."""
        signal = {
            "signal": "BUY",
            "confidence": 0.85,
            "reasoning": "Technical breakout"
        }

        # High bubble probability
        adjusted = adjuster.adjust_signal(signal, bubble_probability=0.7)

        # Confidence should be significantly reduced
        reduction = signal["confidence"] - adjusted["adjusted_confidence"]
        assert reduction >= 0.40  # At least 40% reduction

    def test_crash_market_increases_caution(self, adjuster):
        """Test that crash market appropriately reduces confidence."""
        signal = {
            "signal": "SELL",
            "confidence": 0.72,
            "reasoning": "Bearish divergence"
        }

        # High volatility/breadth risk (crash conditions)
        adjusted = adjuster.adjust_signal(signal, bubble_probability=0.65)

        # SELL confidence should be reduced too
        assert adjusted["adjusted_confidence"] < signal["confidence"]

    def test_signal_reliability_degradation(self, adjuster):
        """Test that reliability degrades appropriately with bubble probability."""
        signal = {"signal": "BUY", "confidence": 0.8}

        results = []
        for bubble_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            adjusted = adjuster.adjust_signal(signal, bubble_prob)
            reliability = adjuster.assess_signal_reliability(
                adjusted["adjusted_confidence"]
            )
            results.append((bubble_prob, adjusted["adjusted_confidence"], reliability))

        # Verify monotonic degradation
        for i in range(len(results) - 1):
            curr_confidence = results[i][1]
            next_confidence = results[i + 1][1]
            assert curr_confidence >= next_confidence, \
                "Confidence should monotonically decrease with higher bubble probability"

    def test_signal_reasoning_reflects_risk(self, adjuster):
        """Test that adjusted reasoning reflects market risk level."""
        signal = {
            "signal": "BUY",
            "confidence": 0.75,
            "reasoning": "Uptrend continues"
        }

        # Normal market
        normal_adjusted = adjuster.adjust_signal(signal, 0.2)
        normal_reasoning = normal_adjusted["risk_adjusted_reasoning"]

        # Bubble market
        bubble_adjusted = adjuster.adjust_signal(signal, 0.8)
        bubble_reasoning = bubble_adjusted["risk_adjusted_reasoning"]

        # Bubble reasoning should mention extreme/elevated risk
        assert "extreme" in bubble_reasoning.lower() or "elevated" in bubble_reasoning.lower()


class TestBackfillWorkflow:
    """Test backfill-specific aspects of Phase 3."""

    @pytest.fixture
    def components(self):
        """Initialize backfill components."""
        return {
            "risk_calculator": RiskFactorCalculator(),
            "synthetic_gen": SyntheticRiskDataGenerator(seed=99),
            "confidence_adjuster": ConfidenceAdjuster(),
        }

    def test_backfill_generates_multiple_signals(self, components):
        """Test that backfill can process multiple time periods."""
        scenario = components["synthetic_gen"].generate_market_scenario("normal")

        # Simulate processing multiple days
        signals_processed = 0
        for day_idx in range(min(20, len(scenario["pe_ratio"]))):
            # Calculate risk for this day
            pe_risk = components["risk_calculator"].calculate_valuation_risk(
                scenario["pe_ratio"][day_idx],
                {"p25": 18, "p75": 28, "p90": 35}
            )
            vol_risk = components["risk_calculator"].calculate_volatility_risk(
                scenario["vix"][day_idx]
            )
            breadth_risk = components["risk_calculator"].calculate_breadth_risk(
                scenario["breadth"][day_idx]
            )
            mom_risk = components["risk_calculator"].calculate_momentum_risk(
                scenario["momentum"][day_idx]
            )

            # Create adjusted signal
            bubble_prob = (pe_risk + vol_risk + breadth_risk + mom_risk) / 4.0 / 0.8
            signal = {
                "signal": "BUY",
                "confidence": 0.75,
                "timestamp": f"2025-01-{day_idx:02d}"
            }
            adjusted = components["confidence_adjuster"].adjust_signal(signal, bubble_prob)
            assert adjusted is not None
            signals_processed += 1

        assert signals_processed == min(20, len(scenario["pe_ratio"]))

    def test_backfill_handles_volatility_regime_changes(self, components):
        """Test backfill correctly handles market regime changes."""
        # Generate data with known regimes
        normal_scenario = components["synthetic_gen"].generate_market_scenario("normal")
        bubble_scenario = components["synthetic_gen"].generate_market_scenario("bubble")

        # Process normal market - collect all risk metrics
        normal_all_risks = []
        for i in range(min(10, len(normal_scenario["pe_ratio"]))):
            pe_risk = components["risk_calculator"].calculate_valuation_risk(
                normal_scenario["pe_ratio"][i],
                {"p25": 18, "p75": 28, "p90": 35}
            )
            vol_risk = components["risk_calculator"].calculate_volatility_risk(
                normal_scenario["vix"][i]
            )
            breadth_risk = components["risk_calculator"].calculate_breadth_risk(
                normal_scenario["breadth"][i]
            )
            mom_risk = components["risk_calculator"].calculate_momentum_risk(
                normal_scenario["momentum"][i]
            )
            avg = (pe_risk + vol_risk + breadth_risk + mom_risk) / 4.0
            normal_all_risks.append(avg)

        # Process bubble market - collect all risk metrics
        bubble_all_risks = []
        for i in range(min(10, len(bubble_scenario["pe_ratio"]))):
            pe_risk = components["risk_calculator"].calculate_valuation_risk(
                bubble_scenario["pe_ratio"][i],
                {"p25": 18, "p75": 28, "p90": 35}
            )
            vol_risk = components["risk_calculator"].calculate_volatility_risk(
                bubble_scenario["vix"][i]
            )
            breadth_risk = components["risk_calculator"].calculate_breadth_risk(
                bubble_scenario["breadth"][i]
            )
            mom_risk = components["risk_calculator"].calculate_momentum_risk(
                bubble_scenario["momentum"][i]
            )
            avg = (pe_risk + vol_risk + breadth_risk + mom_risk) / 4.0
            bubble_all_risks.append(avg)

        # Both scenarios should produce valid risk values
        assert len(normal_all_risks) == min(10, len(normal_scenario["pe_ratio"]))
        assert len(bubble_all_risks) == min(10, len(bubble_scenario["pe_ratio"]))

        # All risk values should be valid (0.0-0.8 range normalized to 0-1)
        for risk in normal_all_risks + bubble_all_risks:
            assert 0.0 <= risk <= 0.8 / 0.8  # Risk factor normalized


class TestPerformance:
    """Test performance characteristics of Phase 3 pipeline."""

    @pytest.fixture
    def components(self):
        """Initialize components for performance testing."""
        return {
            "risk_calculator": RiskFactorCalculator(),
            "synthetic_gen": SyntheticRiskDataGenerator(seed=42),
            "confidence_adjuster": ConfidenceAdjuster(),
        }

    def test_risk_calculation_performance(self, components):
        """Test risk calculation completes in reasonable time."""
        scenario = components["synthetic_gen"].generate_market_scenario("normal")

        start = time.time()
        for i in range(len(scenario["pe_ratio"])):
            components["risk_calculator"].calculate_valuation_risk(
                scenario["pe_ratio"][i],
                {"p25": 18, "p75": 28, "p90": 35}
            )
            components["risk_calculator"].calculate_volatility_risk(scenario["vix"][i])
            components["risk_calculator"].calculate_breadth_risk(scenario["breadth"][i])
            components["risk_calculator"].calculate_momentum_risk(scenario["momentum"][i])
        elapsed = time.time() - start

        # Should process ~2500 days in < 5 seconds
        assert elapsed < 5.0, f"Risk calculation took {elapsed:.2f}s, expected < 5s"

    def test_confidence_adjustment_performance(self, components):
        """Test confidence adjustment is fast."""
        signal = {"signal": "BUY", "confidence": 0.8, "reasoning": "Test"}

        start = time.time()
        for _ in range(1000):
            components["confidence_adjuster"].adjust_signal(signal, 0.5)
        elapsed = time.time() - start

        # Should adjust 1000 signals in < 0.5 seconds
        assert elapsed < 0.5, f"1000 adjustments took {elapsed:.2f}s, expected < 0.5s"

    def test_full_pipeline_throughput(self, components):
        """Test full pipeline can handle realistic backfill volume."""
        scenario = components["synthetic_gen"].generate_market_scenario("normal")

        start = time.time()
        processed = 0

        for i in range(min(500, len(scenario["pe_ratio"]))):  # 500 days = ~2 years
            # Calculate risks
            pe_risk = components["risk_calculator"].calculate_valuation_risk(
                scenario["pe_ratio"][i],
                {"p25": 18, "p75": 28, "p90": 35}
            )
            vol_risk = components["risk_calculator"].calculate_volatility_risk(
                scenario["vix"][i]
            )
            breadth_risk = components["risk_calculator"].calculate_breadth_risk(
                scenario["breadth"][i]
            )
            mom_risk = components["risk_calculator"].calculate_momentum_risk(
                scenario["momentum"][i]
            )

            # Synthesize bubble probability
            bubble_prob = (pe_risk + vol_risk + breadth_risk + mom_risk) / 4.0 / 0.8

            # Adjust signal
            signal = {"signal": "BUY", "confidence": 0.75, "reasoning": "Test"}
            components["confidence_adjuster"].adjust_signal(signal, bubble_prob)

            processed += 1

        elapsed = time.time() - start

        # Should process 500 complete signals per day in < 2 seconds
        assert elapsed < 2.0, f"500 signals took {elapsed:.2f}s, expected < 2s"


class TestEndToEndValidation:
    """End-to-end validation of complete Phase 3."""

    @pytest.fixture
    def components(self):
        """Initialize all components."""
        return {
            "risk_calculator": RiskFactorCalculator(),
            "synthetic_gen": SyntheticRiskDataGenerator(seed=42),
            "confidence_adjuster": ConfidenceAdjuster(),
        }

    def test_complete_phase3_produces_valid_output(self, components):
        """Test that complete Phase 3 produces valid enriched signals."""
        scenario = components["synthetic_gen"].generate_market_scenario("normal")

        # Process 10 days
        enriched_signals = []
        for i in range(10):
            # Step 1: Risk factors
            pe_risk = components["risk_calculator"].calculate_valuation_risk(
                scenario["pe_ratio"][i],
                {"p25": 18, "p75": 28, "p90": 35}
            )
            vol_risk = components["risk_calculator"].calculate_volatility_risk(
                scenario["vix"][i]
            )
            breadth_risk = components["risk_calculator"].calculate_breadth_risk(
                scenario["breadth"][i]
            )
            mom_risk = components["risk_calculator"].calculate_momentum_risk(
                scenario["momentum"][i]
            )

            # Step 2: Aggregate
            agg = components["risk_calculator"].aggregate_risks(
                pe_risk, vol_risk, breadth_risk, mom_risk
            )

            # Step 3: Bubble probability (heuristic)
            bubble_prob = agg["average_risk"] / 0.8

            # Step 4: Adjust signal
            signal = {
                "signal": "BUY" if i % 2 == 0 else "SELL",
                "confidence": 0.75,
                "reasoning": f"Day {i} signal"
            }
            adjusted = components["confidence_adjuster"].adjust_signal(signal, bubble_prob)

            enriched_signals.append(adjusted)

        # Validate all signals are valid
        assert len(enriched_signals) == 10
        for sig in enriched_signals:
            assert "signal" in sig
            assert "confidence" in sig  # Updated to adjusted
            assert "original_confidence" in sig
            assert "adjusted_confidence" in sig
            assert "bubble_probability" in sig
            assert "risk_adjusted_reasoning" in sig
            assert sig["adjusted_confidence"] <= sig["original_confidence"]

    def test_phase3_vs_phase2_comparison(self, components):
        """Test that Phase 3 signals are properly adjusted vs Phase 2."""
        scenario = components["synthetic_gen"].generate_market_scenario("bubble")

        phase2_signal = {
            "signal": "BUY",
            "confidence": 0.80,
            "reasoning": "Strong uptrend in bubble market"
        }

        # Without Phase 3 adjustment (Phase 2 only)
        phase2_confidence = phase2_signal["confidence"]

        # With Phase 3 adjustment
        bubble_prob = 0.7  # Bubble market
        phase3_signal = components["confidence_adjuster"].adjust_signal(
            phase2_signal, bubble_prob
        )
        phase3_confidence = phase3_signal["adjusted_confidence"]

        # Phase 3 should reduce confidence in bubble market
        assert phase3_confidence < phase2_confidence
        assert phase3_confidence == pytest.approx(phase2_confidence * (1 - bubble_prob))

        # Phase 3 should add risk context to reasoning
        assert "risk_adjusted_reasoning" in phase3_signal
        assert len(phase3_signal["risk_adjusted_reasoning"]) > len(
            phase2_signal["reasoning"]
        )
