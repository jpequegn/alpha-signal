"""Tests for BubbleScorer LLM-based risk assessment.

Tests cover input validation, LLM response parsing, heuristic fallback,
boundary conditions, and integration with risk factors.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.daemon.bubble_scorer import BubbleScorer


class TestBubbleScorerInitialization:
    """Test BubbleScorer initialization and validation."""

    def test_init_with_api_key_parameter(self):
        """Test initialization with explicit API key."""
        scorer = BubbleScorer(api_key="test-key-123")
        assert scorer.api_key == "test-key-123"
        assert scorer.model == "claude-3-5-sonnet-20241022"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        scorer = BubbleScorer(api_key="test-key", model="claude-opus-4-1-20250805")
        assert scorer.model == "claude-opus-4-1-20250805"

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key required"):
                BubbleScorer()

    def test_init_anthropic_not_available(self):
        """Test initialization fails if anthropic package not installed."""
        with patch("src.daemon.bubble_scorer.ANTHROPIC_AVAILABLE", False):
            with pytest.raises(ImportError, match="anthropic package not installed"):
                BubbleScorer(api_key="test-key")


class TestInputValidation:
    """Test input validation for risk factors."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance for tests."""
        return BubbleScorer(api_key="test-key")

    def test_valid_inputs(self, scorer):
        """Test validation passes for valid inputs."""
        assert scorer._validate_inputs(0.0, 0.0, 0.0, 0.0)
        assert scorer._validate_inputs(0.4, 0.3, 0.5, 0.2)
        assert scorer._validate_inputs(0.8, 0.8, 0.8, 0.8)

    def test_valuation_too_high(self, scorer):
        """Test validation fails for valuation > 0.8."""
        with pytest.raises(ValueError, match="Valuation risk"):
            scorer._validate_inputs(0.9, 0.3, 0.4, 0.2)

    def test_volatility_too_high(self, scorer):
        """Test validation fails for volatility > 0.8."""
        with pytest.raises(ValueError, match="Volatility risk"):
            scorer._validate_inputs(0.3, 0.85, 0.4, 0.2)

    def test_breadth_negative(self, scorer):
        """Test validation fails for negative breadth."""
        with pytest.raises(ValueError, match="Breadth risk"):
            scorer._validate_inputs(0.3, 0.3, -0.1, 0.2)

    def test_momentum_negative(self, scorer):
        """Test validation fails for negative momentum."""
        with pytest.raises(ValueError, match="Momentum risk"):
            scorer._validate_inputs(0.3, 0.3, 0.4, -0.05)

    def test_non_numeric_input(self, scorer):
        """Test validation fails for non-numeric input."""
        with pytest.raises(ValueError, match="must be a number"):
            scorer._validate_inputs("high", 0.3, 0.4, 0.2)


class TestPromptBuilding:
    """Test LLM prompt construction."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance for tests."""
        return BubbleScorer(api_key="test-key")

    def test_prompt_contains_all_factors(self, scorer):
        """Test prompt includes all 4 risk factors."""
        prompt = scorer._build_prompt(0.5, 0.6, 0.4, 0.3)
        assert "0.50" in prompt  # Valuation
        assert "0.60" in prompt  # Volatility
        assert "0.40" in prompt  # Breadth
        assert "0.30" in prompt  # Momentum

    def test_prompt_explains_scale(self, scorer):
        """Test prompt explains bubble probability scale."""
        prompt = scorer._build_prompt(0.1, 0.1, 0.1, 0.1)
        assert "0.0" in prompt and "No bubble" in prompt
        assert "0.5" in prompt and "Moderate" in prompt
        assert "1.0" in prompt and "Extreme" in prompt

    def test_prompt_requests_json(self, scorer):
        """Test prompt requests JSON response."""
        prompt = scorer._build_prompt(0.2, 0.3, 0.4, 0.1)
        assert "JSON" in prompt
        assert "bubble_probability" in prompt
        assert "reasoning" in prompt
        assert "key_risk" in prompt


class TestResponseParsing:
    """Test LLM response parsing strategies."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance for tests."""
        return BubbleScorer(api_key="test-key")

    def test_parse_direct_json(self, scorer):
        """Test parsing valid JSON directly."""
        response = '{"bubble_probability": 0.65, "reasoning": "High PE", "key_risk": "Valuation"}'
        result = scorer._parse_response(response)
        assert result["bubble_probability"] == 0.65
        assert result["reasoning"] == "High PE"
        assert result["key_risk"] == "Valuation"

    def test_parse_json_with_markdown(self, scorer):
        """Test parsing JSON from markdown code block."""
        response = """Here's the analysis:
```json
{"bubble_probability": 0.45, "reasoning": "Mixed signals", "key_risk": "Momentum"}
```
"""
        result = scorer._parse_response(response)
        assert result["bubble_probability"] == 0.45
        assert result["reasoning"] == "Mixed signals"

    def test_parse_json_without_markdown_label(self, scorer):
        """Test parsing JSON from code block without json label."""
        response = """
```
{"bubble_probability": 0.72, "reasoning": "Elevated risk", "key_risk": "Breadth"}
```
"""
        result = scorer._parse_response(response)
        assert result["bubble_probability"] == 0.72

    def test_parse_json_with_regex(self, scorer):
        """Test parsing JSON using regex when not in code block."""
        response = 'The market shows {"bubble_probability": 0.35, "reasoning": "Normal market", "key_risk": "Volatility"} conditions.'
        result = scorer._parse_response(response)
        assert result["bubble_probability"] == 0.35

    def test_parse_invalid_json(self, scorer):
        """Test parsing fails gracefully for unparseable response."""
        response = "This is not JSON at all"
        with pytest.raises(ValueError, match="Could not parse JSON"):
            scorer._parse_response(response)

    def test_parse_json_with_explanation(self, scorer):
        """Test parsing JSON when LLM includes explanation text."""
        response = """Based on the risk factors, here's my analysis:

The market shows:
```json
{"bubble_probability": 0.65, "reasoning": "Mixed signals with elevated valuation", "key_risk": "Valuation Risk"}
```

This suggests moderate bubble risk."""
        result = scorer._parse_response(response)
        assert result["bubble_probability"] == 0.65
        assert result["reasoning"] == "Mixed signals with elevated valuation"


class TestHeuristicScoring:
    """Test fallback heuristic scoring when LLM unavailable."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance for tests."""
        return BubbleScorer(api_key="test-key")

    def test_heuristic_all_zero(self, scorer):
        """Test heuristic with no risk factors."""
        prob = scorer._score_from_heuristic(0.0, 0.0, 0.0, 0.0)
        assert prob == 0.0

    def test_heuristic_all_max(self, scorer):
        """Test heuristic with maximum risk factors."""
        prob = scorer._score_from_heuristic(0.8, 0.8, 0.8, 0.8)
        assert prob == 1.0

    def test_heuristic_mixed_factors(self, scorer):
        """Test heuristic with mixed risk factors."""
        # Valuation 0.6 (75%), Volatility 0.4 (50%), Breadth 0.4 (50%), Momentum 0.2 (25%)
        # Expected: 0.75*0.30 + 0.50*0.25 + 0.50*0.25 + 0.25*0.20 = 0.225 + 0.125 + 0.125 + 0.05 = 0.525
        prob = scorer._score_from_heuristic(0.6, 0.4, 0.4, 0.2)
        assert 0.5 <= prob <= 0.55

    def test_heuristic_bounded(self, scorer):
        """Test heuristic output is always in 0.0-1.0 range."""
        for v in [0.0, 0.2, 0.4, 0.6, 0.8]:
            for vo in [0.0, 0.2, 0.4, 0.6, 0.8]:
                for b in [0.0, 0.2, 0.4, 0.6, 0.8]:
                    for m in [0.0, 0.2, 0.4, 0.6, 0.8]:
                        prob = scorer._score_from_heuristic(v, vo, b, m)
                        assert 0.0 <= prob <= 1.0

    def test_heuristic_weights_valuation_heaviest(self, scorer):
        """Test that valuation has highest weight in heuristic."""
        # High valuation only
        prob_val = scorer._score_from_heuristic(0.8, 0.0, 0.0, 0.0)
        # High volatility only
        prob_vol = scorer._score_from_heuristic(0.0, 0.8, 0.0, 0.0)
        # Valuation should contribute more
        assert prob_val > prob_vol


class TestKeyRiskIdentification:
    """Test identifying the most concerning risk factor."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance for tests."""
        return BubbleScorer(api_key="test-key")

    def test_valuation_highest(self, scorer):
        """Test identifies valuation as key risk when highest."""
        key = scorer._identify_key_risk(0.7, 0.3, 0.4, 0.2)
        assert key == "Valuation Risk"

    def test_volatility_highest(self, scorer):
        """Test identifies volatility as key risk when highest."""
        key = scorer._identify_key_risk(0.2, 0.8, 0.3, 0.1)
        assert key == "Volatility Risk"

    def test_breadth_highest(self, scorer):
        """Test identifies breadth as key risk when highest."""
        key = scorer._identify_key_risk(0.4, 0.3, 0.7, 0.2)
        assert key == "Breadth Risk"

    def test_momentum_highest(self, scorer):
        """Test identifies momentum as key risk when highest."""
        key = scorer._identify_key_risk(0.3, 0.2, 0.4, 0.8)
        assert key == "Momentum Risk"

    def test_tied_risks(self, scorer):
        """Test when multiple risks are equal, returns first max."""
        key = scorer._identify_key_risk(0.5, 0.5, 0.3, 0.2)
        assert key in ["Valuation Risk", "Volatility Risk"]


class TestLLMScoring:
    """Test LLM-based scoring with mocked responses."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance for tests."""
        return BubbleScorer(api_key="test-key")

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_score_llm_normal_market(self, mock_anthropic, scorer):
        """Test LLM scoring for normal market conditions."""
        # Mock LLM response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json.dumps({
            "bubble_probability": 0.2,
            "reasoning": "Normal market conditions",
            "key_risk": "Momentum Risk"
        })
        mock_client.messages.create.return_value = response

        scorer.client = mock_client

        result = scorer.score(0.2, 0.1, 0.15, 0.1)
        assert result["bubble_probability"] == 0.2
        assert result["reasoning"] == "Normal market conditions"
        assert result["key_risk"] == "Momentum Risk"
        assert "fallback" not in result

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_score_llm_bubble_market(self, mock_anthropic, scorer):
        """Test LLM scoring for bubble market conditions."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json.dumps({
            "bubble_probability": 0.85,
            "reasoning": "Extreme valuation, high momentum",
            "key_risk": "Valuation Risk"
        })
        mock_client.messages.create.return_value = response

        scorer.client = mock_client

        result = scorer.score(0.75, 0.3, 0.65, 0.5)
        assert result["bubble_probability"] == 0.85
        assert result["reasoning"] == "Extreme valuation, high momentum"

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_score_llm_clamps_probability(self, mock_anthropic, scorer):
        """Test that bubble probability is clamped to 0.0-1.0."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # LLM returns out-of-range value
        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json.dumps({
            "bubble_probability": 1.5,  # Out of range
            "reasoning": "Test",
            "key_risk": "Test Risk"
        })
        mock_client.messages.create.return_value = response

        scorer.client = mock_client

        result = scorer.score(0.4, 0.3, 0.4, 0.2)
        assert result["bubble_probability"] == 1.0  # Clamped to max

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_score_llm_fallback_on_error(self, mock_anthropic, scorer):
        """Test fallback to heuristic when LLM fails."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        scorer.client = mock_client

        result = scorer.score(0.6, 0.4, 0.5, 0.3)
        assert "fallback" in result and result["fallback"]
        assert "error" in result
        assert 0.0 <= result["bubble_probability"] <= 1.0
        assert result["reasoning"] == "LLM unavailable, using heuristic scoring"


class TestIntegration:
    """Integration tests with other components."""

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_integration_with_risk_factors(self, mock_anthropic):
        """Test scoring works with RiskFactorCalculator outputs."""
        from src.daemon.risk_factors import RiskFactorCalculator

        scorer = BubbleScorer(api_key="test-key")
        calc = RiskFactorCalculator()

        # Get risk factors for bubble scenario
        val_risk = calc.calculate_valuation_risk(38.5, {"p25": 18, "p75": 28, "p90": 35})
        vol_risk = calc.calculate_volatility_risk(42.0)
        breadth_risk = calc.calculate_breadth_risk(85.0)
        mom_risk = calc.calculate_momentum_risk(12.0)

        # Mock LLM for scoring
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json.dumps({
            "bubble_probability": 0.8,
            "reasoning": "Clear bubble signals",
            "key_risk": "Valuation Risk"
        })
        mock_client.messages.create.return_value = response
        scorer.client = mock_client

        result = scorer.score(val_risk, vol_risk, breadth_risk, mom_risk)
        assert result["bubble_probability"] == 0.8

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_integration_with_synthetic_data(self, mock_anthropic):
        """Test scoring works with SyntheticRiskDataGenerator outputs."""
        from src.backfill.synthetic_data import SyntheticRiskDataGenerator
        from src.daemon.risk_factors import RiskFactorCalculator

        scorer = BubbleScorer(api_key="test-key")
        gen = SyntheticRiskDataGenerator(seed=42)
        calc = RiskFactorCalculator()

        # Generate bubble scenario
        scenario = gen.generate_market_scenario("bubble")

        # Calculate risk factors from scenario
        val_risk = calc.calculate_valuation_risk(
            scenario["pe_ratio"][0],
            {"p25": 18, "p75": 28, "p90": 35}
        )
        vol_risk = calc.calculate_volatility_risk(scenario["vix"][0])
        breadth_risk = calc.calculate_breadth_risk(scenario["breadth"][0])
        mom_risk = calc.calculate_momentum_risk(scenario["momentum"][0])

        # Mock LLM
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json.dumps({
            "bubble_probability": 0.75,
            "reasoning": "Synthetic bubble scenario",
            "key_risk": "Valuation Risk"
        })
        mock_client.messages.create.return_value = response
        scorer.client = mock_client

        result = scorer.score(val_risk, vol_risk, breadth_risk, mom_risk)
        assert 0.5 <= result["bubble_probability"] <= 1.0


class TestBoundaryScenarios:
    """Test extreme market scenarios."""

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_scenario_normal_market(self, mock_anthropic):
        """Test normal market (all low risk)."""
        scorer = BubbleScorer(api_key="test-key")
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json.dumps({
            "bubble_probability": 0.15,
            "reasoning": "Healthy market",
            "key_risk": "None"
        })
        mock_client.messages.create.return_value = response
        scorer.client = mock_client

        result = scorer.score(0.15, 0.1, 0.2, 0.1)
        assert 0.0 <= result["bubble_probability"] <= 0.3

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_scenario_extreme_bubble(self, mock_anthropic):
        """Test extreme bubble (all high risk)."""
        scorer = BubbleScorer(api_key="test-key")
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json.dumps({
            "bubble_probability": 0.95,
            "reasoning": "Extreme bubble",
            "key_risk": "All factors"
        })
        mock_client.messages.create.return_value = response
        scorer.client = mock_client

        result = scorer.score(0.8, 0.8, 0.8, 0.8)
        assert 0.7 <= result["bubble_probability"] <= 1.0

    @patch("src.daemon.bubble_scorer.Anthropic")
    def test_scenario_crash_market(self, mock_anthropic):
        """Test market crash (high volatility/breadth, low valuation)."""
        scorer = BubbleScorer(api_key="test-key")
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        response = MagicMock()
        response.content = [MagicMock()]
        response.content[0].text = json.dumps({
            "bubble_probability": 0.7,
            "reasoning": "Market crisis",
            "key_risk": "Volatility Risk"
        })
        mock_client.messages.create.return_value = response
        scorer.client = mock_client

        result = scorer.score(0.2, 0.75, 0.7, 0.6)  # High fear, low valuation
        assert 0.5 <= result["bubble_probability"] <= 1.0


class TestEdgeCases:
    """Test edge cases and corner cases."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance for tests."""
        return BubbleScorer(api_key="test-key")

    def test_zero_risk_factors(self, scorer):
        """Test with all zero risk factors."""
        prob = scorer._score_from_heuristic(0.0, 0.0, 0.0, 0.0)
        assert prob == 0.0

    def test_max_risk_factors(self, scorer):
        """Test with all maximum risk factors."""
        prob = scorer._score_from_heuristic(0.8, 0.8, 0.8, 0.8)
        assert prob == 1.0

    def test_single_high_factor(self, scorer):
        """Test with only one high risk factor."""
        prob = scorer._score_from_heuristic(0.8, 0.0, 0.0, 0.0)
        # 0.8/0.8 * 0.30 = 0.30
        assert 0.25 <= prob <= 0.35

    def test_floating_point_precision(self, scorer):
        """Test handling of floating point precision."""
        result = scorer._score_from_heuristic(0.333333, 0.666667, 0.111111, 0.555555)
        assert 0.0 <= result <= 1.0
        assert isinstance(result, float)
