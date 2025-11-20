"""Bubble probability scorer using Claude LLM reasoning.

This module synthesizes 4 risk factors (0.0-0.8 scale) into a single
bubble probability (0.0-1.0) using Claude AI for nuanced assessment.
"""

import json
import os
import re
from typing import Any, Dict, Optional

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class BubbleScorer:
    """Synthesize risk factors into bubble probability using LLM reasoning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022"
    ):
        """Initialize bubble scorer with LLM client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use for scoring

        Raises:
            ValueError: If API key not provided and ANTHROPIC_API_KEY not set
            ImportError: If anthropic package not installed
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Provide via api_key parameter or "
                "ANTHROPIC_API_KEY environment variable."
            )

        self.model = model
        self.client = Anthropic(api_key=self.api_key)

    def score(
        self,
        valuation_risk: float,
        volatility_risk: float,
        breadth_risk: float,
        momentum_risk: float
    ) -> Dict[str, Any]:
        """Calculate bubble probability from risk factors using LLM reasoning.

        Args:
            valuation_risk: Valuation risk score (0.0-0.8)
            volatility_risk: Volatility risk score (0.0-0.8)
            breadth_risk: Market breadth risk score (0.0-0.8)
            momentum_risk: Momentum risk score (0.0-0.8)

        Returns:
            Dict containing:
                - bubble_probability: Overall bubble probability (0.0-1.0)
                - reasoning: LLM explanation of market regime
                - key_risk: Most concerning risk factor
                - fallback: True if heuristic used (only if LLM failed)
                - error: Error message (only if LLM failed)

        Raises:
            ValueError: If any risk factor not in 0.0-0.8 range
        """
        # Validate inputs
        self._validate_inputs(valuation_risk, volatility_risk, breadth_risk, momentum_risk)

        # Build LLM prompt
        prompt = self._build_prompt(valuation_risk, volatility_risk, breadth_risk, momentum_risk)

        # Try LLM scoring
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            result = self._parse_response(response_text)

            # Validate and clamp bubble_probability to 0.0-1.0
            bubble_prob = result.get("bubble_probability", 0.5)
            bubble_prob = max(0.0, min(1.0, float(bubble_prob)))

            return {
                "bubble_probability": bubble_prob,
                "reasoning": result.get("reasoning", ""),
                "key_risk": result.get("key_risk", "")
            }

        except Exception as e:
            # Fallback to heuristic scoring
            bubble_prob = self._score_from_heuristic(
                valuation_risk, volatility_risk, breadth_risk, momentum_risk
            )
            key_risk = self._identify_key_risk(
                valuation_risk, volatility_risk, breadth_risk, momentum_risk
            )

            return {
                "bubble_probability": bubble_prob,
                "reasoning": "LLM unavailable, using heuristic scoring",
                "key_risk": key_risk,
                "fallback": True,
                "error": str(e)
            }

    def _validate_inputs(
        self,
        valuation: float,
        volatility: float,
        breadth: float,
        momentum: float
    ) -> bool:
        """Validate that all risk factors are in 0.0-0.8 range.

        Args:
            valuation: Valuation risk score
            volatility: Volatility risk score
            breadth: Market breadth risk score
            momentum: Momentum risk score

        Returns:
            True if all inputs valid

        Raises:
            ValueError: If any risk factor not in 0.0-0.8 range
        """
        risk_names = {
            "Valuation risk": valuation,
            "Volatility risk": volatility,
            "Breadth risk": breadth,
            "Momentum risk": momentum
        }

        for name, risk in risk_names.items():
            if not isinstance(risk, (int, float)):
                raise ValueError(f"{name} must be a number, got {type(risk)}")
            if not (0.0 <= risk <= 0.8):
                raise ValueError(f"{name} must be in 0.0-0.8 range, got {risk}")

        return True

    def _build_prompt(
        self,
        valuation: float,
        volatility: float,
        breadth: float,
        momentum: float
    ) -> str:
        """Build LLM prompt for bubble probability assessment.

        Args:
            valuation: Valuation risk score (0.0-0.8)
            volatility: Volatility risk score (0.0-0.8)
            breadth: Market breadth risk score (0.0-0.8)
            momentum: Momentum risk score (0.0-0.8)

        Returns:
            Formatted prompt string
        """
        return f"""You are a market risk analyst. Given these risk factors (0.0=low risk, 0.8=high risk):
- Valuation Risk: {valuation:.2f}
- Volatility Risk: {volatility:.2f}
- Breadth Risk: {breadth:.2f}
- Momentum Risk: {momentum:.2f}

Synthesize into overall bubble probability (0.0-1.0):
0.0 = No bubble, market healthy
0.5 = Moderate bubble probability
1.0 = Extreme bubble, market extremes

Respond with JSON only:
{{
    "bubble_probability": 0.0-1.0,
    "reasoning": "Why is market in this regime?",
    "key_risk": "Most concerning factor"
}}"""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured data.

        Tries multiple parsing strategies:
        1. Direct JSON parsing
        2. Extract from markdown code blocks
        3. Regex pattern matching for JSON object

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed dict with bubble_probability, reasoning, key_risk

        Raises:
            ValueError: If no valid JSON found in response
        """
        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Regex pattern for JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # If all parsing fails
        raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

    def _score_from_heuristic(
        self,
        valuation: float,
        volatility: float,
        breadth: float,
        momentum: float
    ) -> float:
        """Calculate bubble probability using heuristic (fallback when LLM fails).

        Uses weighted linear combination of risk factors:
        - Valuation: 30% (most important for bubbles)
        - Volatility: 25% (fear/greed indicator)
        - Breadth: 25% (participation quality)
        - Momentum: 20% (rate of change)

        Args:
            valuation: Valuation risk score (0.0-0.8)
            volatility: Volatility risk score (0.0-0.8)
            breadth: Market breadth risk score (0.0-0.8)
            momentum: Momentum risk score (0.0-0.8)

        Returns:
            Bubble probability (0.0-1.0)
        """
        # Weights for each risk factor
        weights = {
            "valuation": 0.30,
            "volatility": 0.25,
            "breadth": 0.25,
            "momentum": 0.20
        }

        # Normalize risk factors from 0.0-0.8 to 0.0-1.0
        normalized = {
            "valuation": valuation / 0.8,
            "volatility": volatility / 0.8,
            "breadth": breadth / 0.8,
            "momentum": momentum / 0.8
        }

        # Weighted average
        bubble_prob = (
            normalized["valuation"] * weights["valuation"] +
            normalized["volatility"] * weights["volatility"] +
            normalized["breadth"] * weights["breadth"] +
            normalized["momentum"] * weights["momentum"]
        )

        # Ensure output is clamped to 0.0-1.0
        return max(0.0, min(1.0, bubble_prob))

    def _identify_key_risk(
        self,
        valuation: float,
        volatility: float,
        breadth: float,
        momentum: float
    ) -> str:
        """Identify the most concerning risk factor.

        Args:
            valuation: Valuation risk score (0.0-0.8)
            volatility: Volatility risk score (0.0-0.8)
            breadth: Market breadth risk score (0.0-0.8)
            momentum: Momentum risk score (0.0-0.8)

        Returns:
            Name of highest risk factor
        """
        risks = {
            "Valuation Risk": valuation,
            "Volatility Risk": volatility,
            "Breadth Risk": breadth,
            "Momentum Risk": momentum
        }

        # Return the risk with highest value
        return max(risks.items(), key=lambda x: x[1])[0]
