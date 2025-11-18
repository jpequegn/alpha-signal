"""Tests for daemon LLM prompts."""

import pytest
import json
from src.daemon.prompts import (
    TREND_ANALYSIS_PROMPT,
    MOMENTUM_ANALYSIS_PROMPT,
    VOLATILITY_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT,
    parse_llm_response
)


class TestPrompts:
    """Test prompt templates."""

    def test_trend_prompt_has_required_fields(self):
        """Test trend analysis prompt is valid."""
        assert "sma" in TREND_ANALYSIS_PROMPT.lower()
        assert "ema" in TREND_ANALYSIS_PROMPT.lower()
        assert "uptrend" in TREND_ANALYSIS_PROMPT.lower() or "trend" in TREND_ANALYSIS_PROMPT.lower()

    def test_momentum_prompt_has_required_fields(self):
        """Test momentum analysis prompt is valid."""
        assert "rsi" in MOMENTUM_ANALYSIS_PROMPT.lower()
        assert "macd" in MOMENTUM_ANALYSIS_PROMPT.lower()

    def test_volatility_prompt_has_required_fields(self):
        """Test volatility analysis prompt is valid."""
        assert "bollinger" in VOLATILITY_ANALYSIS_PROMPT.lower()
        assert "volatility" in VOLATILITY_ANALYSIS_PROMPT.lower()

    def test_synthesis_prompt_contains_instructions(self):
        """Test synthesis prompt instructs LLM properly."""
        prompt = SYNTHESIS_PROMPT.lower()
        assert "buy" in prompt or "sell" in prompt
        assert "confidence" in prompt

    def test_parse_llm_response_valid(self):
        """Test parsing valid LLM response."""
        response = """
        {
            "analysis": "SMA and EMA both pointing upward...",
            "strength": "strong"
        }
        """

        result = parse_llm_response(response)
        assert result is not None
        assert "analysis" in result
        assert result["analysis"] is not None

    def test_parse_llm_response_with_json_block(self):
        """Test parsing response with JSON code block."""
        response = """
        Here's the analysis:

        ```json
        {
            "analysis": "Clear uptrend confirmed",
            "strength": "strong"
        }
        ```

        This aligns with our indicators.
        """

        result = parse_llm_response(response)
        assert result is not None
        assert "analysis" in result
