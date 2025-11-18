"""LLM prompt templates for daemon reasoning nodes."""

import json
import re
from typing import Dict, Optional

# =============================================================================
# TREND ANALYSIS PROMPT
# =============================================================================

TREND_ANALYSIS_PROMPT = """You are a trading analyst. Analyze the trend indicators and provide a concise assessment.

Given:
- SMA (20-day): {sma_20}
- EMA (12-day): {ema_12}
- EMA (26-day): {ema_26}
- Current Price: {current_price}

Provide a brief JSON response analyzing the trend:
{{
    "analysis": "Is price above/below moving averages? Are EMAs converging or diverging? What is the overall trend direction?",
    "trend_direction": "UPTREND|DOWNTREND|SIDEWAYS",
    "strength": "STRONG|MODERATE|WEAK",
    "key_observation": "One specific observation about the trend"
}}

Remember: Be specific and concise. Focus on what the indicators actually show, not speculation."""


# =============================================================================
# MOMENTUM ANALYSIS PROMPT
# =============================================================================

MOMENTUM_ANALYSIS_PROMPT = """You are a trading analyst. Analyze momentum indicators and provide an assessment.

Given:
- RSI (14-period): {rsi_14}
- MACD Line: {macd_line}
- MACD Signal Line: {macd_signal}
- MACD Histogram: {macd_histogram}
- Current Price: {current_price}

Provide a brief JSON response analyzing momentum:
{{
    "analysis": "Is RSI overbought (>70) or oversold (<30)? Is MACD histogram positive (bullish) or negative (bearish)? What does the momentum tell us?",
    "momentum_direction": "BULLISH|BEARISH|NEUTRAL",
    "rsi_status": "OVERBOUGHT|NORMAL|OVERSOLD",
    "macd_status": "BULLISH|BEARISH|NEUTRAL",
    "key_observation": "One specific observation about momentum"
}}

Remember: RSI above 70 is overbought, below 30 is oversold. MACD histogram positive = bullish."""


# =============================================================================
# VOLATILITY ANALYSIS PROMPT
# =============================================================================

VOLATILITY_ANALYSIS_PROMPT = """You are a trading analyst. Analyze volatility indicators.

Given:
- Bollinger Bands Upper: {bb_upper}
- Bollinger Bands Middle (SMA): {bb_middle}
- Bollinger Bands Lower: {bb_lower}
- Current Price: {current_price}
- Bandwidth %: {bb_bandwidth_pct}

Provide a brief JSON response analyzing volatility:
{{
    "analysis": "Is price near upper band (overbought from volatility perspective)? Near lower band (oversold)? Is volatility high (wide bands) or low (tight bands/squeeze)?",
    "volatility_level": "HIGH|NORMAL|LOW",
    "price_position": "NEAR_UPPER|NORMAL|NEAR_LOWER",
    "key_observation": "One specific observation about volatility"
}}

Remember: Wide bands = high volatility (expansion). Tight bands = low volatility (squeeze)."""


# =============================================================================
# SYNTHESIS PROMPT
# =============================================================================

SYNTHESIS_PROMPT = """You are a trading decision maker. Review all indicator analyses and generate a final trading signal.

Analyses received:
- Trend Analysis: {trend_analysis}
- Momentum Analysis: {momentum_analysis}
- Volatility Analysis: {volatility_analysis}

Generate a JSON response with your final decision:
{{
    "signal": "BUY|SELL|HOLD",
    "confidence": 0.0-1.0,
    "key_factors": ["factor1", "factor2", "factor3"],
    "contradictions": ["contradiction1", "contradiction2"] or [],
    "final_reasoning": "A 2-3 sentence explanation of your decision. Why BUY/SELL/HOLD? What indicators convinced you?"
}}

Rules:
- BUY: Multiple indicators aligned bullish, momentum positive, trend upward
- SELL: Multiple indicators aligned bearish, momentum negative, trend downward
- HOLD: Conflicting signals, neutral momentum, unclear trend
- Confidence: 0.0 (very uncertain) to 1.0 (very certain). Usually 0.4-0.8 range.
- Key factors: List 2-3 strongest reasons for your decision
- Contradictions: Note any conflicting indicators

Be decisive but honest about uncertainty."""


# =============================================================================
# UTILITIES
# =============================================================================

def parse_llm_response(response: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    # Try to extract JSON from code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        response = json_match.group(1)

    # Try to find standalone JSON object
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        response = json_match.group(0)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def format_trend_prompt(sma_20: float, ema_12: float, ema_26: float, current_price: float) -> str:
    """Format trend analysis prompt with current values."""
    return TREND_ANALYSIS_PROMPT.format(
        sma_20=sma_20,
        ema_12=ema_12,
        ema_26=ema_26,
        current_price=current_price
    )


def format_momentum_prompt(rsi_14: float, macd_line: float, macd_signal: float, macd_histogram: float, current_price: float) -> str:
    """Format momentum analysis prompt."""
    return MOMENTUM_ANALYSIS_PROMPT.format(
        rsi_14=rsi_14,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        current_price=current_price
    )


def format_volatility_prompt(bb_upper: float, bb_middle: float, bb_lower: float, current_price: float, bb_bandwidth_pct: float) -> str:
    """Format volatility analysis prompt."""
    return VOLATILITY_ANALYSIS_PROMPT.format(
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        current_price=current_price,
        bb_bandwidth_pct=bb_bandwidth_pct
    )


def format_synthesis_prompt(trend_analysis: str, momentum_analysis: str, volatility_analysis: str) -> str:
    """Format synthesis prompt with analyses."""
    return SYNTHESIS_PROMPT.format(
        trend_analysis=trend_analysis,
        momentum_analysis=momentum_analysis,
        volatility_analysis=volatility_analysis
    )
