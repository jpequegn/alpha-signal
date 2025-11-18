"""LangGraph daemon agent for signal generation."""

from typing import Dict, Any, TypedDict, Optional, List
from datetime import datetime
import numpy as np
import json

import anthropic
from langgraph.graph import StateGraph, END

from src.daemon.prompts import (
    format_trend_prompt,
    format_momentum_prompt,
    format_volatility_prompt,
    format_synthesis_prompt,
    parse_llm_response
)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """State passed through LangGraph nodes."""

    symbol: str
    timestamp: datetime
    current_price: float
    closes: np.ndarray  # Historical close prices
    indicator_state: Dict[str, float]  # Latest indicator values
    reasoning_steps: List[Dict[str, Any]]  # Accumulated reasoning
    final_signal: Optional[Dict[str, Any]]  # Final BUY/SELL/HOLD signal


# =============================================================================
# LLM CLIENT
# =============================================================================

def get_llm_client():
    """Get Anthropic LLM client."""
    return anthropic.Anthropic()


# =============================================================================
# REASONING NODES
# =============================================================================

def analyze_trend_node(state: AgentState) -> AgentState:
    """Analyze trend using SMA/EMA indicators.

    Args:
        state: Current agent state

    Returns:
        Updated state with trend analysis
    """
    client = get_llm_client()

    ind = state['indicator_state']
    prompt = format_trend_prompt(
        sma_20=ind['sma_20'],
        ema_12=ind['ema_12'],
        ema_26=ind['ema_26'],
        current_price=state['current_price']
    )

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text
    analysis_dict = parse_llm_response(response_text)

    if analysis_dict is None:
        analysis_dict = {"analysis": response_text, "error": "Could not parse JSON"}

    state['reasoning_steps'].append({
        "indicator_group": "TREND",
        "step_order": 1,
        "analysis": analysis_dict.get('analysis', response_text)
    })

    return state


def analyze_momentum_node(state: AgentState) -> AgentState:
    """Analyze momentum using RSI/MACD indicators.

    Args:
        state: Current agent state

    Returns:
        Updated state with momentum analysis
    """
    client = get_llm_client()

    ind = state['indicator_state']
    prompt = format_momentum_prompt(
        rsi_14=ind['rsi_14'],
        macd_line=ind['macd_line'],
        macd_signal=ind['macd_signal'],
        macd_histogram=ind['macd_histogram'],
        current_price=state['current_price']
    )

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text
    analysis_dict = parse_llm_response(response_text)

    if analysis_dict is None:
        analysis_dict = {"analysis": response_text, "error": "Could not parse JSON"}

    state['reasoning_steps'].append({
        "indicator_group": "MOMENTUM",
        "step_order": 2,
        "analysis": analysis_dict.get('analysis', response_text)
    })

    return state


def analyze_volatility_node(state: AgentState) -> AgentState:
    """Analyze volatility using Bollinger Bands.

    Args:
        state: Current agent state

    Returns:
        Updated state with volatility analysis
    """
    client = get_llm_client()

    ind = state['indicator_state']
    prompt = format_volatility_prompt(
        bb_upper=ind['bb_upper'],
        bb_middle=ind['bb_middle'],
        bb_lower=ind['bb_lower'],
        current_price=state['current_price'],
        bb_bandwidth_pct=ind['bb_bandwidth_pct']
    )

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text
    analysis_dict = parse_llm_response(response_text)

    if analysis_dict is None:
        analysis_dict = {"analysis": response_text, "error": "Could not parse JSON"}

    state['reasoning_steps'].append({
        "indicator_group": "VOLATILITY",
        "step_order": 3,
        "analysis": analysis_dict.get('analysis', response_text)
    })

    return state


def synthesize_decision_node(state: AgentState) -> AgentState:
    """Synthesize all analyses into final signal decision.

    Args:
        state: Current agent state with all reasoning

    Returns:
        Updated state with final_signal
    """
    client = get_llm_client()

    # Extract analyses from reasoning steps
    trend_analysis = state['reasoning_steps'][0].get('analysis', '')
    momentum_analysis = state['reasoning_steps'][1].get('analysis', '')
    volatility_analysis = state['reasoning_steps'][2].get('analysis', '')

    prompt = format_synthesis_prompt(
        trend_analysis=trend_analysis,
        momentum_analysis=momentum_analysis,
        volatility_analysis=volatility_analysis
    )

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=700,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text
    signal_dict = parse_llm_response(response_text)

    if signal_dict is None:
        signal_dict = {
            "signal": "HOLD",
            "confidence": 0.0,
            "key_factors": [],
            "contradictions": [],
            "final_reasoning": response_text
        }

    state['final_signal'] = {
        "signal": signal_dict.get('signal', 'HOLD'),
        "confidence": float(signal_dict.get('confidence', 0.5)),
        "key_factors": signal_dict.get('key_factors', []),
        "contradictions": signal_dict.get('contradictions', []),
        "final_reasoning": signal_dict.get('final_reasoning', ''),
        "timestamp": state['timestamp'],
        "symbol": state['symbol']
    }

    return state


# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_signal_agent_graph():
    """Create LangGraph state machine for signal generation.

    Returns:
        Compiled LangGraph graph ready for execution
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_trend", analyze_trend_node)
    workflow.add_node("analyze_momentum", analyze_momentum_node)
    workflow.add_node("analyze_volatility", analyze_volatility_node)
    workflow.add_node("synthesize_decision", synthesize_decision_node)

    # Define edges (execution order)
    workflow.set_entry_point("analyze_trend")
    workflow.add_edge("analyze_trend", "analyze_momentum")
    workflow.add_edge("analyze_momentum", "analyze_volatility")
    workflow.add_edge("analyze_volatility", "synthesize_decision")
    workflow.add_edge("synthesize_decision", END)

    # Compile graph
    return workflow.compile()


# =============================================================================
# EXECUTION
# =============================================================================

def run_signal_agent(state: AgentState) -> AgentState:
    """Run the signal generation agent.

    Args:
        state: Initial agent state with indicators

    Returns:
        Completed state with final_signal
    """
    graph = create_signal_agent_graph()
    result = graph.invoke(state)
    return result
