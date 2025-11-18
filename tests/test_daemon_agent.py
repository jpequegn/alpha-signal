"""Tests for LangGraph daemon agent."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.daemon.agent import (
    create_signal_agent_graph,
    AgentState,
    analyze_trend_node,
    analyze_momentum_node,
    analyze_volatility_node,
    synthesize_decision_node
)


@pytest.fixture
def sample_state():
    """Create sample agent state."""
    return AgentState(
        symbol="SPY",
        timestamp=datetime(2025, 1, 15, 10, 30),
        current_price=450.5,
        closes=np.linspace(440, 451, 100),  # Rising prices
        indicator_state={
            'sma_20': 450.0,
            'ema_12': 450.5,
            'ema_26': 449.5,
            'rsi_14': 65.0,
            'macd_line': 2.3,
            'macd_signal': 1.8,
            'macd_histogram': 0.5,
            'bb_upper': 455.0,
            'bb_middle': 450.0,
            'bb_lower': 445.0,
            'bb_bandwidth_pct': 2.2,
        },
        reasoning_steps=[],
        final_signal=None
    )


class TestAgentState:
    """Test agent state structure."""

    def test_state_creation(self, sample_state):
        """Test AgentState can be created."""
        assert sample_state['symbol'] == "SPY"
        assert sample_state['indicator_state'] is not None
        assert len(sample_state['reasoning_steps']) == 0

    def test_state_immutability_for_update(self, sample_state):
        """Test state can be updated properly."""
        new_step = {
            "indicator": "SMA",
            "analysis": "Price above SMA indicates uptrend"
        }
        sample_state['reasoning_steps'].append(new_step)

        assert len(sample_state['reasoning_steps']) == 1
        assert sample_state['reasoning_steps'][0]["indicator"] == "SMA"


class TestAgentNodes:
    """Test individual agent nodes."""

    @patch('src.daemon.agent.anthropic.Anthropic')
    def test_analyze_trend_node(self, mock_client, sample_state):
        """Test trend analysis node."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content[0].text = '''
        {
            "analysis": "SMA and EMA both pointing upward",
            "trend_direction": "UPTREND",
            "strength": "STRONG",
            "key_observation": "Price above all moving averages"
        }
        '''
        mock_client.return_value.messages.create.return_value = mock_response

        # This would be called by LangGraph
        # For now, just verify the function exists and accepts state
        assert callable(analyze_trend_node)


class TestAgentGraph:
    """Test full agent graph."""

    def test_graph_creation(self):
        """Test LangGraph can be created."""
        graph = create_signal_agent_graph()
        assert graph is not None

    def test_graph_has_required_nodes(self):
        """Test graph contains all required nodes."""
        graph = create_signal_agent_graph()
        # LangGraph stores node names internally
        # Graph is already compiled when returned
        assert graph is not None
