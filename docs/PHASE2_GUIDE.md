# Phase 2 Guide: LLM Daemon for Signal Generation

## Overview

Phase 2 builds on Phase 1's indicators to create an autonomous daemon that generates trading signals through LLM reasoning.

## Architecture

The daemon uses **LangGraph** to orchestrate multi-turn LLM reasoning:

1. **Data Layer** - Loads historical or real-time prices, calculates indicators
2. **LangGraph Agent** - State machine with 4 reasoning nodes
   - Trend Node: Analyzes SMA/EMA
   - Momentum Node: Analyzes RSI/MACD
   - Volatility Node: Analyzes Bollinger Bands
   - Synthesis Node: Generates final signal
3. **Database** - PostgreSQL stores signals with complete reasoning chain
4. **Runner** - Main loop that orchestrates everything

## Components

### Data Loader (`src/daemon/data_loader.py`)

Loads historical OHLCV data and calculates all Phase 1 indicators.

```python
from src.daemon.data_loader import HistoricalDataLoader

loader = HistoricalDataLoader()
df = loader.load_csv("data.csv")
indicators = loader.calculate_indicators(closes, symbol, timestamp)
```

### LLM Agent (`src/daemon/agent.py`)

LangGraph state machine that reasons about indicators.

```python
from src.daemon.agent import run_signal_agent, AgentState

state = AgentState(
    symbol="SPY",
    timestamp=datetime.now(),
    current_price=450.5,
    closes=prices,
    indicator_state={...},
    reasoning_steps=[],
    final_signal=None
)

result = run_signal_agent(state)
print(result['final_signal'])
# {'signal': 'BUY', 'confidence': 0.78, 'key_factors': [...], ...}
```

### Database Models (`src/daemon/models.py`)

SQLAlchemy models for signals and reasoning:

```python
from src.daemon.models import Signal, IndicatorSnapshot, ReasoningStep

signal = Signal(
    symbol='SPY',
    timestamp=datetime.now(),
    signal='BUY',
    confidence=0.78,
    key_factors=['SMA uptrend', 'RSI bullish'],
    final_reasoning='...'
)
```

### Daemon Runner (`src/daemon/runner.py`)

Orchestrates data loading, agent execution, and signal storage.

```python
from src.daemon.runner import DaemonRunner

runner = DaemonRunner(
    symbol="SPY",
    db_url="postgresql://...",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 6, 30)
)

signals_generated = runner.run(prices, dates)
```

## Understanding the Reasoning Flow

Each signal goes through 4 reasoning steps:

### 1. Trend Analysis
LLM analyzes SMA (20-day) and EMA (12/26-day) to determine trend direction.

**Question**: "Is price above/below moving averages? Are EMAs converging or diverging?"

**Output**:
```json
{
  "analysis": "SMA and EMA both pointing upward, price well above both...",
  "trend_direction": "UPTREND",
  "strength": "STRONG"
}
```

### 2. Momentum Analysis
LLM analyzes RSI (Relative Strength Index) and MACD to assess momentum.

**Question**: "Is RSI overbought/oversold? Is MACD histogram bullish/bearish?"

**Output**:
```json
{
  "analysis": "RSI at 65 (approaching overbought but not yet), MACD histogram positive...",
  "momentum_direction": "BULLISH",
  "rsi_status": "NORMAL"
}
```

### 3. Volatility Analysis
LLM analyzes Bollinger Bands to assess volatility and price positioning.

**Question**: "Is price near upper/lower bands? Is volatility high or low?"

**Output**:
```json
{
  "analysis": "Price near middle band, bands moderately wide indicating normal volatility...",
  "volatility_level": "NORMAL",
  "price_position": "NORMAL"
}
```

### 4. Signal Synthesis
LLM reviews all three analyses and generates final signal.

**Question**: "Given all these analyses, generate BUY/SELL/HOLD with confidence score."

**Output**:
```json
{
  "signal": "BUY",
  "confidence": 0.78,
  "key_factors": ["SMA uptrend", "RSI bullish", "MACD positive"],
  "contradictions": [],
  "final_reasoning": "Multiple indicators align on bullish thesis. Clear trend plus strong momentum."
}
```

## Learning Points

### 1. LangGraph State Machines
- Each node transforms the state (adds reasoning)
- State flows through graph edges
- Nodes can be composed into complex workflows
- Easy to debug - can inspect state at each node

### 2. LLM Prompt Engineering
- Specific prompts yield more consistent outputs
- Requesting JSON helps with parsing
- Breaking reasoning into steps makes LLM more reliable
- Examples in prompt help guide responses

### 3. Error Handling Patterns
- Graceful degradation (skip missing indicators, complete signal anyway)
- Retry logic for transient LLM failures
- Input validation before calling expensive operations
- Logging everything for debugging

### 4. Async Daemon Operations
- Background processing of data streams
- Non-blocking signal generation
- Database transactions ensure consistency

## Testing

### Unit Tests
Each component has unit tests:

```bash
pytest tests/test_daemon_data_loader.py -v
pytest tests/test_daemon_models.py -v
pytest tests/test_daemon_prompts.py -v
pytest tests/test_daemon_agent.py -v
pytest tests/test_daemon_runner.py -v
```

### Integration Tests
End-to-end test of full pipeline:

```bash
pytest tests/test_daemon_integration.py -v
```

## Running the Daemon

### Setup Database
```bash
# PostgreSQL
createdb alpha_signal

# Or use SQLite for testing
export DATABASE_URL="sqlite:///alpha_signal.db"
```

### Generate Signals from Historical Data
```python
from src.daemon.runner import DaemonRunner
from datetime import datetime
import pandas as pd

# Load historical data
df = pd.read_csv("data/spy_2024.csv")
prices = df['close'].values
dates = pd.to_datetime(df['date']).tolist()

# Run daemon
runner = DaemonRunner(
    symbol="SPY",
    db_url="postgresql://localhost/alpha_signal",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)

signals_generated = runner.run(prices, dates)
print(f"Generated {signals_generated} signals")
```

### Query Signals
```python
from src.daemon.models import Signal
from src.daemon.db import get_db_session

session = get_db_session()

# Latest signal
latest = session.query(Signal).filter_by(symbol='SPY')\
    .order_by(Signal.timestamp.desc()).first()

print(f"{latest.timestamp}: {latest.signal} (confidence: {latest.confidence})")

# High-confidence signals
high_conf = session.query(Signal).filter(
    Signal.confidence > 0.75
).all()

print(f"Found {len(high_conf)} high-confidence signals")
```

## Next: Phase 3

Phase 2 lays groundwork for Phase 3: **Multi-Factor Bubble Detection**

In Phase 3, we'll add risk assessment that reduces signal confidence during market extremes.

```python
# Phase 3 will add:
bubble_probability = bubble_detector.assess(market_data)
adjusted_confidence = signal_confidence * (1 - bubble_probability)
```

## Troubleshooting

### "API Rate Limited"
- Batch signals, add delays between LLM calls
- Use caching for repeated analyses

### "Out of Memory"
- Process data in chunks
- Use streaming for large datasets

### "Database Connection Failed"
- Verify PostgreSQL is running
- Check connection string (DATABASE_URL env var)

### "LLM Response Unparseable"
- Check prompts in `src/daemon/prompts.py`
- Add more structure to prompt (e.g., "Respond with ONLY valid JSON")
- Log raw response for debugging

## Success Metrics (Phase 2)

✅ Daemon runs without crashes on 6 months of data
✅ 200+ signals generated with proper timestamps
✅ Complete reasoning chain stored for each signal
✅ <200ms latency per signal (with real LLM calls)
✅ 80%+ test coverage
✅ All signals queryable from database
