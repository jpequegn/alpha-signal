# Phase 2 Design: LLM Daemon for Signal Generation

**Date**: 2025-11-18
**Status**: Design Complete - Ready for Implementation
**Learning Focus**: Maximum understanding of LangGraph, LLM multi-turn reasoning, async daemon patterns

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Phase 2 Daemon                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Data Source              Indicator Processing              │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │ Historical CSV   │────→│ Indicator Calc   │             │
│  │ (Phase 2)        │     │ (SMA,EMA,RSI...) │             │
│  └──────────────────┘     └────────┬─────────┘             │
│                                    │                        │
│  Alpaca API                        ▼                        │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │ Real-time Quotes │────→│ LangGraph Agent  │             │
│  │ (Phase 5)        │     │ Multi-turn LLM   │             │
│  └──────────────────┘     └────────┬─────────┘             │
│                                    │                        │
│                                    ▼                        │
│                         ┌──────────────────┐               │
│                         │ PostgreSQL       │               │
│                         │ - Signals        │               │
│                         │ - Metrics        │               │
│                         │ - Reasoning      │               │
│                         └──────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decisions**:
- **Data abstraction**: Single interface for both historical (Phase 2) and real-time (Phase 5) data
- **Async-first**: Use `asyncio` for daemon operations and future real-time handling
- **LangGraph core**: Use LangGraph's state machine for multi-turn reasoning
- **Separate concerns**: Data fetching, indicator calculation, reasoning, and storage are distinct modules

**Why This Structure**:
- Phase 2 learns daemon patterns with historical data
- Phase 5 swaps data source, keeps everything else unchanged
- Multi-turn reasoning visible in LangGraph state transitions
- Testing is straightforward - mock data source, test reasoning

---

## LangGraph Agent - Multi-Turn Reasoning Flow

The daemon's core is a **LangGraph state machine** that reasons through indicators step-by-step:

```
START
  │
  ▼
┌─────────────────────────────┐
│ Initialize State            │
│ - symbol, prices, timestamp │
│ - empty reasoning list      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Node: Analyze Trend         │
│ LLM examines SMA/EMA        │
│ stores: "SMA uptrend..."    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Node: Analyze Momentum      │
│ LLM examines RSI/MACD       │
│ stores: "RSI at 65..."      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Node: Analyze Volatility    │
│ LLM examines Bollinger Bands│
│ stores: "Bands wide..."     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Node: Synthesize Decision   │
│ LLM reviews all findings    │
│ generates: signal, conf,    │
│ key_factors, reasoning      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│ Node: Store & Return        │
│ Save to PostgreSQL          │
│ Return final signal         │
└────────────┬────────────────┘
             │
             ▼
           END
```

**State Structure** (passed between nodes):
```python
{
  "symbol": "SPY",
  "timestamp": "2025-01-15T10:30:00Z",
  "prices": np.ndarray,           # OHLCV data
  "indicator_state": {
    "sma_20": 450.5,
    "ema_12": 451.2,
    "rsi_14": 65,
    "macd": {...},
    "bollinger_bands": {...}
  },
  "reasoning_steps": [
    {"indicator": "SMA", "analysis": "..."},
    # filled by each node
  ],
  "final_signal": {...}            # filled by synthesis node
}
```

**LLM Prompts** (each node has targeted prompt):
- Trend node: "Analyze these moving averages and describe the trend..."
- Momentum node: "Given RSI and MACD, assess momentum..."
- Volatility node: "What does Bollinger Bands tell you about volatility?"
- Synthesis node: "Review all findings and generate final BUY/SELL/HOLD signal..."

**Why LangGraph for this**:
- Forces you to think in state machines (fundamental agent pattern)
- Built-in state threading (each node sees previous results)
- Easy to debug (can inspect state at each step)
- Scales to complex reasoning without spaghetti callbacks

---

## Database Schema & Storage

PostgreSQL will store all signals with full reasoning context:

```sql
-- Core signals table
CREATE TABLE signals (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  signal VARCHAR(10) NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
  confidence FLOAT NOT NULL,     -- 0.0-1.0
  key_factors TEXT[],            -- ['SMA uptrend', 'RSI bullish', ...]
  contradictions TEXT[],         -- ['Price below EMA', ...]
  final_reasoning TEXT,          -- LLM's synthesis paragraph
  created_at TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(symbol, timestamp)      -- One signal per symbol per time
);

-- Indicator state snapshot at signal time
CREATE TABLE indicator_snapshots (
  id SERIAL PRIMARY KEY,
  signal_id INTEGER REFERENCES signals(id),
  sma_20 FLOAT,
  ema_12 FLOAT,
  ema_26 FLOAT,
  rsi_14 FLOAT,
  macd_line FLOAT,
  macd_signal FLOAT,
  macd_histogram FLOAT,
  bb_upper FLOAT,
  bb_middle FLOAT,
  bb_lower FLOAT,
  bb_bandwidth_pct FLOAT
);

-- Reasoning steps for auditability
CREATE TABLE reasoning_steps (
  id SERIAL PRIMARY KEY,
  signal_id INTEGER REFERENCES signals(id),
  step_order INTEGER,           -- 1=trend, 2=momentum, 3=volatility
  indicator_group VARCHAR(50),  -- 'TREND', 'MOMENTUM', 'VOLATILITY'
  analysis TEXT,                -- LLM's per-indicator reasoning
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Why This Design**:
- `signals` table: Query recent signals, by symbol, by confidence
- `indicator_snapshots`: Reproduce decisions (what indicators were at signal time?)
- `reasoning_steps`: Full auditability (why did it decide BUY?)
- Timestamp + symbol uniqueness: Prevents duplicate signals

**Queries You'll Write**:
```python
# Latest signal for a symbol
SELECT * FROM signals WHERE symbol='SPY' ORDER BY timestamp DESC LIMIT 1;

# High-confidence signals from past week
SELECT * FROM signals WHERE confidence > 0.75 AND timestamp > NOW() - INTERVAL '7 days';

# Debug: Why did it buy on this date?
SELECT s.*, rs.analysis FROM signals s
  JOIN reasoning_steps rs ON s.id = rs.signal_id
  WHERE s.symbol='SPY' AND s.timestamp='2025-01-15'
  ORDER BY rs.step_order;
```

---

## Error Handling & Resilience

Error handling teaches you production patterns:

```python
# LangGraph handles errors at node level
class DaemonNodes:

  def analyze_trend(state):
    """Analyze SMA/EMA indicators"""
    try:
      prices = state["prices"]
      sma = SMA(period=20).calculate(prices)
      ema = EMA(period=12).calculate(prices)

      # Check data quality before LLM
      if np.all(np.isnan(sma)) or np.all(np.isnan(ema)):
        raise DataQualityError("Insufficient data for trend analysis")

      # LLM analyzes trends
      response = llm.invoke({
        "sma_values": sma[-10:].tolist(),
        "ema_values": ema[-10:].tolist(),
        "prompt": "Describe the trend..."
      })

      state["reasoning_steps"].append({
        "indicator": "SMA/EMA",
        "analysis": response
      })
      return state

    except DataQualityError as e:
      # Graceful degradation: mark as skipped, continue
      state["reasoning_steps"].append({
        "indicator": "SMA/EMA",
        "analysis": f"Skipped: {e}",
        "valid": False
      })
      return state

    except LLMError as e:
      # Retry logic - LLM timeouts are common
      logger.warning(f"LLM timeout: {e}, retrying...")
      raise  # Let LangGraph retry

# LangGraph graph configuration
graph.add_error_handler(
  node_id="analyze_trend",
  handler=retry_with_backoff,  # exponential backoff
  max_retries=3
)
```

**Key Patterns You'll Learn**:
- Input validation (check data quality before calling LLM)
- Graceful degradation (skip missing indicators, complete signal anyway)
- Structured error recovery (retry transient failures, skip permanent ones)
- Observability (log all failures with context)

**Testing Strategy**:
- Unit tests: Each node with mocked LLM (deterministic)
- Integration tests: Full graph with test data (known outputs)
- Chaos tests: Missing data, LLM errors, DB failures
- Instrumentation: Every error logged with full state for debugging

---

## Output Structure

The final signal output balances transparency with simplicity:

```json
{
  "signal": "BUY",
  "confidence": 0.78,
  "key_factors": [
    "SMA uptrend confirmed",
    "RSI at 65 (bullish)",
    "MACD histogram positive"
  ],
  "contradictions": [],
  "final_reasoning": "Multiple indicators align on bullish thesis. SMA shows clear uptrend, RSI confirms momentum without overbought condition, MACD histogram remains positive. Volume could be stronger but momentum is solid.",
  "timestamp": "2025-01-15T10:30:00Z",
  "indicator_state": {
    "sma_20": 450.5,
    "ema_12": 451.2,
    "rsi_14": 65,
    "macd": {...}
  }
}
```

**Why This Structure**:
- **signal**: Clear decision (BUY/SELL/HOLD)
- **confidence**: 0-1 score for signal strength
- **key_factors**: Key bullish/bearish factors
- **contradictions**: Factors that don't align (for learning)
- **final_reasoning**: LLM synthesis paragraph
- **indicator_state**: Full snapshot for reproducibility

---

## Implementation Tasks (Phase 2)

**Task 2.1: Data Layer & Indicator Calculator**
- Create `src/daemon/data_loader.py` - Load historical CSV, calculate indicators
- Extend Phase 1 indicators to batch mode (multiple symbols/dates)
- Unit tests for data loading and calculation accuracy

**Task 2.2: Database Schema & ORM**
- Set up PostgreSQL locally (or Docker)
- Create migration files (alembic) for schema
- Write `src/daemon/models.py` - SQLAlchemy ORM models for signals/snapshots/reasoning
- Test CRUD operations

**Task 2.3: LLM Prompt Templates & Testing**
- Design prompts for each node (trend, momentum, volatility, synthesis)
- Create `src/daemon/prompts.py` - Prompt templates with test harness
- Unit tests: Mock LLM responses, verify reasoning parsing
- Goal: Understand prompt engineering for structured LLM outputs

**Task 2.4: LangGraph Agent Implementation**
- Create `src/daemon/agent.py` - LangGraph state graph definition
- Implement 4 nodes (trend, momentum, volatility, synthesis)
- Implement node-to-node state threading
- Integration tests: Run full graph on test data

**Task 2.5: Daemon Runner & Orchestration**
- Create `src/daemon/runner.py` - Main daemon loop
- Implement: Load data → iterate dates → run agent → store results
- Logging and monitoring (track signals, errors, latency)
- Performance: <200ms per signal (from success criteria)

**Task 2.6: Testing & Documentation**
- Full integration tests (end-to-end: data → signal → database)
- Backtest on 6 months historical data (200+ signals)
- Document daemon architecture and reasoning patterns
- Write Phase 2 completion guide

**Estimated Effort**: 15-20 hours

---

## Design Goals Met

✅ Maximum learning focus - You'll understand LangGraph state machines deeply
✅ Multi-turn reasoning - Each node has targeted LLM reasoning
✅ Hybrid data approach - Phase 2 uses historical, Phase 5 adds real-time
✅ Full auditability - Every signal has complete reasoning chain
✅ Error-handling education - Learn production patterns for reliability
✅ Clear task breakdown - 6 concrete implementation tasks

---

## Next: Implementation Planning

Ready to move to detailed implementation planning with exact file paths, code templates, and step-by-step tasks?
