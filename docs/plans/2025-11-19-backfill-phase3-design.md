# Backfill + Phase 3: Risk Assessment Design

**Date**: 2025-11-19
**Status**: Design Complete - Ready for Implementation
**Scope**: Historical data backfill (2015-2024) + Per-signal bubble detection

---

## Problem Statement

**Backfill**: Phase 2 daemon is production-ready but untested on real historical data. Need comprehensive 2015-2024 signal history to validate Phase 2 and enable Phase 4 backtesting.

**Phase 3**: Current Phase 2 signals ignore market regime (bubble vs normal). Need risk assessment to adjust confidence during market extremes, improving signal quality.

---

## Architecture Overview

```
Historical Data (2015-2024, daily OHLCV)
    ↓
┌─────────────────────────────────────┐
│ Phase 2: Signal Generation          │
│ (trend → momentum → volatility       │
│  → synthesis decision)               │
└─────────────────────────────────────┘
    ↓
Original Signal Output:
├─ Signal: BUY|SELL|HOLD
├─ Confidence: 0.0-1.0
└─ Reasoning: Full explanation
    ↓
┌─────────────────────────────────────┐
│ Phase 3: Risk Assessment (NEW)      │
│ Per-signal bubble detection         │
└─────────────────────────────────────┘
    ↓
Risk Factor Evaluation (4 factors):
├─ Valuation Risk (P/E vs historical)
├─ Volatility Risk (VIX level)
├─ Market Breadth Risk (% above 200MA)
└─ Momentum Risk (rate of price change)
    ↓
LLM Synthesis:
├─ Synthesize 4 factors into bubble probability
├─ Reasoning: Why is market in bubble state?
└─ Confidence Adjustment: original × (1 - bubble_prob)
    ↓
PostgreSQL Storage:
├─ Original signal + confidence
├─ 4 risk factor scores
├─ Bubble probability
├─ Adjusted confidence
└─ Risk reasoning audit trail
```

---

## Backfill Component: Historical Data Loading

**Purpose**: Generate complete signal history for 2015-2024 (10 years of SPY data)

**Data Source**: yfinance (free, reliable, covers full period)

**Process**:
1. Load daily OHLCV for SPY (2015-01-01 to 2024-12-31) ~2,500 price points
2. Batch process through Phase 2 daemon at regular intervals (weekly or monthly snapshots)
3. Store signals in new `backfill_signals` table for analysis
4. Expected output: ~500-1000 signals (roughly 2-3 per week average)

**Why Separate Table**:
- Backfill is historical analysis, not real-time signals
- Can analyze without affecting production signal tracking
- Easy to compare Phase 2 performance (before Phase 3) vs Phase 2+3 (after)

**Key Design**:
- Reuse Phase 2 daemon entirely (no code changes)
- Just feed it historical prices instead of real-time
- This validates Phase 2 works at scale

---

## Phase 3 Risk Assessment: Per-Signal Evaluation

**Purpose**: Adjust signal confidence based on market risk factors

**Design Principle**: Risk is evaluated *per signal*, not market-wide
- Each signal's confidence adjusted independently
- Allows nuanced assessment (uptrend bullish even in bubble if valuation OK)
- More complex reasoning, better signal quality

### Risk Factor 1: Valuation Risk

```python
# P/E Ratio historical context
if pe_ratio > 90th_percentile_historical:
    valuation_risk = 0.6-0.8  # Elevated
elif pe_ratio > 75th_percentile:
    valuation_risk = 0.3-0.6  # Moderate
else:
    valuation_risk = 0.0-0.2  # Normal

# During backfill: Synthetic P/E data
# Oscillates 15-35 range, mimics real market
```

### Risk Factor 2: Volatility Risk

```python
# VIX as proxy for market fear
if vix > 40:
    volatility_risk = 0.6-0.8  # Extreme fear/greed
elif vix > 30:
    volatility_risk = 0.3-0.6  # Elevated
else:
    volatility_risk = 0.0-0.2  # Normal

# During backfill: Synthetic VIX 10-50 range
```

### Risk Factor 3: Market Breadth Risk

```python
# % of stocks trading above 200-day moving average
if breadth < 30%:
    breadth_risk = 0.6-0.8  # Market declining, few stocks participating
elif breadth < 50%:
    breadth_risk = 0.3-0.6  # Moderate decline
else:
    breadth_risk = 0.0-0.2  # Healthy participation

# During backfill: Calculate from synthetic data
```

### Risk Factor 4: Momentum Risk

```python
# Rate of price change (can signal extremes)
if price_change > +10% per week:
    momentum_risk = 0.3-0.5  # Parabolic move, extreme
elif price_change > +5% per week:
    momentum_risk = 0.1-0.3  # Strong momentum
elif price_change < -10% per week:
    momentum_risk = 0.5-0.7  # Panic selling
else:
    momentum_risk = 0.0-0.2  # Normal
```

---

## LLM Bubble Probability Synthesis

**Input**: 4 risk factor scores (0.0-0.8 range)

**LLM Prompt**:
```
You are a market risk analyst. Given these risk factors
(0.0=low risk, 0.8=high risk):
- Valuation Risk: {valuation_risk}
- Volatility Risk: {volatility_risk}
- Breadth Risk: {breadth_risk}
- Momentum Risk: {momentum_risk}

Synthesize into overall bubble probability (0.0-1.0):
0.0 = No bubble, market healthy
0.5 = Moderate bubble probability
1.0 = Extreme bubble, market extremes

Respond with JSON:
{
    "bubble_probability": 0.0-1.0,
    "reasoning": "Why is market in this regime?",
    "key_risk": "Most concerning factor"
}
```

**Output**: Bubble probability (0-1) + reasoning

**Confidence Adjustment**:
```python
adjusted_confidence = original_confidence × (1 - bubble_probability)

# Examples:
# Normal market (bubble_prob=0.2): 0.78 × 0.8 = 0.624
# Moderate bubble (bubble_prob=0.5): 0.78 × 0.5 = 0.390
# Extreme bubble (bubble_prob=0.8): 0.78 × 0.2 = 0.156
```

---

## Database Schema Extension

New `backfill_signals` table (for historical analysis):

```sql
CREATE TABLE backfill_signals (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10),
  timestamp TIMESTAMPTZ,

  -- Phase 2 Original Signal
  signal VARCHAR(10),           -- BUY|SELL|HOLD
  confidence FLOAT,
  final_reasoning TEXT,

  -- Phase 3 Risk Factors
  valuation_risk FLOAT,         -- 0.0-0.8
  volatility_risk FLOAT,        -- 0.0-0.8
  breadth_risk FLOAT,           -- 0.0-0.8
  momentum_risk FLOAT,          -- 0.0-0.8

  -- Phase 3 Bubble Assessment
  bubble_probability FLOAT,     -- 0.0-1.0
  risk_reasoning TEXT,

  -- Adjusted Confidence
  adjusted_confidence FLOAT,    -- original × (1 - bubble_prob)

  created_at TIMESTAMPTZ
);
```

---

## Implementation Tasks

**Backfill Phase** (3 tasks):
1. Data fetcher: Load 2015-2024 SPY data from yfinance
2. Signal generator: Batch run Phase 2 daemon on historical data
3. Storage: Save to backfill_signals table

**Phase 3 Risk Assessment** (5 tasks):
4. Risk factors: Calculate 4 risk factor scores
5. Synthetic data generator: Create realistic but fake risk data for testing
6. Bubble scorer: LLM synthesis of 4 factors → bubble probability
7. Confidence adjuster: Apply adjustment to signals
8. Integration & tests: Full backfill + Phase 3 end-to-end

**Estimated Effort**: 20-25 hours total
- Backfill: 5-7 hours
- Phase 3: 15-18 hours

---

## Success Criteria

✅ Backfill generates 500+ signals from 2015-2024 data
✅ Phase 3 evaluates all signals with risk assessment
✅ Adjusted confidence decreases during synthetic bubble periods
✅ Full reasoning audit trail stored (can replay any signal)
✅ 80%+ test coverage on risk calculation modules
✅ Database queries show signal quality improvement (Phase 2 vs Phase 2+3)

---

## Data: Synthetic vs Real

**Backfill Phase 2 Signals**: Use real 2015-2024 SPY prices (yfinance)

**Phase 3 Risk Data**: Start with synthetic to learn, add real in Phase 5
- Valuation: Oscillate P/E 15-35 with trends
- Volatility: Synthetic VIX 10-50 with clustering
- Breadth: Synthetic % 20-80% correlated with prices
- Momentum: Derived from price rate of change

**Why Synthetic**:
- Allows full control for testing
- Validates algorithms before adding real API complexity
- Can test extreme scenarios (2008 crash, 2021 bubble, etc.)

**Phase 5 Upgrade**: Replace with real APIs (yfinance, Alpaca)

---

## Next Steps

1. **Create detailed implementation plan** with exact file paths, code templates, TDD steps
2. **Dispatch subagents per task** (like Phase 2 implementation)
3. **Generate 10-year signal history** via backfill
4. **Analyze Phase 2 vs Phase 2+3 signal quality**
5. **Proceed to Phase 4** (backtesting with performance metrics)

