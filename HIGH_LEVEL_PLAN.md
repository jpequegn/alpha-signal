# AlphaSignal - Full Project Roadmap (7 Weeks)

**Project Goal**: Build an autonomous, LLM-powered daemon that generates trading signals with transparent reasoning, learning through first principles rather than black-box systems.

**Learning Philosophy**: Karpathy Method - build from scratch, understand deeply, iterate incrementally.

---

## Timeline Overview

```
Week 1-2: Build Core Indicators (Phase 1)
Week 3-4: LLM Daemon & Signal Generation (Phase 2)
Week 5:   Bubble Detection & Risk Management (Phase 3)
Week 6:   Backtesting & Validation (Phase 4)
Week 7:   Real-Time Integration (Phase 5)
```

---

## Phase 1: Build Custom Indicators (Weeks 1-2)

**Goal**: Implement 5 core trading indicators from first principles, understanding each deeply.

**Why This Matters**:
- Most traders use black-box indicators (talib, pandas-ta)
- You won't understand what fails or why
- Building from scratch forces deep learning
- Each indicator tests a different concept

**Deliverables**:
- ✅ 5 custom indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- ✅ 100% test coverage for indicator library
- ✅ Backtesting on 10 years of S&P 500 data
- ✅ Validation against professional implementations
- ✅ Clear documentation of mechanics for each

### Phase 1 Tasks

**Task 1.1**: Simple Moving Average (SMA)
- **Concept**: Average closing price over N periods
- **Why First**: Simplest indicator, foundation for others
- **Estimated Time**: 3 hours
- **Key Learning**: Windows, numpy operations, rolling data
- **Success Criteria**:
  - SMA(20) matches professional implementations (np.allclose)
  - 100% test coverage
  - Handles edge cases (insufficient data, NaN values)

**Task 1.2**: Exponential Moving Average (EMA)
- **Concept**: Weighted average favoring recent prices
- **Why Second**: Builds on SMA, introduces smoothing factor
- **Estimated Time**: 4 hours
- **Key Learning**: Recursion, alpha smoothing, numerical stability
- **Success Criteria**:
  - EMA(12) and EMA(26) match professional implementations
  - Handles initialization properly (SMA seed)
  - Numerically stable for long sequences

**Task 1.3**: Relative Strength Index (RSI)
- **Concept**: Momentum indicator (0-100), overbought/oversold
- **Why Third**: Introduces gain/loss calculations
- **Estimated Time**: 5 hours
- **Key Learning**: Average gains/losses, normalization, edge cases
- **Success Criteria**:
  - RSI(14) matches TradingView/professional implementations
  - Correctly identifies overbought (>70) and oversold (<30)
  - Handles choppy markets (all gains or all losses)

**Task 1.4**: MACD (Moving Average Convergence Divergence)
- **Concept**: Trend-following momentum (uses EMA12, EMA26, Signal9)
- **Why Fourth**: Combines multiple EMAs, introduces signal crossing
- **Estimated Time**: 4 hours
- **Key Learning**: Multi-indicator composition, signal lines, histogram
- **Success Criteria**:
  - MACD line, signal line, histogram all match professional
  - Correctly identifies bullish/bearish crosses
  - No forward-looking bias in signal calculation

**Task 1.5**: Bollinger Bands
- **Concept**: Volatility bands around SMA (mean ± 2 std devs)
- **Why Fifth**: Introduces volatility concept
- **Estimated Time**: 3 hours
- **Key Learning**: Standard deviation, band logic, mean reversion
- **Success Criteria**:
  - Upper/lower bands match professional implementations
  - Correctly identifies when price touches bands
  - Handles periods of low volatility

### Phase 1 Success Criteria (Overall)

- ✅ All 5 indicators implemented from scratch (no talib, pandas-ta, etc.)
- ✅ Each indicator has 10+ unit tests covering normal + edge cases
- ✅ Backtester can load 10 years of S&P 500 daily data
- ✅ Each indicator backtested on 2015-2025 data
- ✅ Professional validation (results ≈ TradingView/Yahoo Finance)
- ✅ Documentation explains *why* each indicator works
- ✅ Code is readable, well-commented, follows PEP 8

### Phase 1 Estimated Effort
- **Total Hours**: ~20 hours hands-on
- **Research**: ~5 hours reading about indicators
- **Testing**: ~5 hours writing/debugging tests
- **Validation**: ~3 hours comparing implementations

---

## Phase 2: LLM Daemon for Signal Generation (Weeks 3-4)

**Goal**: Build autonomous daemon that reasons about multiple signals using LLM.

**Architecture**:
```
Market Data Stream (Redis) →
  Data Fetcher →
    Indicator Calculator →
      LLM Signal Agent →
        Signal Database (PostgreSQL)
```

**Key Components**:

1. **Data Fetcher Service**
   - Real-time market data (OHLCV)
   - Sliding window (keep 200+ periods)
   - Redis stream for inter-service communication

2. **Indicator Calculator**
   - Uses Phase 1 indicators
   - Calculates all 5 for each symbol
   - Detects convergence/divergence

3. **LLM Signal Agent (LangGraph)**
   - Receives indicator state
   - Reasons: "Given SMA uptrend + RSI at 65 + MACD bullish..."
   - Generates: {signal: "BUY", confidence: 0.78, reasoning: "..."}
   - Stores to database

4. **Signal Storage**
   - All signals with timestamp
   - Confidence scores
   - Full reasoning from LLM
   - Indicator states at time of signal

**Deliverables**:
- ✅ LangGraph daemon structure
- ✅ LLM prompt template for signal reasoning
- ✅ PostgreSQL schema for signals/metrics
- ✅ 1+ week of signal history
- ✅ Basic dashboard showing recent signals

### Phase 2 Success Criteria

- ✅ Daemon runs 24/5 without crashes
- ✅ Signals generated with timestamps + confidence
- ✅ Full reasoning stored for each signal
- ✅ < 200ms latency from data arrival to signal
- ✅ Can query signals by symbol, time range, confidence
- ✅ LLM reasoning is reproducible and logged

---

## Phase 3: Multi-Factor Bubble Detection (Week 5)

**Goal**: Implement risk awareness - reduce signal confidence during market bubbles.

**Risk Factors**:
1. **Valuation Metrics**
   - P/E ratio vs. historical average
   - P/B ratio trend
   - Dividend yield compression

2. **Volatility Indicators**
   - VIX trend (increasing = more risk)
   - Daily swing size (large = risk)
   - Correlation breakdown (when diversification fails)

3. **Market Breadth**
   - % of stocks above 200-day MA
   - New highs vs. new lows
   - Sector concentration

4. **Sentiment/Momentum**
   - News toxicity (negative scores)
   - Social media mentions
   - Insider selling/buying

**LLM Bubble Detector**:
- Synthesize all risk factors
- Generate: {bubble_probability: 0.67, risk_level: "HIGH"}
- Reduce signal confidence proportionally

**Example**:
```
Normal Market: BUY signal confidence = 0.78
Bubble Market: 0.78 × (1 - 0.67) = 0.26 (weaker signal)
```

### Phase 3 Success Criteria

- ✅ Detects market extremes (2008, 2020, 2021 peaks)
- ✅ Bubble probability correlates with subsequent crashes
- ✅ Signal confidence reduced during risk periods
- ✅ LLM reasoning explains bubble detection rationale

---

## Phase 4: Backtesting & Validation (Week 6)

**Goal**: Prove signals work (or learn why they don't).

**Backtesting Framework**:
1. **Load Historical Data**: 2015-2025 S&P 500 daily
2. **Replay Signals**: Generate signals on each date
3. **Simulate Trades**: Execute on signal, hold until exit
4. **Calculate Metrics**:
   - Win rate (% profitable)
   - Profit factor (gross profit / gross loss)
   - Sharpe ratio (risk-adjusted returns)
   - Max drawdown
   - Sortino ratio

5. **Comparison**: vs. buy-and-hold baseline

**Analysis**:
- Identify which signals work best
- Identify which signals are noise
- Refine signal weights
- A/B test variations

### Phase 4 Success Criteria

- ✅ Backtest runs in <5 minutes on 10 years data
- ✅ Win rate > 55% (better than random)
- ✅ Profit factor > 1.5 (1.5x gross profit vs. losses)
- ✅ Sharpe ratio > 0.5 (reasonable risk-adjusted return)
- ✅ Detailed report explaining signal quality

---

## Phase 5: Real-Time Integration (Week 7)

**Goal**: Connect to live market data, run daemon for paper trading.

**Integration Points**:
1. **Market Data Provider**: Alpaca API
   - Real-time quotes
   - Trade & quote updates
   - Paper trading endpoint

2. **Data Pipeline**:
   - Stream market data to Redis
   - Daemon consumes, generates signals
   - Store signals in PostgreSQL

3. **Paper Trading**:
   - Execute signals on paper account
   - Track P&L
   - Verify signal quality in real-time

4. **Claude Code Integration** (from LLM daemon research):
   - MCP resource for querying latest signals
   - Hooks for automatic report generation
   - Daily digest of signal performance

### Phase 5 Success Criteria

- ✅ Daemon connects to Alpaca API
- ✅ Real-time signals generated continuously
- ✅ Paper trading account synced
- ✅ Daily performance reports
- ✅ Ready for Phase 6 (live trading)

---

## Phase 6+ (Future)

**Phase 6**: Live Trading (with proper risk management, position sizing)
**Phase 7**: Multi-symbol expansion (not just S&P 500)
**Phase 8**: Advanced features (options strategies, hedging)

---

## Success Metrics (Project-Wide)

### Technical
- ✅ 100% test coverage on indicator library
- ✅ Zero crashes in daemon (24/5 uptime)
- ✅ < 200ms signal latency
- ✅ All code follows PEP 8 / best practices

### Learning
- ✅ Understand how each indicator works (mechanics, not magic)
- ✅ Know why signals succeed/fail
- ✅ Comfortable with async Python + LLMs
- ✅ Can explain system to non-technical person

### Financial
- ✅ Backtest win rate > 55%
- ✅ Profit factor > 1.5
- ✅ Sharpe ratio > 0.5
- ✅ Outperforms buy-and-hold 60%+ of periods

### Portfolio
- ✅ Production-ready LLM daemon
- ✅ Real-time market integration
- ✅ Comprehensive documentation
- ✅ Demonstrates AI + finance skills

---

## Technology Stack (Summary)

| Layer | Technology | Why |
|-------|-----------|-----|
| **Language** | Python 3.11+ | Ecosystem, numpy/pandas |
| **Async** | asyncio | Daemon operations |
| **LLM Framework** | LangGraph | Production-ready daemon |
| **LLM** | Claude (Anthropic) | Reasoning quality |
| **Data Stream** | Redis Streams | Low-latency pub-sub |
| **Data Store** | PostgreSQL | Relational data, query flexibility |
| **Market Data** | Alpaca API | Free, reliable, paper trading |
| **Backtesting** | Custom (vectorized) | Simple, fast, understandable |
| **Testing** | pytest | Standard, comprehensive |
| **Docs** | Markdown + Jupyter | Clear + interactive |

---

## Git Workflow

Each phase is a milestone with tagged releases:
```
v0.1.0 - Phase 1 complete (indicators)
v0.2.0 - Phase 2 complete (LLM daemon)
v0.3.0 - Phase 3 complete (bubble detection)
v0.4.0 - Phase 4 complete (validated backtests)
v0.5.0 - Phase 5 complete (live integration)
v1.0.0 - Production ready
```

Each task is a GitHub issue, closed when complete + merged to main.

---

## Resource Allocation

**Estimated Total Time**: 60-80 hours
- Phase 1: 20 hours
- Phase 2: 15 hours
- Phase 3: 10 hours
- Phase 4: 12 hours
- Phase 5: 10 hours
- Buffer: 5-15 hours

**Recommended Pace**: 10-15 hours/week → 6-8 weeks total

---

## Next Steps

1. **Read**: [docs/IMPLEMENTATION_GUIDE.md](./docs/IMPLEMENTATION_GUIDE.md) - Phase 1 detailed breakdown
2. **Create GitHub Issues**: 5 tasks for Phase 1 (see IMPLEMENTATION_GUIDE)
3. **Start Task 1.1**: Simple Moving Average indicator
4. **Track Progress**: Use GitHub issues + project board
5. **Document Learning**: Write brief notes on each indicator's "aha moment"

---

## Questions?

- **"Why build from scratch?"** → Karpathy method - deep understanding vs. black boxes
- **"How do I validate my indicators?"** → Compare against TradingView, Yahoo Finance, professional implementations
- **"What if indicators don't work?"** → That's the learning - understand *why*, refine approach
- **"Can I skip ahead?"** → Not recommended - each phase builds on previous
- **"How much time do I have?"** → Flexible - go at your pace, but follow order

---

**Start Date**: [Insert today's date]
**Target Completion**: [Insert date + 8 weeks]
**Current Status**: Planning Phase 1

---

**Let's build something great! 🚀**
