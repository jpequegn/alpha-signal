# AlphaSignal - Autonomous Trading Signal Generation Engine

An LLM-powered daemon for autonomous trading signal generation, built using the Karpathy Method (learn from first principles, build from scratch, iterate incrementally).

## Project Overview

**Goal**: Build a production-ready daemon that generates trading signals by:
1. Computing custom indicators from raw market data (no black-box libraries)
2. Using LLMs to reason about multiple signals simultaneously
3. Running continuously, monitoring market data streams
4. Providing transparent decision rationale for each signal

## Key Technologies

- **Python 3.11+** with async/await for daemon operations
- **NumPy + Pandas** for signal computation (learning the mechanics)
- **LangGraph** for LLM daemon orchestration (from research)
- **PostgreSQL** for signal storage and metrics
- **Redis Streams** for real-time market data
- **Alpaca API** for market data and paper trading

## Project Structure

```
alpha-signal/
├── README.md                    # This file
├── HIGH_LEVEL_PLAN.md           # Full 7-week project roadmap
├── docs/
│   ├── IMPLEMENTATION_GUIDE.md   # Phase 1 detailed guide
│   ├── phase1/
│   │   ├── TASK_1.md           # SMA indicator implementation
│   │   ├── TASK_2.md           # EMA indicator implementation
│   │   ├── TASK_3.md           # RSI indicator implementation
│   │   ├── TASK_4.md           # MACD indicator implementation
│   │   └── TASK_5.md           # Bollinger Bands implementation
│   └── architecture/
│       └── DAEMON_DESIGN.md     # LLM daemon architecture
├── src/
│   ├── __init__.py
│   ├── indicators/              # Custom indicator implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Base indicator class
│   │   ├── moving_average.py    # SMA, EMA implementations
│   │   ├── momentum.py          # RSI, MACD implementations
│   │   └── volatility.py        # Bollinger Bands implementation
│   ├── data/                    # Market data handling
│   │   ├── __init__.py
│   │   └── loader.py           # Historical data loading
│   └── backtester/              # Backtesting framework
│       ├── __init__.py
│       └── engine.py            # Backtesting engine
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py       # Unit tests for indicators
│   └── test_backtester.py       # Backtesting tests
├── notebooks/                   # Jupyter notebooks for exploration
│   └── .gitkeep
├── data/                        # Historical market data
│   └── .gitkeep
├── .gitignore
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup

```

## Quick Start

### Prerequisites
- Python 3.11+
- pip/poetry
- PostgreSQL (for Phase 4+)
- Alpaca account (for Phase 5+)

### Installation

```bash
git clone https://github.com/yourusername/alpha-signal.git
cd alpha-signal
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/indicators --cov-report=term-missing

# Run specific test class
pytest tests/test_indicators.py::TestBollingerBands -v
```

## Quick Links

**Getting Started:**
- 🚀 **[5-Minute Quick Start](./docs/QUICK_START.md)** - Get up and running immediately
- 📖 **[User Guide](./docs/PHASE1_USER_GUIDE.md)** - Complete trading guide with examples

**Developer Resources:**
- 🏗️ **[Architecture Guide](./docs/ARCHITECTURE.md)** - System design and patterns
- 📚 **[API Reference](./docs/API_REFERENCE.md)** - Complete API documentation
- 🤝 **[Contributing Guide](./docs/CONTRIBUTING.md)** - How to contribute

**Implementation Details:**
- 📋 **[Implementation Guide](./docs/IMPLEMENTATION_GUIDE.md)** - Phase 1 detailed guide
- 📑 **[Task Documentation](./docs/phase1/)** - Individual indicator guides

## Development Path (Karpathy Method)

### Phase 1: Build Custom Indicators (Weeks 1-2) ✅ COMPLETE

**Status**: ✅ COMPLETE
**Completion**: All 5 core indicators implemented and tested
**Code Coverage**: 90% (exceeds 80% target)
**Total Tests**: 71 passing (44 existing + 27 Bollinger Bands)

#### Implemented Indicators

| Indicator | Type | Purpose | Status | Tests |
|-----------|------|---------|--------|-------|
| **SMA** | Trend | Simple moving average smoothing | ✅ | 8 |
| **EMA** | Trend | Exponential moving average with alpha | ✅ | 9 |
| **RSI** | Momentum | Relative strength index (0-100) | ✅ | 12 |
| **MACD** | Composition | MA convergence/divergence + signals | ✅ | 15 |
| **Bollinger Bands** | Volatility | Statistical bands with 95% property | ✅ | 15 |

#### Key Achievements

✅ All 5 indicators built from first principles (no black-box libraries)
✅ 90% code coverage across all indicators
✅ Comprehensive docstrings with mathematical formulas
✅ Signal generation for each indicator type
✅ Edge case handling (constant prices, insufficient data, etc.)
✅ Production-ready code with error handling
✅ Full test suite with parametric and fixture-based tests
✅ Complete documentation (API ref, user guide, architecture)

#### Documentation

- **[PHASE1_USER_GUIDE.md](./docs/PHASE1_USER_GUIDE.md)** - Complete user guide with examples and strategies
- **[API_REFERENCE.md](./docs/API_REFERENCE.md)** - Full API documentation for all indicators
- **[ARCHITECTURE.md](./docs/ARCHITECTURE.md)** - System design and implementation details
- **[QUICK_START.md](./docs/QUICK_START.md)** - 5-minute getting started guide
- **[CONTRIBUTING.md](./docs/CONTRIBUTING.md)** - Developer guide for contributions

### Phase 2: LLM Agent for Signal Generation (Weeks 3-4) ✅ COMPLETE

**Status**: ✅ COMPLETE

Completed:
- ✅ LangGraph daemon with 4-node reasoning architecture
- ✅ Multi-turn LLM reasoning (trend → momentum → volatility → synthesis)
- ✅ PostgreSQL signal storage with reasoning auditability
- ✅ Historical data loader with indicator batch calculation
- ✅ 50+ integration tests covering full pipeline
- ✅ 85%+ code coverage on daemon modules
- ✅ Full daemon documentation and guides

Key Achievements:
- Multi-turn reasoning flow teaches LangGraph patterns
- Complete signal reasoning chain stored for learning/debugging
- Graceful error handling with retry patterns
- Ready for Phase 5 real-time integration (just swap data source)

**Documentation**: See [PHASE2_GUIDE.md](./docs/PHASE2_GUIDE.md) for complete Phase 2 guide

### Phase 3: Multi-Factor Bubble Detection (Week 5) ✅ COMPLETE

**Status**: ✅ COMPLETE
**Completion**: 8 tasks, 255+ tests, 95%+ code coverage
**Duration**: 2025-11-19 to 2025-11-20

Completed:
- ✅ Historical data backfill (2015-2024, ~2500 signals)
- ✅ 4-factor risk assessment (valuation, volatility, breadth, momentum)
- ✅ LLM bubble probability synthesis (Claude reasoning)
- ✅ Signal confidence adjustment during market extremes
- ✅ Complete end-to-end integration testing
- ✅ Database model for persistence (BackfillSignal)

Key Achievements:
- Per-signal risk evaluation (not market-wide)
- Heuristic fallback when LLM unavailable
- Performance: Process 500 signals/day in <2 seconds
- Full test coverage: 255+ tests across 6 modules

**Documentation**: See [PHASE3_GUIDE.md](./docs/PHASE3_GUIDE.md) for complete guide

### Phase 4: Backtesting & Validation (Week 6)
- Comprehensive historical testing
- Generate performance reports
- Compare vs. buy-and-hold
- Success: Signals prove profitable or learning identifies why

### Phase 5: Real-Time Integration (Week 7)
- Connect to market data feeds
- Stream data to daemon
- Real-time signal generation
- Success: Live daemon ready for paper trading

## Documentation

### Project Planning
- **[HIGH_LEVEL_PLAN.md](./HIGH_LEVEL_PLAN.md)** - Full project roadmap with timelines

### Phase Guides
- **[docs/PHASE1_USER_GUIDE.md](./docs/PHASE1_USER_GUIDE.md)** - Phase 1 complete user guide
- **[docs/PHASE2_GUIDE.md](./docs/PHASE2_GUIDE.md)** - Phase 2 LLM daemon guide
- **[docs/PHASE3_GUIDE.md](./docs/PHASE3_GUIDE.md)** - Phase 3 risk assessment guide

### Technical References
- **[docs/IMPLEMENTATION_GUIDE.md](./docs/IMPLEMENTATION_GUIDE.md)** - Phase 1 implementation details
- **[docs/phase1/](./docs/phase1/)** - Individual indicator implementation guides
- **[docs/architecture/](./docs/architecture/)** - System design documentation
- **[docs/API_REFERENCE.md](./docs/API_REFERENCE.md)** - Complete API documentation

## GitHub Issues - Phase 1 ✅ COMPLETE

All Phase 1 tasks completed and closed:
- [x] Task 1.1: Build SMA (Simple Moving Average) Indicator ✅
- [x] Task 1.2: Build EMA (Exponential Moving Average) Indicator ✅
- [x] Task 1.3: Build RSI (Relative Strength Index) Indicator ✅
- [x] Task 1.4: Build MACD (Moving Average Convergence Divergence) Indicator ✅
- [x] Task 1.5: Build Bollinger Bands Indicator ✅

## Learning Philosophy

This project follows the **Karpathy Method**:
1. **Build from First Principles** - Implement indicators from raw OHLCV data
2. **Learn by Doing** - Code every concept, test iteratively
3. **Minimal Dependencies** - Avoid black-box libraries initially
4. **Incremental Progress** - One indicator at a time, test each
5. **Deep Understanding** - Know why each indicator works

No TradingView scripts, no talib, no pandas-ta. Build it yourself and understand it deeply.

## Success Metrics - Phase 1 ✅ MET

- ✅ All 5 indicators implemented from first principles
- ✅ 90% code coverage for indicator library (exceeded 80% target)
- ✅ 71 unit tests covering all code paths
- ✅ Edge case handling (constant prices, insufficient data, etc.)
- ✅ Clear documentation with mathematical formulas
- ✅ Signal generation for each indicator type
- ✅ Production-ready error handling and validation
- ✅ Comprehensive user guide with examples
- ✅ Complete API reference documentation
- ✅ Architecture and design documentation
- ✅ Contributing guide for future development

## Contributing

This is a personal learning project, but contributions are welcome. Please:
1. Follow the Karpathy method (implement from scratch)
2. Add tests for any new code
3. Document deeply why, not just what
4. Keep implementations simple and readable

## License

MIT

## Contact

Questions? Issues? Open a GitHub issue or reach out.

---

**Next Step**: Read [HIGH_LEVEL_PLAN.md](./HIGH_LEVEL_PLAN.md) for the full roadmap, then [docs/IMPLEMENTATION_GUIDE.md](./docs/IMPLEMENTATION_GUIDE.md) for Phase 1 details.
