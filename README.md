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

### Running Phase 1 Tests

```bash
# Run indicator tests
pytest tests/test_indicators.py -v

# Run backtester tests
pytest tests/test_backtester.py -v
```

## Development Path (Karpathy Method)

### Phase 1: Build Custom Indicators (Weeks 1-2)
- Implement 5 core indicators from first principles
- Understand each indicator's mechanics deeply
- Test on historical data (10 years, S&P 500)
- Success: All indicators validated, no external indicator libraries

**Status**: In Planning
**Tasks**: 5 GitHub issues, estimated 40 hours

### Phase 2: LLM Agent for Signal Generation (Weeks 3-4)
- Build LangGraph-based daemon
- Implement LLM reasoning about signals
- Store decisions with rationale
- Success: Daemon generates signals continuously with reasoning

### Phase 3: Multi-Factor Bubble Detection (Week 5)
- Monitor market risk factors
- Implement bubble probability scoring
- Reduce signal confidence during bubbles
- Success: System detects and adapts to market extremes

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

- **[HIGH_LEVEL_PLAN.md](./HIGH_LEVEL_PLAN.md)** - Full project roadmap with timelines
- **[docs/IMPLEMENTATION_GUIDE.md](./docs/IMPLEMENTATION_GUIDE.md)** - Phase 1 detailed guide
- **[docs/phase1/](./docs/phase1/)** - Individual task implementation guides
- **[docs/architecture/](./docs/architecture/)** - System design documentation

## GitHub Issues

Phase 1 tasks tracked as GitHub issues:
- [ ] Task 1: Build SMA (Simple Moving Average) Indicator
- [ ] Task 2: Build EMA (Exponential Moving Average) Indicator
- [ ] Task 3: Build RSI (Relative Strength Index) Indicator
- [ ] Task 4: Build MACD (Moving Average Convergence Divergence) Indicator
- [ ] Task 5: Build Bollinger Bands Indicator

## Learning Philosophy

This project follows the **Karpathy Method**:
1. **Build from First Principles** - Implement indicators from raw OHLCV data
2. **Learn by Doing** - Code every concept, test iteratively
3. **Minimal Dependencies** - Avoid black-box libraries initially
4. **Incremental Progress** - One indicator at a time, test each
5. **Deep Understanding** - Know why each indicator works

No TradingView scripts, no talib, no pandas-ta. Build it yourself and understand it deeply.

## Success Metrics

- ✅ All indicators implemented and unit tested
- ✅ 100% code coverage for indicator library
- ✅ Validated against professional implementations
- ✅ Backtesting shows meaningful signal quality
- ✅ Clear documentation of each indicator's mechanics

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
