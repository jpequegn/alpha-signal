# GitHub Issues for Phase 1

This document contains templates for creating 6 GitHub issues for Phase 1: 1 setup task (Task 0) + 5 indicator tasks (Tasks 1.1-1.5).

**Important**: Complete Task 0 (Setup & Infrastructure) first - all other tasks depend on it.

---

## Issue #0: Task 0 - Repository Setup & Infrastructure (PREREQUISITE)

**Title**: Task 0: Repository Setup & Infrastructure - Implement Base Classes & Backtester

**Labels**: `phase-1`, `setup`, `prerequisite`

**Assignee**: (yourself)

**Milestone**: Phase 1: Build Custom Indicators

**Description**:

This is the **foundational task** that must be completed before starting any indicator implementations (Tasks 1.1-1.5). It establishes the shared infrastructure that all indicators depend on.

**What You'll Do**:
- Create the directory structure
- Implement the `Indicator` abstract base class
- Implement the `Backtester` framework
- Set up pytest configuration and fixtures
- Write infrastructure validation tests
- Verify everything works end-to-end

**What You'll Learn**:
- Python package structure and imports
- Abstract base classes and interface design
- Composition patterns for building reusable components
- Test fixtures for data-driven testing
- NumPy best practices
- Professional project organization

**Why This Matters**:
All 5 indicators (SMA, EMA, RSI, MACD, Bollinger Bands) inherit from the base `Indicator` class. The backtester validates that signals work correctly. Without this foundation, each task would duplicate code and have no consistent testing framework.

**Requirements**:
- [ ] Create directory structure as specified
- [ ] Implement `src/indicators/base.py` with `Indicator` abstract base class
- [ ] Implement `src/backtester/engine.py` with backtester function and results dataclass
- [ ] Create all `__init__.py` package files
- [ ] Create `tests/conftest.py` with 5 test fixtures
- [ ] Create `tests/test_setup.py` with 9 infrastructure validation tests
- [ ] All 9 tests passing
- [ ] 100% code coverage on src/indicators and src/backtester
- [ ] Document technology choices (Why NumPy? Why pytest? Why abstract base class?)

**Technical Details**:

**Indicator Base Class**:
```python
class Indicator(ABC):
    def __init__(self, period: int): ...
    @abstractmethod
    def calculate(self, data: np.ndarray) -> np.ndarray: ...
    def __call__(self, data): ...
    def _validate_input(self, data): ...
```

**Backtester**:
```python
def backtest_signal(prices, signals, initial_capital=10000, transaction_cost=0.001):
    # Returns: BacktestResult with metrics
    # - cumulative_return
    # - sharpe_ratio
    # - max_drawdown
    # - win_rate
    # - num_trades
```

**Test Fixtures** (in conftest.py):
- `sample_prices` - 100 realistic prices
- `constant_prices` - 50 constant values (edge case)
- `small_price_array` - 10 prices (minimal)
- `uptrend_prices` - Strongly uptrending
- `downtrend_prices` - Strongly downtrending

**Success Criteria**:
- ✅ Directory structure created
- ✅ Indicator base class prevents instantiation (abstract)
- ✅ Backtester calculates metrics correctly
- ✅ All 9 setup tests passing
- ✅ 100% code coverage
- ✅ Imports work: `from src.indicators import Indicator`
- ✅ Imports work: `from src.backtester import backtest_signal`
- ✅ Technology choices documented

**Resources**:
- [docs/phase1/TASK_0_SETUP.md](./docs/phase1/TASK_0_SETUP.md) - Complete implementation guide with all code
- [Python ABC Documentation](https://docs.python.org/3/library/abc.html)
- [pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)

**Estimated Time**: 3-4 hours

**Blocking**: All other Phase 1 tasks depend on this

**After Completion**:
Once Task 0 is complete, you can immediately start Task 1.1 (SMA) because:
- Base class is ready to inherit from
- Test fixtures are ready to use
- Backtester is ready to validate signals
- Project structure is professional and organized

---

## Issue #1: Task 1.1 - Simple Moving Average (SMA) Indicator

**Title**: Task 1.1: Implement Simple Moving Average (SMA) Indicator

**Labels**: `phase-1`, `indicator`, `good-first-issue`

**Assignee**: (yourself)

**Milestone**: Phase 1: Build Custom Indicators

**Description**:

Implement the Simple Moving Average (SMA) indicator from first principles using NumPy.

**What is SMA?**
- Average of closing prices over N periods
- Used to identify trends (price above SMA = uptrend, below = downtrend)
- Foundation for other indicators

**What You'll Learn**:
- NumPy array operations and windowing
- Efficient rolling calculations using convolution
- Edge case handling (NaN, insufficient data)
- Indicator architecture and reusability

**Requirements**:
- [ ] Create `src/indicators/base.py` with `Indicator` base class
- [ ] Implement `SMA` class in `src/indicators/moving_average.py`
- [ ] Write 6 unit tests in `tests/test_indicators.py`
- [ ] Validate against professional implementations (TradingView, Yahoo Finance)
- [ ] Achieve 100% code coverage
- [ ] Document code with clear docstrings

**Technical Details**:
- Use `np.convolve()` for efficiency
- First `period-1` values should be NaN
- Support any period (5, 20, 50, 200, etc.)

**Success Criteria**:
- ✅ SMA(20) on S&P 500 data matches TradingView within 0.01
- ✅ All unit tests pass
- ✅ No external dependencies (only NumPy)
- ✅ Code is PEP 8 compliant

**Resources**:
- [docs/phase1/TASK_1_SMA.md](./docs/phase1/TASK_1_SMA.md) - Detailed implementation guide
- [docs/IMPLEMENTATION_GUIDE.md](./docs/IMPLEMENTATION_GUIDE.md) - Phase 1 overview

**Estimated Time**: 3 hours

---

## Issue #2: Task 1.2 - Exponential Moving Average (EMA) Indicator

**Title**: Task 1.2: Implement Exponential Moving Average (EMA) Indicator

**Labels**: `phase-1`, `indicator`

**Assignee**: (yourself)

**Milestone**: Phase 1: Build Custom Indicators

**Description**:

Implement the Exponential Moving Average (EMA) indicator - a weighted moving average that emphasizes recent prices.

**What is EMA?**
- EMA responds faster to price changes than SMA
- Uses recursive formula: EMA(t) = (Price × α) + (EMA(t-1) × (1-α))
- Foundation for MACD and other indicators
- α (alpha) = 2 / (period + 1)

**What You'll Learn**:
- Recursive calculations and sequential computation
- Smoothing factors and exponential decay
- Why recent data matters in trading
- How to compose indicators (EMA is used in MACD)

**Requirements**:
- [ ] Implement `EMA` class in `src/indicators/moving_average.py`
- [ ] Include `alpha` property for smoothing factor
- [ ] Write 6 unit tests
- [ ] Validate against pandas `ewm()` function
- [ ] Demonstrate responsiveness compared to SMA
- [ ] 100% code coverage

**Technical Details**:
- Recursive calculation (each value depends on previous)
- Initialize with SMA at index `period-1`
- alpha = 2.0 / (period + 1)

**Success Criteria**:
- ✅ EMA(12) and EMA(26) match professional implementations
- ✅ Validated against pandas `ewm(span=period, adjust=False)`
- ✅ Shows responsiveness to price changes vs SMA
- ✅ All tests pass

**Dependency**:
- Requires Task 1.1 (SMA) to be complete

**Resources**:
- [docs/phase1/TASK_2_EMA.md](./docs/phase1/TASK_2_EMA.md) - Detailed guide
- Task 1.1 completion

**Estimated Time**: 4 hours

---

## Issue #3: Task 1.3 - Relative Strength Index (RSI) Indicator

**Title**: Task 1.3: Implement Relative Strength Index (RSI) Indicator

**Labels**: `phase-1`, `indicator`, `momentum`

**Assignee**: (yourself)

**Milestone**: Phase 1: Build Custom Indicators

**Description**:

Implement the Relative Strength Index (RSI) - a momentum indicator (0-100 scale) identifying overbought/oversold conditions.

**What is RSI?**
- Measures momentum on 0-100 scale
- Formula: RSI = 100 - (100 / (1 + RS)) where RS = AvgGain / AvgLoss
- RSI > 70: Overbought (potential sell)
- RSI < 30: Oversold (potential buy)
- 50: Neutral

**What You'll Learn**:
- Gain/loss calculation and separation
- Wilder's smoothing method
- Momentum analysis
- Threshold-based trading signals

**Requirements**:
- [ ] Implement `RSI` class in `src/indicators/momentum.py` (new file)
- [ ] Separate gains and losses correctly
- [ ] Apply Wilder's smoothing (not EMA)
- [ ] Implement signal generation (>70, <30 thresholds)
- [ ] Write 7 unit tests
- [ ] Validate against professional implementations
- [ ] 100% code coverage

**Technical Details**:
- Calculate changes = diff(prices)
- gains = where(changes > 0, changes, 0)
- losses = where(changes < 0, -changes, 0)
- Use smoothing: (prev × (period-1) + current) / period
- Handle division by zero cases

**Success Criteria**:
- ✅ RSI(14) matches TradingView/professional implementations
- ✅ Correctly identifies overbought/oversold
- ✅ Values always within 0-100 range
- ✅ Signal generation works
- ✅ All tests pass

**Dependency**:
- Requires Task 1.1 (SMA) understanding

**Resources**:
- [docs/phase1/TASK_3_RSI.md](./docs/phase1/TASK_3_RSI.md) - Detailed guide

**Estimated Time**: 5 hours

---

## Issue #4: Task 1.4 - MACD (Moving Average Convergence Divergence) Indicator

**Title**: Task 1.4: Implement MACD (Moving Average Convergence Divergence) Indicator

**Labels**: `phase-1`, `indicator`, `trend-following`

**Assignee**: (yourself)

**Milestone**: Phase 1: Build Custom Indicators

**Description**:

Implement MACD - a trend-following momentum indicator that combines two EMAs. This task teaches indicator composition.

**What is MACD?**
- MACD Line = EMA(12) - EMA(26)
- Signal Line = EMA(9) of MACD Line
- Histogram = MACD - Signal
- MACD > Signal: Bullish (BUY signal)
- MACD < Signal: Bearish (SELL signal)

**What You'll Learn**:
- Composing indicators (MACD reuses EMA)
- Signal line crossover patterns
- Histogram interpretation
- Why composition reduces code duplication

**Requirements**:
- [ ] Implement `MACD` class in `src/indicators/momentum.py`
- [ ] Reuse `EMA` class (don't recalculate)
- [ ] Return tuple: (macd_line, signal_line, histogram)
- [ ] Implement signal generation (crossovers)
- [ ] Write 5 unit tests
- [ ] Validate against TradingView
- [ ] 100% code coverage

**Technical Details**:
- MACD = EMA(12) - EMA(26)
- Signal = EMA(9) of MACD
- Histogram = MACD - Signal
- Signal generation: histogram crossing zero

**Success Criteria**:
- ✅ All calculations match professional implementations
- ✅ Signal generation detects crossovers correctly
- ✅ Demonstrates indicator composition benefits
- ✅ All tests pass

**Dependency**:
- Requires Task 1.2 (EMA) to be complete

**Resources**:
- [docs/phase1/TASK_4_MACD.md](./docs/phase1/TASK_4_MACD.md) - Detailed guide
- Task 1.2 completion

**Estimated Time**: 4 hours

---

## Issue #5: Task 1.5 - Bollinger Bands Indicator

**Title**: Task 1.5: Implement Bollinger Bands Indicator

**Labels**: `phase-1`, `indicator`, `volatility`

**Assignee**: (yourself)

**Milestone**: Phase 1: Build Custom Indicators

**Description**:

Implement Bollinger Bands - a volatility indicator showing upper/lower price bands around a moving average. Final Phase 1 task!

**What are Bollinger Bands?**
- Middle Band = SMA(period), typically 20
- Upper Band = Middle + (2 × StdDev)
- Lower Band = Middle - (2 × StdDev)
- ~95% of prices fall within bands (statistical)
- Wide bands = high volatility; narrow bands = low volatility

**What You'll Learn**:
- Standard deviation and volatility
- Mean reversion concept (different from trend-following)
- Band-based trading signals
- How to measure volatility over time

**Requirements**:
- [ ] Implement `BollingerBands` class in `src/indicators/volatility.py` (new file)
- [ ] Reuse `SMA` class
- [ ] Return tuple: (upper_band, middle_band, lower_band)
- [ ] Implement band width calculation
- [ ] Implement bandwidth percentage
- [ ] Implement signal generation
- [ ] Write 6 unit tests
- [ ] Validate statistical property (95% within bands)
- [ ] 100% code coverage

**Technical Details**:
- Standard deviation: np.std(window) with ddof=0
- Upper = SMA + (std × num_std_dev)
- Lower = SMA - (std × num_std_dev)
- Band width = Upper - Lower
- Signal: Band touches (mean reversion)

**Success Criteria**:
- ✅ Bands correctly bound prices (~95% within)
- ✅ Band width expands during volatility, contracts during calm
- ✅ Signal generation works (band touches)
- ✅ All tests pass
- ✅ **PHASE 1 COMPLETE!**

**Dependency**:
- Requires Task 1.1 (SMA) to be complete

**Resources**:
- [docs/phase1/TASK_5_BOLLINGER_BANDS.md](./docs/phase1/TASK_5_BOLLINGER_BANDS.md) - Detailed guide
- Task 1.1 completion

**Estimated Time**: 3 hours

---

## How to Create These Issues in GitHub

1. Go to your repository on GitHub
2. Click "Issues" → "New Issue"
3. Copy the title and description above
4. Add labels: `phase-1`, indicator type
5. Assign to yourself
6. Set milestone to "Phase 1: Build Custom Indicators"
7. Create issue

### Create Milestone First

Go to Issues → Milestones → New Milestone
- **Title**: Phase 1: Build Custom Indicators
- **Description**: Build 5 custom trading indicators from first principles
- **Due Date**: [2 weeks from now]

---

## Tracking Progress

### Checklist

Phase 1 completion checklist:
- [ ] Issue #1: SMA - COMPLETE
- [ ] Issue #2: EMA - COMPLETE
- [ ] Issue #3: RSI - COMPLETE
- [ ] Issue #4: MACD - COMPLETE
- [ ] Issue #5: Bollinger Bands - COMPLETE
- [ ] All tests passing
- [ ] 100% code coverage achieved
- [ ] All indicators validated
- [ ] Ready for Phase 2

### GitHub Project Board

Create a project board:
1. Projects → New Project
2. Create columns: To Do, In Progress, Review, Done
3. Move issues as you progress

---

## Tips for Success

1. **Do them in order** - Each builds on previous knowledge
2. **Complete all tests** - Unit tests validate your work
3. **Validate against professionals** - Compare with TradingView/Yahoo Finance
4. **Document learnings** - Note key insights for each indicator
5. **Don't skip edge cases** - Tests will catch them

---

**Ready to start Phase 1? 🚀**

Begin with Issue #1 (SMA) and work through in order. Each issue has a detailed implementation guide linked in the description.

