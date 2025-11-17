"""Backtesting framework for trading signals."""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class BacktestResult:
    """Results of a backtest run."""

    signals: np.ndarray          # Buy (1), Sell (-1), Hold (0)
    entry_prices: np.ndarray     # Price when signal triggered
    exit_prices: np.ndarray      # Price when position closed
    trade_returns: np.ndarray    # Per-trade return percentage
    cumulative_return: float     # Total return %
    sharpe_ratio: float          # Risk-adjusted return
    max_drawdown: float          # Worst peak-to-trough decline
    win_rate: float              # % of profitable trades
    num_trades: int              # Total number of trades

    def summary(self) -> str:
        """Print human-readable summary."""
        return f"""
Backtest Results:
  Total Return: {self.cumulative_return:.2f}%
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Max Drawdown: {self.max_drawdown:.2f}%
  Win Rate: {self.win_rate:.2f}%
  Total Trades: {self.num_trades}
"""


def backtest_signal(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001
) -> BacktestResult:
    """
    Simple backtester for indicator signals.

    Args:
        prices: Array of closing prices
        signals: Array of signals (1=buy, -1=sell, 0=hold)
        initial_capital: Starting capital
        transaction_cost: Percentage cost per trade (0.001 = 0.1%)

    Returns:
        BacktestResult with performance metrics
    """
    if len(prices) != len(signals):
        raise ValueError("prices and signals must have same length")

    if len(prices) < 2:
        raise ValueError("Need at least 2 price points")

    # Track trades
    position = 0              # 0=no position, 1=long
    entry_price = None
    trades = []

    # Generate trade prices
    entry_prices = np.full(len(prices), np.nan)
    exit_prices = np.full(len(prices), np.nan)

    for i, (price, signal) in enumerate(zip(prices, signals)):
        # Entry signal
        if signal == 1 and position == 0:
            position = 1
            entry_price = price * (1 + transaction_cost)
            entry_prices[i] = entry_price

        # Exit signal
        elif signal == -1 and position == 1:
            exit_price = price * (1 - transaction_cost)
            exit_prices[i] = exit_price

            trade_return = (exit_price - entry_price) / entry_price
            trades.append(trade_return)

            position = 0
            entry_price = None

    # Calculate metrics
    trade_returns = np.array(trades) if trades else np.array([0.0])
    cumulative_return = np.prod(1 + trade_returns) - 1

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(trades) > 1 and np.std(trades) > 0:
        sharpe_ratio = np.mean(trades) / np.std(trades) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Max drawdown
    cumul = np.cumprod(1 + trade_returns)
    running_max = np.maximum.accumulate(cumul)
    drawdown = (cumul - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

    # Win rate
    wins = np.sum(trade_returns > 0)
    win_rate = wins / len(trades) if trades else 0.0

    return BacktestResult(
        signals=signals,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        trade_returns=trade_returns,
        cumulative_return=cumulative_return * 100,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown * 100,
        win_rate=win_rate * 100,
        num_trades=len(trades)
    )
