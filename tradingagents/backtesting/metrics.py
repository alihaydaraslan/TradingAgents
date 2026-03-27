"""Financial metrics for backtesting results."""

import math
from typing import List, Tuple


def calculate_sharpe(daily_values: List[Tuple[str, float]], risk_free_rate: float = 0.04) -> float:
    """Calculate annualized Sharpe ratio from daily portfolio values."""
    if len(daily_values) < 2:
        return 0.0

    values = [v for _, v in daily_values]
    daily_returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            daily_returns.append((values[i] - values[i - 1]) / values[i - 1])

    if not daily_returns:
        return 0.0

    mean_return = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
    std = math.sqrt(variance) if variance > 0 else 0.001

    daily_rf = risk_free_rate / 252
    sharpe = (mean_return - daily_rf) / std * math.sqrt(252)
    return round(sharpe, 2)


def calculate_max_drawdown(daily_values: List[Tuple[str, float]]) -> float:
    """Calculate maximum drawdown percentage."""
    if len(daily_values) < 2:
        return 0.0

    values = [v for _, v in daily_values]
    peak = values[0]
    max_dd = 0.0

    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return round(max_dd * 100, 2)


def calculate_win_rate(trades) -> float:
    """Calculate win rate from completed trade pairs (buy then sell)."""
    buy_price = None
    wins = 0
    total = 0

    for trade in trades:
        if trade.action == "BUY" and buy_price is None:
            buy_price = trade.price
        elif trade.action == "SELL" and buy_price is not None:
            total += 1
            if trade.price > buy_price:
                wins += 1
            buy_price = None

    return round((wins / total * 100) if total > 0 else 0.0, 1)
