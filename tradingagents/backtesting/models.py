from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Trade:
    date: str
    action: str  # BUY, SELL, HOLD
    price: float
    shares: int
    portfolio_value: float
    signal_raw: str = ""


@dataclass
class BacktestResult:
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    trades: List[Trade] = field(default_factory=list)
    daily_values: List[Tuple[str, float]] = field(default_factory=list)
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    buy_hold_return_pct: float = 0.0
    signals_generated: int = 0
    signals_cached: int = 0
