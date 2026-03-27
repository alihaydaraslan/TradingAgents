"""Backtesting engine that simulates trading based on agent signals."""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import yfinance as yf

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from .models import BacktestResult, Trade
from .metrics import calculate_sharpe, calculate_max_drawdown, calculate_win_rate


class BacktestEngine:
    """Runs backtests by generating signals for historical dates and simulating trades."""

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        config: dict = None,
        initial_capital: float = 100_000,
    ):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.config = config or DEFAULT_CONFIG.copy()
        self.initial_capital = initial_capital

        # Cache directory
        results_dir = self.config.get("results_dir", "./results")
        self.cache_dir = Path(results_dir) / "backtest_cache" / self.ticker
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_trading_days(self) -> list:
        """Get actual trading days in the date range using yfinance."""
        data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
        )
        if data.empty:
            return []

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        days = []
        for idx in data.index:
            date_str = idx.strftime("%Y-%m-%d")
            close = float(data.loc[idx, "Close"])
            days.append((date_str, close))
        return days

    def _get_cached_signal(self, date: str) -> Optional[str]:
        """Load a cached signal for a given date."""
        cache_file = self.cache_dir / f"{date}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                return data.get("signal", "HOLD")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_signal(self, date: str, signal: str, full_decision: str = ""):
        """Cache a signal to disk."""
        cache_file = self.cache_dir / f"{date}.json"
        data = {
            "ticker": self.ticker,
            "date": date,
            "signal": signal,
            "full_decision": full_decision[:5000],
            "generated_at": datetime.now().isoformat(),
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _signal_to_action(self, signal: str) -> str:
        """Map 5-tier signal to trade action."""
        signal = signal.strip().upper()
        if signal in ("BUY", "OVERWEIGHT"):
            return "BUY"
        elif signal in ("SELL", "UNDERWEIGHT"):
            return "SELL"
        return "HOLD"

    def run(
        self,
        on_progress: Optional[Callable] = None,
        interval_days: int = 5,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            on_progress: Callback(day_index, total_days, partial_result) for UI updates
            interval_days: Run analysis every N trading days (default 5 = weekly)
        """
        trading_days = self._get_trading_days()
        if not trading_days:
            return BacktestResult(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                final_value=self.initial_capital,
                total_return_pct=0.0,
            )

        # Initialize graph once
        graph = TradingAgentsGraph(
            selected_analysts=["market", "fundamentals"],
            config=self.config,
            debug=False,
        )

        # Portfolio state
        cash = self.initial_capital
        shares = 0
        trades = []
        daily_values = []
        signals_generated = 0
        signals_cached = 0

        first_price = trading_days[0][1]
        current_signal = "HOLD"

        for i, (date, close_price) in enumerate(trading_days):
            # Calculate portfolio value
            portfolio_value = cash + shares * close_price
            daily_values.append((date, portfolio_value))

            # Generate signal every N days
            if i % interval_days == 0:
                cached = self._get_cached_signal(date)
                if cached:
                    current_signal = cached
                    signals_cached += 1
                else:
                    try:
                        _, decision = graph.propagate(self.ticker, date)
                        signal = graph.process_signal(decision)
                        current_signal = signal.strip().upper()
                        self._save_signal(date, current_signal, decision)
                        signals_generated += 1
                    except Exception as e:
                        current_signal = "HOLD"
                        self._save_signal(date, "HOLD", f"Error: {e}")
                        signals_generated += 1

            # Execute trade
            action = self._signal_to_action(current_signal)

            if action == "BUY" and shares == 0 and cash > 0:
                shares = int(cash / close_price)
                cost = shares * close_price
                cash -= cost
                trades.append(Trade(date, "BUY", close_price, shares, portfolio_value, current_signal))
            elif action == "SELL" and shares > 0:
                revenue = shares * close_price
                cash += revenue
                trades.append(Trade(date, "SELL", close_price, shares, portfolio_value, current_signal))
                shares = 0

            # Progress callback
            if on_progress:
                partial = BacktestResult(
                    ticker=self.ticker,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_capital=self.initial_capital,
                    final_value=portfolio_value,
                    total_return_pct=((portfolio_value - self.initial_capital) / self.initial_capital) * 100,
                    trades=trades,
                    daily_values=daily_values,
                    signals_generated=signals_generated,
                    signals_cached=signals_cached,
                )
                on_progress(i, len(trading_days), partial)

        # Final calculations
        final_value = cash + shares * trading_days[-1][1]
        last_price = trading_days[-1][1]
        buy_hold_return = ((last_price - first_price) / first_price) * 100

        result = BacktestResult(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            final_value=round(final_value, 2),
            total_return_pct=round(((final_value - self.initial_capital) / self.initial_capital) * 100, 2),
            trades=trades,
            daily_values=daily_values,
            win_rate=calculate_win_rate(trades),
            max_drawdown=calculate_max_drawdown(daily_values),
            sharpe_ratio=calculate_sharpe(daily_values),
            buy_hold_return_pct=round(buy_hold_return, 2),
            signals_generated=signals_generated,
            signals_cached=signals_cached,
        )

        return result
