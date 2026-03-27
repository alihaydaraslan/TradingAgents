"""Multi-stock scanner that analyzes a watchlist and ranks by conviction."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from .models import ScanResult


SIGNAL_SCORES = {
    "BUY": 5,
    "OVERWEIGHT": 4,
    "HOLD": 3,
    "UNDERWEIGHT": 2,
    "SELL": 1,
}

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AMD", "MU", "AVGO",
    "CRM", "NFLX", "DKNG", "PLTR", "SOFI",
    "COIN", "SQ", "SHOP", "SNOW", "NET",
]


class StockScanner:
    """Scans multiple stocks and ranks them by trading signal conviction."""

    def __init__(
        self,
        watchlist: List[str] = None,
        date: str = None,
        config: dict = None,
    ):
        self.watchlist = [t.upper() for t in (watchlist or DEFAULT_WATCHLIST)]
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.config = config or DEFAULT_CONFIG.copy()

        # Cache directory (shared with backtest)
        results_dir = self.config.get("results_dir", "./results")
        self.cache_dir = Path(results_dir) / "backtest_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cached_signal(self, ticker: str) -> Optional[dict]:
        """Check cache for existing signal."""
        cache_file = self.cache_dir / ticker / f"{self.date}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def _save_signal(self, ticker: str, signal: str, full_decision: str):
        """Save signal to shared cache."""
        ticker_dir = self.cache_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        cache_file = ticker_dir / f"{self.date}.json"
        data = {
            "ticker": ticker,
            "date": self.date,
            "signal": signal,
            "full_decision": full_decision[:5000],
            "generated_at": datetime.now().isoformat(),
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _extract_signal(self, text: str) -> str:
        """Extract signal keyword from text."""
        text_upper = text.strip().upper()
        for signal in ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"):
            if signal in text_upper:
                return signal
        return "HOLD"

    def scan(
        self,
        on_progress: Optional[Callable] = None,
    ) -> List[ScanResult]:
        """Run analysis on all tickers in watchlist.

        Args:
            on_progress: Callback(ticker_index, total, current_results) for UI updates
        """
        results = []

        # Use fewer analysts for speed
        graph = TradingAgentsGraph(
            selected_analysts=["market", "fundamentals"],
            config=self.config,
            debug=False,
        )

        for i, ticker in enumerate(self.watchlist):
            start = time.time()

            # Check cache first
            cached = self._get_cached_signal(ticker)
            if cached:
                signal = cached.get("signal", "HOLD")
                summary = cached.get("full_decision", "")[:200]
                elapsed = 0.1  # cached
            else:
                try:
                    _, decision = graph.propagate(ticker, self.date)
                    signal_text = graph.process_signal(decision)
                    signal = self._extract_signal(signal_text)
                    summary = decision[:200] if decision else ""
                    self._save_signal(ticker, signal, decision or "")
                except Exception as e:
                    signal = "HOLD"
                    summary = f"Error: {str(e)[:150]}"
                    elapsed = time.time() - start
                    results.append(ScanResult(
                        ticker=ticker,
                        date=self.date,
                        signal=signal,
                        conviction_score=3,
                        summary=summary,
                        analysis_time_seconds=round(elapsed, 1),
                        error=str(e),
                    ))
                    if on_progress:
                        on_progress(i + 1, len(self.watchlist), results)
                    continue

            elapsed = time.time() - start
            score = SIGNAL_SCORES.get(signal, 3)

            results.append(ScanResult(
                ticker=ticker,
                date=self.date,
                signal=signal,
                conviction_score=score,
                summary=summary,
                analysis_time_seconds=round(elapsed, 1),
            ))

            if on_progress:
                on_progress(i + 1, len(self.watchlist), results)

        # Sort by conviction (BUY first, SELL last)
        results.sort(key=lambda r: r.conviction_score, reverse=True)
        return results
