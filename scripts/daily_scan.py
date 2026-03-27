"""Daily automated stock scanner.

Run manually:
    python scripts/daily_scan.py

Schedule with Windows Task Scheduler:
    python scripts/daily_scan.py --watchlist config/watchlist.txt --output-dir results/daily

"""

import sys
import io
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.scanner.scanner import StockScanner, DEFAULT_WATCHLIST


def load_watchlist(path: str) -> list:
    """Load tickers from a text file (one per line)."""
    with open(path, "r") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Daily stock scanner")
    parser.add_argument("--watchlist", default=None, help="Path to watchlist file")
    parser.add_argument("--output-dir", default="results/daily", help="Output directory")
    parser.add_argument("--date", default=None, help="Analysis date (YYYY-MM-DD), defaults to yesterday")
    args = parser.parse_args()

    # Date
    if args.date:
        scan_date = args.date
    else:
        scan_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Watchlist
    if args.watchlist and os.path.exists(args.watchlist):
        watchlist = load_watchlist(args.watchlist)
    else:
        watchlist = DEFAULT_WATCHLIST

    # Config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "ollama"
    config["deep_think_llm"] = "qwen3:14b"
    config["quick_think_llm"] = "qwen3:14b"
    config["backend_url"] = "http://localhost:11434/v1"
    config["max_debate_rounds"] = 1

    print(f"Starting daily scan for {scan_date}")
    print(f"Watchlist: {', '.join(watchlist)}")
    print(f"Total: {len(watchlist)} tickers")
    print("-" * 50)

    scanner = StockScanner(watchlist=watchlist, date=scan_date, config=config)

    def progress(idx, total, results):
        latest = results[-1]
        print(f"[{idx}/{total}] {latest.ticker}: {latest.signal} (score: {latest.conviction_score}) - {latest.analysis_time_seconds}s")

    results = scanner.scan(on_progress=progress)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_file = output_dir / f"{scan_date}.json"
    json_data = {
        "date": scan_date,
        "generated_at": datetime.now().isoformat(),
        "total_tickers": len(results),
        "results": [
            {
                "ticker": r.ticker,
                "signal": r.signal,
                "conviction": r.conviction_score,
                "summary": r.summary,
                "time_seconds": r.analysis_time_seconds,
                "error": r.error,
            }
            for r in results
        ],
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 50)
    print(f"DAILY SCAN RESULTS - {scan_date}")
    print("=" * 50)

    buys = [r for r in results if r.signal in ("BUY", "OVERWEIGHT")]
    sells = [r for r in results if r.signal in ("SELL", "UNDERWEIGHT")]
    holds = [r for r in results if r.signal == "HOLD"]

    if buys:
        print(f"\nBUY/OVERWEIGHT ({len(buys)}):")
        for r in buys:
            print(f"  {r.ticker:6s} - {r.signal}")

    if holds:
        print(f"\nHOLD ({len(holds)}):")
        for r in holds:
            print(f"  {r.ticker:6s}")

    if sells:
        print(f"\nSELL/UNDERWEIGHT ({len(sells)}):")
        for r in sells:
            print(f"  {r.ticker:6s} - {r.signal}")

    print(f"\nResults saved to: {json_file}")


if __name__ == "__main__":
    main()
