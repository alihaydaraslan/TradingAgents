from dataclasses import dataclass


@dataclass
class ScanResult:
    ticker: str
    date: str
    signal: str  # BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL
    conviction_score: int  # 1-5
    summary: str  # First part of portfolio manager decision
    analysis_time_seconds: float
    error: str = ""
