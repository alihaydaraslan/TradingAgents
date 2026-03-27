"""Post-processing step to generate actionable trade parameters from portfolio manager decisions."""

from dataclasses import dataclass, asdict
from typing import Optional
import json
import re


@dataclass
class ActionPlan:
    signal: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    risk_reward_ratio: float
    time_horizon: str
    confidence: int
    reasoning: str

    def to_dict(self):
        return asdict(self)

    def to_markdown(self):
        if self.signal in ("HOLD",):
            return (
                f"## Action Plan: {self.signal}\n\n"
                f"**Recommendation:** No action needed.\n\n"
                f"**Reasoning:** {self.reasoning}"
            )

        direction = "Long" if self.signal in ("BUY", "OVERWEIGHT") else "Short/Exit"

        # Calculate risk/reward percentages
        if self.entry_price > 0:
            sl_pct = abs(self.entry_price - self.stop_loss) / self.entry_price * 100
            tp_pct = abs(self.take_profit - self.entry_price) / self.entry_price * 100
        else:
            sl_pct = tp_pct = 0

        return (
            f"## Action Plan: {self.signal}\n\n"
            f"| Parameter | Value |\n"
            f"|-----------|-------|\n"
            f"| **Direction** | {direction} |\n"
            f"| **Entry Price** | ${self.entry_price:.2f} |\n"
            f"| **Stop Loss** | ${self.stop_loss:.2f} ({sl_pct:.1f}% risk) |\n"
            f"| **Take Profit** | ${self.take_profit:.2f} ({tp_pct:.1f}% target) |\n"
            f"| **Position Size** | {self.position_size_pct:.1f}% of portfolio |\n"
            f"| **Risk/Reward** | 1:{self.risk_reward_ratio:.1f} |\n"
            f"| **Time Horizon** | {self.time_horizon} |\n"
            f"| **Confidence** | {self.confidence}/5 |\n\n"
            f"**Reasoning:** {self.reasoning}"
        )


class ActionProcessor:
    """Generates actionable trade parameters from portfolio decisions."""

    def __init__(self, llm):
        self.llm = llm

    def generate_action_plan(
        self,
        full_decision: str,
        ticker: str,
        current_price: float,
        atr: Optional[float] = None,
    ) -> ActionPlan:
        """Generate specific trade parameters from the portfolio manager's decision."""

        atr_info = f"Current ATR (14-day): ${atr:.2f}" if atr else "ATR not available"

        prompt = f"""You are a trade execution specialist. Given the portfolio manager's decision below, generate specific actionable trade parameters.

**Ticker:** {ticker}
**Current Price:** ${current_price:.2f}
**{atr_info}**

**Portfolio Manager's Decision:**
{full_decision[:3000]}

---

Respond ONLY with a valid JSON object (no markdown, no explanation) with these exact keys:
{{
    "signal": "BUY or SELL or HOLD or OVERWEIGHT or UNDERWEIGHT",
    "entry_price": <float - suggested entry price>,
    "stop_loss": <float - stop loss price>,
    "take_profit": <float - take profit target>,
    "position_size_pct": <float - percentage of portfolio to allocate, 1-10>,
    "risk_reward_ratio": <float - risk to reward ratio>,
    "time_horizon": "<short-term / medium-term / long-term>",
    "confidence": <int 1-5>,
    "reasoning": "<one sentence explaining the trade setup>"
}}

Rules:
- Stop loss should be 1-3 ATR below entry for BUY (above for SELL), or 2-5% if ATR unavailable
- Take profit should give at least 2:1 reward-to-risk ratio
- Position size: 1-3% for low confidence, 3-5% for medium, 5-10% for high
- For HOLD: set entry=current_price, stop_loss=0, take_profit=0, position_size_pct=0, risk_reward_ratio=0"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)

            return ActionPlan(
                signal=str(data.get("signal", "HOLD")).upper(),
                entry_price=float(data.get("entry_price", current_price)),
                stop_loss=float(data.get("stop_loss", 0)),
                take_profit=float(data.get("take_profit", 0)),
                position_size_pct=float(data.get("position_size_pct", 0)),
                risk_reward_ratio=float(data.get("risk_reward_ratio", 0)),
                time_horizon=str(data.get("time_horizon", "medium-term")),
                confidence=int(data.get("confidence", 3)),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as e:
            # Fallback: generate basic plan from signal
            return self._fallback_plan(full_decision, current_price, atr)

    def _fallback_plan(self, full_decision: str, current_price: float, atr: Optional[float]) -> ActionPlan:
        """Generate a basic plan when LLM parsing fails."""
        signal = "HOLD"
        for s in ("BUY", "SELL", "OVERWEIGHT", "UNDERWEIGHT"):
            if s in full_decision.upper():
                signal = s
                break

        if atr and atr > 0:
            sl_distance = atr * 2
        else:
            sl_distance = current_price * 0.03

        if signal in ("BUY", "OVERWEIGHT"):
            stop_loss = current_price - sl_distance
            take_profit = current_price + (sl_distance * 2.5)
        elif signal in ("SELL", "UNDERWEIGHT"):
            stop_loss = current_price + sl_distance
            take_profit = current_price - (sl_distance * 2.5)
        else:
            stop_loss = 0
            take_profit = 0

        rr = (abs(take_profit - current_price) / sl_distance) if sl_distance > 0 and take_profit > 0 else 0

        return ActionPlan(
            signal=signal,
            entry_price=current_price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            position_size_pct=3.0,
            risk_reward_ratio=round(rr, 1),
            time_horizon="medium-term",
            confidence=3,
            reasoning="Auto-generated plan based on ATR-based risk levels.",
        )
