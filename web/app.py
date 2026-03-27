import sys
import io
import os
import time
import json
import datetime
from collections import deque
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.graph.action_processor import ActionProcessor
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.backtesting.engine import BacktestEngine
from tradingagents.scanner.scanner import StockScanner, DEFAULT_WATCHLIST
from cli.stats_handler import StatsCallbackHandler


# --- Shared Config ---

def make_config(depth=1):
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "ollama"
    config["deep_think_llm"] = "qwen3:14b"
    config["quick_think_llm"] = "qwen3:14b"
    config["backend_url"] = "http://localhost:11434/v1"
    config["max_debate_rounds"] = int(depth)
    config["max_risk_discuss_rounds"] = int(depth)
    return config


# --- Message Buffer ---

ANALYST_MAPPING = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

FIXED_AGENTS = {
    "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
    "Trading Team": ["Trader"],
    "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
    "Portfolio Management": ["Portfolio Manager"],
}

SECTION_TITLES = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Team Plan",
    "final_trade_decision": "Portfolio Management Decision",
}

STATUS_ICONS = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}


def extract_content_string(content):
    if content is None or content == "":
        return None
    if isinstance(content, str):
        s = content.strip()
        return s if s else None
    if isinstance(content, dict):
        text = content.get("text", "")
        return text.strip() if text and text.strip() else None
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", "").strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        result = " ".join(t for t in parts if t)
        return result if result else None
    s = str(content).strip()
    return s if s else None


def classify_message_type(message):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    content = extract_content_string(getattr(message, "content", None))
    if isinstance(message, HumanMessage):
        return ("Control" if content and content.strip() == "Continue" else "User", content)
    if isinstance(message, ToolMessage):
        return ("Data", content)
    if isinstance(message, AIMessage):
        return ("Agent", content)
    return ("System", content)


class WebMessageBuffer:
    def __init__(self):
        self.messages = deque(maxlen=100)
        self.tool_calls = deque(maxlen=100)
        self.agent_status = {}
        self.report_sections = {}
        self.selected_analysts = []
        self._last_message_id = None

    def init_for_analysis(self, selected_analysts):
        self.selected_analysts = [a.lower() for a in selected_analysts]
        self.agent_status = {}
        for ak in self.selected_analysts:
            if ak in ANALYST_MAPPING:
                self.agent_status[ANALYST_MAPPING[ak]] = "pending"
        for agents in FIXED_AGENTS.values():
            for agent in agents:
                self.agent_status[agent] = "pending"
        analyst_map = {"market_report": "market", "sentiment_report": "social",
                       "news_report": "news", "fundamentals_report": "fundamentals"}
        self.report_sections = {}
        for section in SECTION_TITLES:
            ak = analyst_map.get(section)
            if ak is None or ak in self.selected_analysts:
                self.report_sections[section] = None
        self.messages.clear()
        self.tool_calls.clear()
        self._last_message_id = None

    def add_message(self, t, c):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((ts, t, c))

    def add_tool_call(self, name, args):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((ts, name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status

    def update_report_section(self, section, content):
        if section in self.report_sections:
            self.report_sections[section] = content

    def get_status_markdown(self):
        lines = ["| Agent | Status |", "|-------|--------|"]
        for agent, status in self.agent_status.items():
            lines.append(f"| {agent} | {STATUS_ICONS.get(status, '⬜')} {status} |")
        return "\n".join(lines)

    def get_messages_text(self):
        lines = []
        for ts, t, c in list(self.messages)[-20:]:
            short = (c[:120] + "...") if c and len(c) > 120 else (c or "")
            lines.append(f"[{ts}] {t}: {short}")
        return "\n".join(lines)

    def get_reports_markdown(self):
        parts = []
        for section, content in self.report_sections.items():
            if content:
                parts.append(f"## {SECTION_TITLES.get(section, section)}\n\n{content}")
        return "\n\n---\n\n".join(parts) if parts else "*Analysis in progress...*"


def update_analyst_statuses(buf, chunk):
    report_keys = {"market_report": "Market Analyst", "sentiment_report": "Social Analyst",
                   "news_report": "News Analyst", "fundamentals_report": "Fundamentals Analyst"}
    selected = [(k, a) for k, a in report_keys.items() if k in buf.report_sections]
    found_active = False
    for key, agent in selected:
        if bool(chunk.get(key)) or bool(buf.report_sections.get(key)):
            buf.update_agent_status(agent, "completed")
        elif not found_active:
            buf.update_agent_status(agent, "in_progress")
            found_active = True
    if not found_active and selected and buf.agent_status.get("Bull Researcher") == "pending":
        buf.update_agent_status("Bull Researcher", "in_progress")


# ============================================================
# TAB 1: Single Stock Analysis + Actionable Signals
# ============================================================

def run_analysis(ticker, date_str, analysts, depth):
    if not ticker or not ticker.strip():
        yield ("", "", "Please enter a ticker symbol.", "", "", "")
        return

    ticker = ticker.strip().upper()
    analyst_list = [a.lower() for a in analysts] or ["market", "social", "news", "fundamentals"]
    config = make_config(depth)
    stats_handler = StatsCallbackHandler()
    buf = WebMessageBuffer()
    buf.init_for_analysis(analyst_list)

    yield (buf.get_status_markdown(), "*Initializing...*", "Starting...", "", "", "")

    try:
        graph = TradingAgentsGraph(analyst_list, config=config, debug=True, callbacks=[stats_handler])
    except Exception as e:
        yield ("", "", f"Error: {e}", "", "", "")
        return

    buf.add_message("System", f"Analyzing {ticker} on {date_str}")
    start_time = time.time()
    init_state = graph.propagator.create_initial_state(ticker, date_str)
    args = graph.propagator.get_graph_args(callbacks=[stats_handler])

    trace = []
    try:
        for chunk in graph.graph.stream(init_state, **args):
            if len(chunk.get("messages", [])) > 0:
                last_msg = chunk["messages"][-1]
                msg_id = getattr(last_msg, "id", None)
                if msg_id != buf._last_message_id:
                    buf._last_message_id = msg_id
                    mt, mc = classify_message_type(last_msg)
                    if mc and mc.strip():
                        buf.add_message(mt, mc)
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            n, a = (tc["name"], tc["args"]) if isinstance(tc, dict) else (tc.name, tc.args)
                            buf.add_tool_call(n, a)

            update_analyst_statuses(buf, chunk)
            for key in ["market_report", "sentiment_report", "news_report", "fundamentals_report"]:
                if chunk.get(key):
                    buf.update_report_section(key, chunk[key])

            if chunk.get("investment_debate_state"):
                ds = chunk["investment_debate_state"]
                bull, bear = ds.get("bull_history", "").strip(), ds.get("bear_history", "").strip()
                judge = ds.get("judge_decision", "").strip()
                if bull or bear:
                    buf.update_agent_status("Bull Researcher", "in_progress")
                    buf.update_agent_status("Bear Researcher", "in_progress")
                if bull: buf.update_report_section("investment_plan", f"### Bull Researcher\n{bull}")
                if bear: buf.update_report_section("investment_plan", f"### Bear Researcher\n{bear}")
                if judge:
                    buf.update_report_section("investment_plan", f"### Research Manager Decision\n{judge}")
                    for a in ["Bull Researcher", "Bear Researcher", "Research Manager"]:
                        buf.update_agent_status(a, "completed")
                    buf.update_agent_status("Trader", "in_progress")

            if chunk.get("trader_investment_plan"):
                buf.update_report_section("trader_investment_plan", chunk["trader_investment_plan"])
                buf.update_agent_status("Trader", "completed")
                buf.update_agent_status("Aggressive Analyst", "in_progress")

            if chunk.get("risk_debate_state"):
                rs = chunk["risk_debate_state"]
                for key, agent in [("aggressive_history", "Aggressive Analyst"),
                                   ("conservative_history", "Conservative Analyst"),
                                   ("neutral_history", "Neutral Analyst")]:
                    val = rs.get(key, "").strip()
                    if val:
                        buf.update_agent_status(agent, "in_progress")
                        buf.update_report_section("final_trade_decision", f"### {agent}\n{val}")
                judge = rs.get("judge_decision", "").strip()
                if judge:
                    buf.update_report_section("final_trade_decision", f"### Portfolio Manager Decision\n{judge}")
                    for a in ["Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Portfolio Manager"]:
                        buf.update_agent_status(a, "completed")

            elapsed = time.time() - start_time
            stats = stats_handler.get_stats()
            stats_text = (f"**Time:** {int(elapsed//60)}m {int(elapsed%60)}s | "
                          f"**LLM:** {stats['llm_calls']} | **Tools:** {stats['tool_calls']} | "
                          f"**Tokens:** {stats['tokens_in']:,} in / {stats['tokens_out']:,} out")
            trace.append(chunk)
            yield (buf.get_status_markdown(), buf.get_reports_markdown(), buf.get_messages_text(), stats_text, "", "")

    except Exception as e:
        yield (buf.get_status_markdown(), buf.get_reports_markdown(),
               buf.get_messages_text() + f"\n\nERROR: {e}", "", f"Error: {e}", "")
        return

    # Final decision + action plan
    decision_md = ""
    action_md = ""
    if trace:
        final_state = trace[-1]
        final_signal = final_state.get("final_trade_decision", "")
        if final_signal:
            try:
                decision = graph.process_signal(final_signal)
            except Exception:
                decision = final_signal[:200]

            d = decision.strip().upper() if decision else "N/A"
            if "BUY" in d:
                decision_md = f"# 🟢 {d}"
            elif "SELL" in d:
                decision_md = f"# 🔴 {d}"
            else:
                decision_md = f"# 🟡 {d}"

            # Generate action plan
            try:
                import yfinance as yf
                tk = yf.Ticker(ticker)
                hist = tk.history(period="5d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
                action_proc = ActionProcessor(graph.quick_thinking_llm)
                plan = action_proc.generate_action_plan(final_signal, ticker, current_price)
                action_md = plan.to_markdown()
            except Exception as e:
                action_md = f"*Could not generate action plan: {e}*"

    elapsed = time.time() - start_time
    stats = stats_handler.get_stats()
    stats_text = (f"**Time:** {int(elapsed//60)}m {int(elapsed%60)}s | "
                  f"**LLM:** {stats['llm_calls']} | **Tools:** {stats['tool_calls']} | "
                  f"**Tokens:** {stats['tokens_in']:,} in / {stats['tokens_out']:,} out")

    yield (buf.get_status_markdown(), buf.get_reports_markdown(), buf.get_messages_text(),
           stats_text, decision_md, action_md)


# ============================================================
# TAB 2: Backtesting
# ============================================================

def run_backtest(ticker, start_date, end_date, initial_capital, interval_days):
    if not ticker or not ticker.strip():
        yield ("", "", "")
        return

    ticker = ticker.strip().upper()
    config = make_config(1)

    yield (f"*Initializing backtest for {ticker}...*", "", "")

    engine = BacktestEngine(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        config=config,
        initial_capital=float(initial_capital),
    )

    last_progress = ""

    def on_progress(idx, total, partial):
        nonlocal last_progress
        pct = partial.total_return_pct
        last_progress = (
            f"**Progress:** {idx}/{total} days | "
            f"**Return:** {pct:+.1f}% | "
            f"**Signals:** {partial.signals_generated} generated, {partial.signals_cached} cached"
        )

    # This is blocking but yields progress via callback
    result = engine.run(on_progress=on_progress, interval_days=int(interval_days))

    # Build metrics summary
    metrics_md = (
        f"## Backtest Results: {ticker}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| **Period** | {result.start_date} to {result.end_date} |\n"
        f"| **Initial Capital** | ${result.initial_capital:,.0f} |\n"
        f"| **Final Value** | ${result.final_value:,.0f} |\n"
        f"| **Total Return** | {result.total_return_pct:+.1f}% |\n"
        f"| **Buy & Hold Return** | {result.buy_hold_return_pct:+.1f}% |\n"
        f"| **Win Rate** | {result.win_rate:.1f}% |\n"
        f"| **Max Drawdown** | {result.max_drawdown:.1f}% |\n"
        f"| **Sharpe Ratio** | {result.sharpe_ratio:.2f} |\n"
        f"| **Total Trades** | {len(result.trades)} |\n"
        f"| **Signals Generated** | {result.signals_generated} |\n"
        f"| **Signals Cached** | {result.signals_cached} |\n"
    )

    # Build equity curve (text-based)
    if result.daily_values:
        metrics_md += "\n\n## Equity Curve\n\n"
        step = max(1, len(result.daily_values) // 20)
        metrics_md += "```\n"
        max_val = max(v for _, v in result.daily_values)
        for i in range(0, len(result.daily_values), step):
            date, val = result.daily_values[i]
            bar_len = int((val / max_val) * 40) if max_val > 0 else 0
            metrics_md += f"{date} | {'█' * bar_len} ${val:,.0f}\n"
        metrics_md += "```\n"

    # Trades log
    trades_md = "## Trade Log\n\n| Date | Action | Price | Shares | Portfolio |\n|------|--------|-------|--------|----------|\n"
    for t in result.trades:
        trades_md += f"| {t.date} | {t.action} | ${t.price:.2f} | {t.shares} | ${t.portfolio_value:,.0f} |\n"

    if not result.trades:
        trades_md += "| - | No trades executed | - | - | - |\n"

    yield (metrics_md, trades_md, last_progress)


# ============================================================
# TAB 3: Scanner
# ============================================================

def run_scanner(watchlist_text, date_str):
    if not watchlist_text.strip():
        yield ("", "Please enter at least one ticker.")
        return

    tickers = [t.strip().upper() for t in watchlist_text.strip().split("\n") if t.strip() and not t.startswith("#")]
    config = make_config(1)

    yield ("*Starting scan...*", f"Scanning {len(tickers)} tickers...")

    scanner = StockScanner(watchlist=tickers, date=date_str, config=config)

    progress_text = ""

    def on_progress(idx, total, results):
        nonlocal progress_text
        latest = results[-1]
        progress_text = f"**[{idx}/{total}]** {latest.ticker}: {latest.signal} ({latest.analysis_time_seconds}s)"

    results = scanner.scan(on_progress=on_progress)

    # Build results table
    signal_colors = {"BUY": "🟢", "OVERWEIGHT": "🟢", "HOLD": "🟡", "UNDERWEIGHT": "🔴", "SELL": "🔴"}

    md = "## Scan Results\n\n"
    md += "| # | Ticker | Signal | Score | Time | Summary |\n"
    md += "|---|--------|--------|-------|------|--------|\n"
    for i, r in enumerate(results, 1):
        icon = signal_colors.get(r.signal, "⬜")
        summary = r.summary[:80] + "..." if len(r.summary) > 80 else r.summary
        summary = summary.replace("|", "/").replace("\n", " ")
        md += f"| {i} | **{r.ticker}** | {icon} {r.signal} | {r.conviction_score}/5 | {r.analysis_time_seconds}s | {summary} |\n"

    # Summary counts
    buys = sum(1 for r in results if r.signal in ("BUY", "OVERWEIGHT"))
    sells = sum(1 for r in results if r.signal in ("SELL", "UNDERWEIGHT"))
    holds = sum(1 for r in results if r.signal == "HOLD")
    md += f"\n**Summary:** {buys} Buy | {holds} Hold | {sells} Sell"

    yield (md, f"Scan complete. {len(results)} tickers analyzed.")


# ============================================================
# TAB 4: Daily Reports
# ============================================================

def load_daily_reports():
    reports_dir = Path("results/daily")
    if not reports_dir.exists():
        return "*No daily reports found. Run a scan first.*"

    files = sorted(reports_dir.glob("*.json"), reverse=True)
    if not files:
        return "*No daily reports found.*"

    md = "## Saved Daily Reports\n\n"
    for f in files[:10]:
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            date = data.get("date", f.stem)
            total = data.get("total_tickers", 0)
            results = data.get("results", [])
            buys = sum(1 for r in results if r["signal"] in ("BUY", "OVERWEIGHT"))
            sells = sum(1 for r in results if r["signal"] in ("SELL", "UNDERWEIGHT"))
            md += f"### {date} ({total} tickers)\n"
            md += f"🟢 {buys} Buy | 🔴 {sells} Sell\n\n"
            for r in results[:5]:
                icon = "🟢" if r["signal"] in ("BUY", "OVERWEIGHT") else "🔴" if r["signal"] in ("SELL", "UNDERWEIGHT") else "🟡"
                md += f"- {icon} **{r['ticker']}**: {r['signal']}\n"
            if len(results) > 5:
                md += f"- *...and {len(results) - 5} more*\n"
            md += "\n---\n\n"
        except Exception:
            continue

    return md


# ============================================================
# Gradio App
# ============================================================

def get_default_date():
    return (datetime.date.today() - datetime.timedelta(days=1)).isoformat()


with gr.Blocks(title="TradingAgents") as app:

    gr.Markdown("# TradingAgents Dashboard")

    with gr.Tabs():

        # --- Tab 1: Analysis ---
        with gr.Tab("Analysis"):
            with gr.Row():
                ticker_input = gr.Textbox(label="Ticker", value="MU", placeholder="AAPL, NVDA...", scale=1)
                date_input = gr.Textbox(label="Date", value=get_default_date(), scale=1)
                analysts_input = gr.CheckboxGroup(
                    ["market", "social", "news", "fundamentals"],
                    value=["market", "social", "news", "fundamentals"],
                    label="Analysts", scale=2)
                depth_input = gr.Slider(1, 3, value=1, step=1, label="Depth", scale=1)

            run_btn = gr.Button("Run Analysis", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=1):
                    a_status = gr.Markdown(value="*Ready*")
                with gr.Column(scale=2):
                    a_decision = gr.Markdown()

            a_action = gr.Markdown()

            with gr.Tabs():
                with gr.Tab("Reports"):
                    a_reports = gr.Markdown(value="*Reports will appear here...*")
                with gr.Tab("Live Messages"):
                    a_messages = gr.Textbox(label="Messages", lines=12, interactive=False)

            a_stats = gr.Markdown()

            run_btn.click(
                fn=run_analysis,
                inputs=[ticker_input, date_input, analysts_input, depth_input],
                outputs=[a_status, a_reports, a_messages, a_stats, a_decision, a_action],
            )

        # --- Tab 2: Backtest ---
        with gr.Tab("Backtest"):
            gr.Markdown("### Backtest Trading Signals")
            with gr.Row():
                bt_ticker = gr.Textbox(label="Ticker", value="MU", scale=1)
                bt_start = gr.Textbox(label="Start Date", value="2025-01-01", scale=1)
                bt_end = gr.Textbox(label="End Date", value=get_default_date(), scale=1)
                bt_capital = gr.Number(label="Initial Capital ($)", value=100000, scale=1)
                bt_interval = gr.Slider(1, 20, value=5, step=1, label="Signal Every N Days", scale=1)

            bt_btn = gr.Button("Run Backtest", variant="primary", size="lg")
            bt_progress = gr.Markdown()
            bt_metrics = gr.Markdown()
            bt_trades = gr.Markdown()

            bt_btn.click(
                fn=run_backtest,
                inputs=[bt_ticker, bt_start, bt_end, bt_capital, bt_interval],
                outputs=[bt_metrics, bt_trades, bt_progress],
            )

        # --- Tab 3: Scanner ---
        with gr.Tab("Scanner"):
            gr.Markdown("### Multi-Stock Scanner")
            with gr.Row():
                sc_watchlist = gr.Textbox(
                    label="Watchlist (one per line)",
                    value="\n".join(DEFAULT_WATCHLIST[:10]),
                    lines=10, scale=2)
                with gr.Column(scale=1):
                    sc_date = gr.Textbox(label="Date", value=get_default_date())
                    sc_btn = gr.Button("Run Scan", variant="primary", size="lg")

            sc_progress = gr.Markdown()
            sc_results = gr.Markdown()

            sc_btn.click(
                fn=run_scanner,
                inputs=[sc_watchlist, sc_date],
                outputs=[sc_results, sc_progress],
            )

        # --- Tab 4: Daily Reports ---
        with gr.Tab("Daily Reports"):
            gr.Markdown("### Saved Daily Scan Reports")
            dr_refresh = gr.Button("Refresh", size="sm")
            dr_content = gr.Markdown(value=load_daily_reports())
            dr_refresh.click(fn=load_daily_reports, outputs=[dr_content])


def main():
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
