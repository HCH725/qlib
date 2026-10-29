"""Config-driven formal research cell: frozen signal + house DCA overlay x research_leverage.

One cell is one frozen identity (symbol, timeframe, research_leverage) of the
round0 Formal Baseline. This file is the frozen cell entrypoint: the
materializer copies it byte-identically next to each cell's
effective_config.json, and the thin transport runs it inside the fixed Qlib
runtime as ``python <cell>/run_cell.entrypoint.py`` with ``QLIB_OUT_DIR`` set.

Boundary, deliberately narrow:

- This file is a coverage + shared execution shell only: symbol, timeframe,
  leverage, provider, house overlay, evidence and MLflow. It carries no
  strategy rule. It never names a strategy, a family, an indicator or a
  parameter.
- Strategy signal semantics come from the frozen, run-scoped strategy
  artifact: one Python file with one callable ``signal()``, loaded from the
  cell directory and bound to the sha256 declared in the cell config.
- ``research_leverage`` multiplies the house quote allocation only (base quote
  allocation x research_leverage). It is a research exposure multiplier. It is
  not a Binance liquidation, isolated-margin, funding or account-equity model;
  canonical economics stay with the Node13 Hummingbot certification lane.
- DCA stage semantics reuse the exported Qlib <-> Hummingbot DCA path
  (``qlib.backtest.backtest_hummingbot_dca``) over the frozen provider.
- Every overlay parameter comes from the cell config. This file invents none:
  a missing approved value is a fail-closed error, never a default.

The strategy artifact contract, deliberately one function::

    def signal(*, high, low, close, volume, times, params, window_start, window_end):
        # Return (rows, entry_ts).
        # rows: list of dicts, each carrying at least
        #       datetime, high, low, close, entry_transition (0|1),
        #       exit_transition (0|1).
        # entry_ts: datetime of the artifact's single entry, or None. It stays
        #           for ABI compatibility and must name one of the transition
        #           bars; the run replays every transition, not just this one.

``volume`` is a float array, or None when the frozen provider has no volume
field; an artifact that needs volume must fail closed on None.

Full-strategy replay, one long cycle at a time::

    Each ``entry_transition`` opens one house DCA cycle priced from that bar's
    close, the next ``exit_transition`` is its close boundary, and a cycle still
    open on the last evaluation bar is forced flat there. Overlay TP/SL wins a
    tie on the same bar, then the strategy exit. An entry while the source is
    still long, an exit while it is flat, or a source that is long on the first
    evaluation bar fails closed instead of being guessed. Between cycles the
    account is flat; while the overlay has closed but the source is still long
    the account stays flat too, so the overlay closing early never invents a new
    source entry. ``filled_amount_quote`` is a quote allocation, never equity.

Evidence written per cell::

    summary.json   cycle plan and per-cycle stage facts
    signals.csv    the artifact's rows inside the frozen evaluation window
    trades.csv     one row per replayed cycle: entry/exit/close, stages, PnL
    events.csv     one row per replayed stage (the pinned transport reads it)
    equity.csv     the whole evaluation window, flat bars included
    metrics.json   fee-inclusive performance metrics plus their definitions
    manifest.json  sha256 of every non-MLflow evidence file in this directory

Verbs::

    python research_cell.py run                     # run this cell (default)
    python research_cell.py materialize --spec S.json --root /results/_cells

ponytail: stdlib only at import time. numpy/pandas/qlib/mlflow load inside the
run path, so the deterministic planner (identities, config validation, stage
scaling, materialization) stays testable without the runtime.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

SCHEMA = "qlib-formal-research-cell-v1"
#: Frozen round0 coverage. Symbols/timeframes/leverages are the 42-cell set.
#: Research domain is fixed here: a strategy artifact never widens it.
CELL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT")
CELL_TIMEFRAMES = ("1h", "4h", "1d")
CELL_LEVERAGES = (3, 5)
QLIB_FREQ = {"1h": "60min", "4h": "240min", "1d": "day"}
ENTRYPOINT_NAME = "run_cell.entrypoint.py"
CONFIG_NAME = "effective_config.json"
#: The single run-scoped strategy artifact: one file, one function.
SIGNAL_ARTIFACT_NAME = "strategy_signal.py"
SIGNAL_FUNCTION = "signal"
SIGNAL_ROW_KEYS = ("datetime", "high", "low", "close", "entry_transition", "exit_transition")
HEX64 = re.compile(r"^[a-f0-9]{64}$")

#: The verified primitives every frozen cell declares. One reusable name per
#: capability, never a per-symbol/per-timeframe entry: the cell reads the same
#: Binance USD-M OHLCV provider through the same Qlib bar-close decision path
#: for any covered symbol and timeframe. Symbol/timeframe coverage is a
#: measured provider-readiness fact, not a capability. A cell that needs a
#: genuinely different execution/data/timing primitive must add it here only
#: after one representative validation, which is an engineering gate, not a
#: second strategy admission.
CAPABILITIES = {
    "execution": ["qlib-native-backtest"],
    "data": ["binance-usdm-ohlcv"],
    "timing": ["bar-close"],
    #: The shared execution shell opens one long leg through the house DCA
    #: overlay. A short leg or an atomic reversal is a genuinely different
    #: execution primitive, so it is not in this set until it is built. A
    #: strategy that declares it must HOLD instead of being silently reduced
    #: to the side the shell happens to support.
    "sides": ["long"],
}


#: Required house-overlay keys. No default: an unapproved value must hold.
#: take_profit_pct/stop_loss_pct may be an explicit null (disabled), never absent.
OVERLAY_KEYS = (
    "base_order_quote_pct",
    "safety_order_quote_pct",
    "safety_order_levels",
    "so_spacing_pct",
    "take_profit_pct",
    "stop_loss_pct",
    "trade_cost",
    "initial_capital_quote",
)
#: Keys that must carry a usable number, not just be declared.
OVERLAY_NUMERIC_KEYS = (
    "base_order_quote_pct",
    "safety_order_quote_pct",
    "safety_order_levels",
    "so_spacing_pct",
    "trade_cost",
    "initial_capital_quote",
)
COST_MODEL_KEYS = ("fees_per_side", "slippage", "funding", "approved")


def cell_id(symbol, timeframe, research_leverage):
    """Return the stable identity string for one cell."""
    return "%s-%s-x%s" % (symbol, timeframe, research_leverage)


def campaign_identities():
    """Return the frozen 42 identities in a stable order (no sweep)."""
    return [
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "research_leverage": leverage,
            "cell_id": cell_id(symbol, timeframe, leverage),
        }
        for symbol in CELL_SYMBOLS
        for timeframe in CELL_TIMEFRAMES
        for leverage in CELL_LEVERAGES
    ]


def problems_in_config(cfg):
    """Return the list of fail-closed config problems. Empty means runnable."""
    bad = []
    if not isinstance(cfg, dict):
        return ["effective_config is not a JSON object"]
    candidate = cfg.get("candidate")
    if not isinstance(candidate, dict) or not candidate.get("path") or not candidate.get("git_blob_sha"):
        bad.append("candidate identity is missing")
    runtime = cfg.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("commit"):
        bad.append("runtime identity is missing")
    execution = cfg.get("execution")
    if not isinstance(execution, dict) or not execution.get("entrypoint"):
        bad.append("execution identity is missing")
    data = cfg.get("data")
    if not isinstance(data, dict) or not data.get("provider_ref"):
        bad.append("data provider reference is missing")
    strategy = cfg.get("strategy")
    if not isinstance(strategy, dict):
        bad.append("strategy block is missing")
    else:
        if strategy.get("symbol") not in CELL_SYMBOLS:
            bad.append("strategy symbol is outside the frozen 42-cell set")
        if strategy.get("timeframe") not in CELL_TIMEFRAMES:
            bad.append("strategy timeframe is outside the frozen 42-cell set")
        if not strategy.get("window_start_utc") or not strategy.get("window_end_utc_inclusive"):
            bad.append("strategy window bounds are missing")
        params = strategy.get("params")
        if params is not None and not isinstance(params, dict):
            bad.append("strategy params is not a JSON object")
        warmup = strategy.get("warmup_days", 0)
        if not isinstance(warmup, (int, float)) or isinstance(warmup, bool) or warmup < 0:
            bad.append("strategy warmup_days must be a non-negative number")
        artifact = strategy.get("artifact")
        if not isinstance(artifact, dict):
            bad.append("strategy artifact is not declared")
        else:
            if not isinstance(artifact.get("ref"), str) or not artifact["ref"].endswith(".py"):
                bad.append("strategy artifact ref is not a python path")
            if not HEX64.match(str(artifact.get("sha256"))):
                bad.append("strategy artifact sha256 is not a 64-hex digest")
        # A declared side the shell cannot execute is a genuine capability gap,
        # not a smaller version of the same strategy. Fail closed here; the
        # readiness gate reports it as ENGINEERING_HOLD.
        sides = strategy.get("sides")
        if not isinstance(sides, list) or not sides:
            bad.append("strategy execution sides are not declared")
        else:
            outside = [side for side in sides if side not in CAPABILITIES["sides"]]
            if outside:
                bad.append("strategy requires an execution side outside the engine capability: " + ", ".join(map(str, outside)))
    capabilities = cfg.get("capabilities")
    if not isinstance(capabilities, dict) or not capabilities:
        bad.append("capability declaration is missing")
    else:
        for bucket, values in capabilities.items():
            values = values if isinstance(values, list) else [values]
            for value in values:
                if bucket not in CAPABILITIES or value not in CAPABILITIES[bucket]:
                    bad.append("capability is outside the engine's frozen set: %s:%s" % (bucket, value))
    leverage = cfg.get("research_leverage")
    if leverage not in CELL_LEVERAGES:
        bad.append("research_leverage is outside the frozen set")
    overlay = cfg.get("house_overlay")
    if not isinstance(overlay, dict):
        bad.append("house_overlay block is missing")
    else:
        for key in OVERLAY_KEYS:
            if key not in overlay:
                bad.append("house_overlay.%s has no approved value" % key)
        for key in OVERLAY_NUMERIC_KEYS:
            if key in overlay and not isinstance(overlay[key], (int, float)):
                bad.append("house_overlay.%s is not a usable number" % key)
    cost = cfg.get("research_cost_model")
    if not isinstance(cost, dict):
        bad.append("research cost model is not declared")
    else:
        for key in COST_MODEL_KEYS:
            if key not in cost:
                bad.append("research cost model does not declare %s" % key)
    return bad


def cost_model_claimable(cfg):
    """True only when the cell config declares an approved research cost model."""
    cost = cfg.get("research_cost_model")
    return isinstance(cost, dict) and cost.get("approved") is True


def house_dca_amounts(base_pct, safety_pct, levels, equity_quote, leverage):
    """Return the house DCA quote allocations scaled by research_leverage."""
    base = Decimal(str(base_pct))
    safety = Decimal(str(safety_pct))
    equity = Decimal(str(equity_quote))
    factor = Decimal(int(leverage))
    return [(base if level == 0 else safety) * equity * factor for level in range(int(levels) + 1)]


def stage_prices(entry_price, so_spacing_pct, levels):
    """Return the DCA stage prices: base order then evenly spaced safety orders."""
    entry = Decimal(str(entry_price))
    spacing = Decimal(str(so_spacing_pct))
    return [entry * (Decimal(1) - spacing * level) for level in range(int(levels) + 1)]


def dca_simulator():
    """Resolve the exported Qlib <-> Hummingbot DCA path from the installed package."""
    from qlib.backtest import backtest_hummingbot_dca

    return backtest_hummingbot_dca


def dca_overlay_ledger(df, prices, amounts_quote, take_profit, stop_loss, trade_cost, simulator=None):
    """Run the house DCA overlay on one cell frame and return (ledger, close_type)."""
    simulator = simulator or dca_simulator()
    return simulator(
        df,
        prices=prices,
        amounts_quote=amounts_quote,
        take_profit=take_profit,
        stop_loss=stop_loss,
        trade_cost=trade_cost,
        side="BUY",
        mode="MAKER",
    )


# --------------------------------------------------------------- full replay
#
# One materialized cell replays the artifact's whole long-only transition
# series. The DCA ledger is the only source of a cycle's mark to market: its
# ``filled_amount_quote`` column is a quote allocation, never account equity.

#: Bars per year for each frozen timeframe. Annualizes bar returns only.
PERIODS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
#: The DCA close types that are the overlay's own exit. Any other close type is
#: the frozen frame boundary: the strategy exit, or the forced flat at window end.
OVERLAY_CLOSE_TYPES = ("TAKE_PROFIT", "STOP_LOSS", "TRAILING_STOP")
#: Funding is applied nowhere in the replay: the existing DCA primitive models no
#: funding rate and no funding payment. Net-performance evidence needs funding in
#: PnL, so this stays False and keeps the performance claim false regardless of what
#: a cost model declares.
FUNDING_APPLIED_TO_PNL = False
#: Every definition is persisted with the metrics, so no reader has to guess.
REPLAY_DEFINITIONS = {
    "cycle": "one long cycle: an entry_transition bar through the next exit_transition bar. The house DCA overlay runs on that frame only, so its own exit is a close boundary and the source state decides when the next cycle may start.",
    "close_reason_precedence": "TP > SL > trailing > strategy exit on the same bar: the existing DCA primitive reports TAKE_PROFIT before STOP_LOSS before TRAILING_STOP. Otherwise the close bar is the frame boundary = STRATEGY_EXIT, or FORCE_FLAT for a cycle still open on the last evaluation bar.",
    "stop_loss_scope": "unchanged existing primitive semantics: stop_loss is evaluated on the last DCA stage only, i.e. after the safety orders filled.",
    "time_limit": "no overlay time limit is declared, so the primitive's TIME_LIMIT close is the frozen frame boundary itself, not an invented house parameter.",
    "fee_model": "unchanged existing primitive semantics: 2 * house_overlay.trade_cost of each filled stage's quote amount is subtracted from that stage's return, from the stage's entry bar onward. Fees are therefore a round-trip estimate charged for the whole open path, not per-fill exchange event timing, and no slippage or funding is applied.",
    "position_state": "flat = no source cycle and no account exposure; long = source cycle open and the DCA overlay holds quote exposure; source_long = source cycle open, the overlay already closed its own exit, account flat.",
    "equity": "initial_capital_quote + realized net PnL of finished cycles + the open cycle's own DCA ledger net_pnl_quote. filled_amount_quote is a quote allocation (doubled on its final bar by the primitive) and is never treated as account equity.",
    "net_return_pct": "100 * (end_equity / start_equity - 1), fee-inclusive account equity.",
    "cagr_pct": "100 * ((end_equity / start_equity) ** (365*24*3600 / elapsed_seconds) - 1): true geometric CAGR over the actual elapsed evaluation time. Null when either end is not positive or elapsed time is not positive.",
    "sharpe": "sqrt(periods_per_year) * mean(bar_return) / stdev(bar_return, ddof=1), zero risk-free, every evaluation-window bar including flat bars. Null when fewer than 2 bars or the stdev is zero.",
    "max_drawdown_pct": "100 * max over bars of (running peak equity - equity) / running peak equity; the peak includes the current bar; 0.0 when equity never falls.",
    "bar_return": "equity[t] / equity[t-1] - 1; 0.0 on the first bar and on a bar whose prior equity is not positive.",
    "win_rate_pct": "100 * cycles with positive realized net PnL / total cycles. Null when there is no cycle.",
    "total_trades": "completed or forced long cycles replayed inside the evaluation window.",
    "profit_factor": "gross positive realized cycle net PnL / abs(gross negative realized cycle net PnL). Null when there is no cycle, or when no cycle lost money (undefined/infinite).",
    "net_profit_usdt": "sum of realized cycle net PnL; equals end_equity - start_equity.",
    "performance_claimable": "true only when the applied economics are proven equivalent to an approved research cost model AND every required promotion metric is defined (CAGR, Sharpe) AND at least one cycle was replayed. Approval alone is not enough, and funding is not applied to PnL at all, so no cell can claim net performance in this version: the flag stays false until funding actually reaches PnL.",
}


def bar_returns(equity):
    """Return the per-bar simple returns of one equity series; the first bar is 0.0."""
    out = []
    for index, value in enumerate(equity):
        if index == 0 or equity[index - 1] <= 0:
            out.append(0.0)
        else:
            out.append(value / equity[index - 1] - 1.0)
    return out


def max_drawdown_pct(equity):
    """Return peak-to-trough drawdown in percent of the running peak."""
    peak, worst = None, 0.0
    for value in equity:
        peak = value if peak is None else max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return worst


def applied_cost_matches_model(cfg):
    """Return (matches, reason) for the applied economics against the declared model.

    The replay applies exactly one cost, the existing DCA primitive's frozen
    semantics: ``2 * house_overlay.trade_cost`` of each filled stage's quote
    amount. Slippage is applied nowhere and funding is applied nowhere.

    Funding is the hard stop: a run whose PnL carries no funding cannot support
    net-performance evidence, so declaring funding as zero or "not_modeled" is not
    accepted as a match. That is a deliberate fail-closed gap, lifted only by
    actually applying funding to PnL or by a separately audited policy that says
    otherwise.

    ponytail: while FUNDING_APPLIED_TO_PNL is False this returns False for every
    config, which is why the checks below are the ones a funding implementation must
    still satisfy. Do not weaken them to make a claim true.
    """
    cost = cfg.get("research_cost_model")
    overlay = cfg.get("house_overlay")
    if not isinstance(cost, dict) or not isinstance(overlay, dict):
        return False, "research cost model or house overlay is not declared"
    if cost.get("approved") is not True:
        return False, "research cost model is not approved"
    if not FUNDING_APPLIED_TO_PNL:
        return False, "funding is not applied to PnL, so no declared model can be proven equivalent"
    try:
        fee = float(cost.get("fees_per_side") or 0.0)
        slippage = float(cost.get("slippage") or 0.0)
        trade_cost = float(overlay.get("trade_cost") or 0.0)
    except (TypeError, ValueError):
        return False, "declared or applied cost is not numeric"
    if slippage != 0.0:
        return False, "slippage is declared but the replay applies none"
    if fee != trade_cost:
        return False, "the declared per-side fee is not the fee the overlay applies"
    return True, "the declared model equals the applied fee-only economics"


def performance_claimable(cfg, metrics):
    """True only when applied economics match an approved model and metrics are defined."""
    matches, _ = applied_cost_matches_model(cfg)
    return bool(
        matches
        and metrics.get("total_trades", 0) > 0
        and metrics.get("cagr_pct") is not None
        and metrics.get("sharpe") is not None
    )


def plan_cycles(rows, first_eval_index, last_eval_index):
    """Return (cycles inside the evaluation window, count of cycles before it).

    Fail closed on every impossible sequence rather than guessing: a bar carrying
    both transitions, an entry while the source is already long (no pyramiding,
    no overlapping cycles), an exit while it is flat, and a source still long on
    the first evaluation bar. A cycle still open on the last evaluation bar is
    forced flat there: that is the frozen window boundary, not a strategy exit.
    """
    cycles, open_index = [], None
    for index in range(int(last_eval_index) + 1):
        row = rows[index]
        if row["entry_transition"] and row["exit_transition"]:
            raise SystemExit("HOLD: bar %d carries both an entry and an exit transition" % index)
        if row["entry_transition"]:
            if open_index is not None:
                raise SystemExit(
                    "HOLD: entry at bar %d while the source is still long since bar %d" % (index, open_index))
            open_index = index
        elif row["exit_transition"]:
            if open_index is None:
                raise SystemExit("HOLD: exit at bar %d without an open source cycle" % index)
            cycles.append({"entry_index": open_index, "exit_index": index, "forced": False})
            open_index = None
    if open_index is not None:
        cycles.append({"entry_index": open_index, "exit_index": int(last_eval_index), "forced": True})
    window = []
    for cycle in cycles:
        if cycle["exit_index"] < first_eval_index:
            continue  # a cycle that completed before the frozen window is not replayed
        if cycle["entry_index"] < first_eval_index:
            raise SystemExit(
                "HOLD: the source strategy is long on the first evaluation bar (entry at bar %d, exit at bar %d)"
                % (cycle["entry_index"], cycle["exit_index"]))
        window.append(cycle)
    return window, len(cycles) - len(window)


def cycle_frame(rows, entry_index, exit_index):
    """Return the DCA input frame for one cycle: entry bar through close boundary."""
    import pandas as pd

    span = range(entry_index, exit_index + 1)
    frame = pd.DataFrame(
        {
            "timestamp": [rows[index]["datetime"].timestamp() for index in span],
            "close": [rows[index]["close"] for index in span],
            "low": [rows[index]["low"] for index in span],
            "high": [rows[index]["high"] for index in span],
        }
    )
    frame.index = frame["timestamp"].to_numpy(dtype=float)
    return frame


def replay_cycles(rows, cycles, overlay, leverage, simulator=None):
    """Run one house DCA overlay per source cycle and return the cycle records."""
    records = []
    for cycle in cycles:
        entry_index, exit_index = cycle["entry_index"], cycle["exit_index"]
        entry_price = rows[entry_index]["close"]
        prices = stage_prices(entry_price, overlay["so_spacing_pct"], overlay["safety_order_levels"])
        amounts = house_dca_amounts(
            overlay["base_order_quote_pct"],
            overlay["safety_order_quote_pct"],
            overlay["safety_order_levels"],
            overlay["initial_capital_quote"],
            leverage,
        )
        ledger, dca_close_type = dca_overlay_ledger(
            cycle_frame(rows, entry_index, exit_index),
            prices,
            amounts,
            None if overlay["take_profit_pct"] is None else Decimal(str(overlay["take_profit_pct"])),
            None if overlay["stop_loss_pct"] is None else Decimal(str(overlay["stop_loss_pct"])),
            float(overlay["trade_cost"] or 0.0),
            simulator=simulator,
        )
        if ledger is None or ledger.empty:
            raise SystemExit("HOLD: the house DCA overlay produced no ledger for the cycle at bar %d" % entry_index)
        close_index = entry_index + len(ledger) - 1
        if close_index > exit_index:
            raise SystemExit("HOLD: the house DCA overlay closed after the cycle boundary at bar %d" % entry_index)
        realized = float(ledger["net_pnl_quote"].iloc[-1])
        overlay_exit = dca_close_type in OVERLAY_CLOSE_TYPES
        pnl_by_index = {
            entry_index + offset: float(ledger["net_pnl_quote"].iloc[offset]) for offset in range(len(ledger))
        }
        # The overlay is flat from its own exit bar on; the source stays long
        # until its exit, so the cycle's realized PnL is simply carried.
        for index in range(close_index + 1, exit_index + 1):
            pnl_by_index[index] = realized
        stages = []
        for level, (price, amount) in enumerate(zip(prices, amounts)):
            filled = ledger.get("filled_amount_quote_%d" % level)
            stages.append(
                {
                    "stage": level,
                    "stage_price": float(price),
                    "stage_quote": float(amount),
                    "filled_quote": 0.0 if filled is None else float(filled.iloc[-1]),
                }
            )
        records.append(
            {
                "entry_index": entry_index,
                "exit_index": exit_index,
                "close_index": close_index,
                "entry_ts": rows[entry_index]["datetime"],
                "exit_ts": rows[exit_index]["datetime"],
                "close_ts": rows[close_index]["datetime"],
                "entry_price": float(entry_price),
                "forced": bool(cycle["forced"]),
                "dca_close_type": dca_close_type,
                "close_reason": dca_close_type
                if overlay_exit
                else ("FORCE_FLAT" if cycle["forced"] else "STRATEGY_EXIT"),
                "realized_pnl_quote": realized,
                "max_exposure_quote": float(max(ledger["filled_amount_quote"])),
                "ledger_rows": int(len(ledger)),
                "stage_prices": [float(value) for value in prices],
                "stage_quote_amounts": [float(value) for value in amounts],
                "stages": stages,
                "pnl_by_index": pnl_by_index,
            }
        )
    return records


def equity_projection(eval_indices, records, initial_capital):
    """Return one account-equity bar per evaluation-window bar.

    Realized PnL of finished cycles carries forward, the open cycle contributes
    its own DCA ledger mark to market, and every bar of the window is present so
    a reader can see flat periods. ``filled_amount_quote`` never enters here.
    """
    projection, carried, cursor = [], 0.0, 0
    for bar in eval_indices:
        while cursor < len(records) and records[cursor]["exit_index"] < bar:
            carried += records[cursor]["realized_pnl_quote"]
            cursor += 1
        position, number, mark = "flat", None, 0.0
        if cursor < len(records) and records[cursor]["entry_index"] <= bar:
            record = records[cursor]
            number = cursor + 1
            mark = record["pnl_by_index"].get(bar, record["realized_pnl_quote"])
            position = "long" if bar <= record["close_index"] else "source_long"
        projection.append(
            {
                "equity_quote": initial_capital + carried + mark,
                "position_state": position,
                "cycle_number": number,
                "realized_pnl_quote": carried,
                "mark_pnl_quote": mark,
            }
        )
    for bar, value in zip(projection, bar_returns([item["equity_quote"] for item in projection])):
        bar["bar_return"] = value
    return projection


def metrics_from_equity(timestamps, equity, realized_pnls, periods_per_year):
    """Return the required fee-inclusive metrics plus their definitions."""
    if not equity or len(timestamps) != len(equity):
        raise SystemExit("HOLD: the evaluation window produced no usable equity series")
    start, end = float(equity[0]), float(equity[-1])
    elapsed = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0.0
    returns = bar_returns(equity)
    sharpe = None
    if len(returns) > 1:
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        deviation = variance ** 0.5
        if deviation > 0:
            sharpe = mean / deviation * (periods_per_year ** 0.5)
    cagr = None
    if start > 0 and end > 0 and elapsed > 0:
        cagr = ((end / start) ** (SECONDS_PER_YEAR / elapsed) - 1.0) * 100.0
    winners = [value for value in realized_pnls if value > 0]
    losers = [value for value in realized_pnls if value < 0]
    gross_loss = -sum(losers)
    total = len(realized_pnls)
    return {
        "net_return_pct": (end / start - 1.0) * 100.0,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct(equity),
        "win_rate_pct": (100.0 * len(winners) / total) if total else None,
        "total_trades": total,
        "profit_factor": (sum(winners) / gross_loss) if losers and gross_loss > 0 else None,
        "net_profit_usdt": sum(realized_pnls),
        "start_equity_quote": start,
        "end_equity_quote": end,
        "elapsed_seconds": elapsed,
        "evaluation_bars": len(equity),
        "periods_per_year": periods_per_year,
        "bar_returns_ddof": 1,
        "max_drawdown_definition": REPLAY_DEFINITIONS["max_drawdown_pct"],
        "definitions": dict(REPLAY_DEFINITIONS),
    }


def trades_fieldnames(stage_count):
    """Return the trades.csv header for an overlay with `stage_count` DCA stages."""
    names = [
        "cycle",
        "entry_ts_utc",
        "exit_ts_utc",
        "close_ts_utc",
        "close_reason",
        "dca_close_type",
        "forced_flat",
        "entry_price",
        "realized_pnl_quote",
        "max_exposure_quote",
        "ledger_rows",
    ]
    for level in range(stage_count):
        names += ["stage_%d_price" % level, "stage_%d_quote" % level, "stage_%d_filled_quote" % level]
    return names


def write_evidence(out_dir, summary, signal_rows, records, evaluation_times, projection, metrics):
    """Write the cell's non-MLflow evidence into out_dir. Return the file names."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    signal_fields = list(signal_rows[0])
    with (out_dir / "signals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=signal_fields)
        writer.writeheader()
        for row in signal_rows:
            writer.writerow(
                {
                    key: row[key].isoformat() if isinstance(row[key], datetime) else row[key]
                    for key in signal_fields
                }
            )
    stage_count = len(records[0]["stages"]) if records else 0
    with (out_dir / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trades_fieldnames(stage_count))
        writer.writeheader()
        for number, record in enumerate(records, start=1):
            row = {
                "cycle": number,
                "entry_ts_utc": record["entry_ts"].isoformat(),
                "exit_ts_utc": record["exit_ts"].isoformat(),
                "close_ts_utc": record["close_ts"].isoformat(),
                "close_reason": record["close_reason"],
                "dca_close_type": record["dca_close_type"],
                "forced_flat": int(record["forced"]),
                "entry_price": record["entry_price"],
                "realized_pnl_quote": record["realized_pnl_quote"],
                "max_exposure_quote": record["max_exposure_quote"],
                "ledger_rows": record["ledger_rows"],
            }
            for stage in record["stages"]:
                for key, value in (
                    ("price", stage["stage_price"]),
                    ("quote", stage["stage_quote"]),
                    ("filled_quote", stage["filled_quote"]),
                ):
                    row["stage_%d_%s" % (stage["stage"], key)] = value
            writer.writerow(row)
    # The pinned transport requires a readable events.csv and treats fewer than
    # two lines as "no execution". One row per replayed stage keeps that
    # contract; a cell with no cycle has no stage to report and stays header-only.
    with (out_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cycle", "stage", "stage_price", "stage_quote_amount", "filled_quote"]
        )
        writer.writeheader()
        for number, record in enumerate(records, start=1):
            for stage in record["stages"]:
                writer.writerow(
                    {
                        "cycle": number,
                        "stage": stage["stage"],
                        "stage_price": stage["stage_price"],
                        "stage_quote_amount": stage["stage_quote"],
                        "filled_quote": stage["filled_quote"],
                    }
                )
    with (out_dir / "equity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "datetime",
                "equity_quote",
                "bar_return",
                "realized_pnl_quote",
                "mark_pnl_quote",
                "position_state",
                "cycle_number",
            ],
        )
        writer.writeheader()
        for stamp, bar in zip(evaluation_times, projection):
            writer.writerow(
                {
                    "datetime": stamp.isoformat(),
                    "equity_quote": bar["equity_quote"],
                    "bar_return": bar["bar_return"],
                    "realized_pnl_quote": bar["realized_pnl_quote"],
                    "mark_pnl_quote": bar["mark_pnl_quote"],
                    "position_state": bar["position_state"],
                    "cycle_number": "" if bar["cycle_number"] is None else bar["cycle_number"],
                }
            )
    return ["summary.json", "signals.csv", "trades.csv", "events.csv", "equity.csv", "metrics.json"]


def write_manifest(out_dir):
    """Hash every non-MLflow evidence file into manifest.json and return the manifest."""
    out_dir = Path(out_dir)
    manifest = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and "mlruns" not in path.parts and path.name != "manifest.json":
            manifest[str(path.relative_to(out_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_signal_artifact(cell_dir, artifact):
    """Return the frozen strategy ``signal`` callable. Fail closed on mismatch.

    The trust boundary is the cell directory plus the declared sha256. The
    transport already binds the entrypoint and effective_config as siblings in
    that directory, so an artifact outside it is a rejected identity, not a
    sandbox question.
    """
    if not isinstance(artifact, dict):
        raise SystemExit("HOLD: strategy artifact is not declared")
    ref, want = artifact.get("ref"), artifact.get("sha256")
    if not isinstance(ref, str) or not ref:
        raise SystemExit("HOLD: strategy artifact ref is missing")
    path = Path(ref)
    if not path.is_file():
        raise SystemExit("HOLD: strategy artifact is not readable: " + ref)
    if path.resolve().parent != Path(cell_dir).resolve():
        raise SystemExit("HOLD: strategy artifact is outside the cell directory")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != want:
        raise SystemExit("HOLD: strategy artifact hash mismatch: %s != %s" % (digest, want))
    spec = importlib.util.spec_from_file_location("cell_strategy_signal_" + digest[:12], path)
    if spec is None or spec.loader is None:
        raise SystemExit("HOLD: strategy artifact is not importable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # any artifact-side error is a fail-closed HOLD
        raise SystemExit("HOLD: strategy artifact import failed: %s" % exc)
    function = getattr(module, SIGNAL_FUNCTION, None)
    if not callable(function):
        raise SystemExit("HOLD: strategy artifact has no callable %s()" % SIGNAL_FUNCTION)
    return function


def problems_in_signal_rows(rows):
    """Return the list of fail-closed problems in one artifact's row output."""
    if not isinstance(rows, (list, tuple)) or not rows:
        return ["strategy artifact returned no rows"]
    bad = []
    last = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            bad.append("row %d is not an object" % index)
            continue
        missing = [key for key in SIGNAL_ROW_KEYS if key not in row]
        if missing:
            bad.append("row %d is missing %s" % (index, ",".join(missing)))
            continue
        if not isinstance(row["datetime"], datetime):
            bad.append("row %d datetime is not a datetime" % index)
            continue
        if last is not None and row["datetime"] < last:
            bad.append("row %d datetime goes backwards" % index)
        last = row["datetime"]
        for key in ("entry_transition", "exit_transition"):
            if row[key] not in (0, 1):
                bad.append("row %d %s is not 0 or 1" % (index, key))
        if len(bad) > 8:
            break
    return bad


def build_effective_config(spec, identity, artifact_sha256):
    """Return one cell's effective_config.json content from the campaign spec.

    The strategy block is the frozen record's own parameter block plus the
    run-scoped artifact identity. This file never interprets those parameters.
    """
    entry_dir = "%s/%s" % (spec["materialize"]["cells_dir"], identity["cell_id"])
    strategy = dict(spec["strategy"], symbol=identity["symbol"], timeframe=identity["timeframe"])
    strategy["params"] = dict(strategy.get("params") or {})
    strategy["sides"] = list(strategy.get("sides") or [])
    strategy["artifact"] = {
        "ref": "%s/%s" % (entry_dir, SIGNAL_ARTIFACT_NAME),
        "sha256": artifact_sha256,
    }
    # The frozen capability declaration is the engine's verified primitive set
    # plus the sides this strategy actually declares. The readiness gate reads
    # this block from the frozen config, so a two-sided strategy cannot be
    # re-labelled as long-only by whoever materialized it.
    capabilities = {key: list(values) for key, values in CAPABILITIES.items()}
    capabilities["sides"] = list(strategy["sides"])
    return {
        "schema": SCHEMA,
        "scope": "frozen round0 Formal Baseline cell; not Survivor/HB Certified",
        "candidate": dict(spec["candidate"]),
        "runtime": dict(spec["runtime"]),
        "data": dict(spec["data"]),
        "execution": {
            "engine": "qlib.backtest.research_cell (house DCA overlay via backtest_hummingbot_dca)",
            "entrypoint": "%s/%s" % (entry_dir, ENTRYPOINT_NAME),
        },
        "research_leverage": identity["research_leverage"],
        "strategy": strategy,
        "capabilities": capabilities,
        "house_overlay": dict(spec["house_overlay"]),
        "research_cost_model": dict(spec["research_cost_model"]),
        "economics_boundary": {
            "research_leverage_is_exposure_multiplier": True,
            "isolated_margin_funding_slippage": "not modeled here; Node13 Hummingbot owns canonical economics",
        },
    }


def out_dir_short(frozen):
    """Return the transport's out_dir suffix for one frozen cell identity.

    Mirrors research-transport.py ``out_dir_for``: the JSON of the four frozen
    fields, keys sorted, separators collapsed, sha256, first 12 hex chars.
    """
    canonical = json.dumps(frozen, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def materialize(spec_path, root, force=False):
    """Write the 42 frozen cell dirs and the campaign manifest. Return the manifest."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    overlay = spec.get("house_overlay") or {}
    missing = [key for key in OVERLAY_KEYS if key not in overlay]
    missing += [key for key in OVERLAY_NUMERIC_KEYS if key not in missing and not isinstance(overlay[key], (int, float))]
    if missing:
        raise SystemExit("HOLD: house_overlay has no approved value for: " + ", ".join(missing))
    if not isinstance(spec.get("research_cost_model"), dict) or "approved" not in spec["research_cost_model"]:
        raise SystemExit("HOLD: research_cost_model must declare an explicit approved flag")
    artifact_spec = spec.get("strategy_artifact") or {}
    artifact_ref = artifact_spec.get("source")
    if not isinstance(artifact_ref, str) or not artifact_ref:
        raise SystemExit("HOLD: strategy_artifact.source is not declared")
    artifact_source = Path(artifact_ref)
    if not artifact_source.is_file():
        raise SystemExit("HOLD: strategy artifact source is not readable: " + artifact_ref)
    artifact_bytes = artifact_source.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    # The strategy must state which execution sides it implements. A frozen
    # two-sided record reduced to a long-only artifact is a capability gap, so
    # it is refused before any cell is written instead of being shipped as a
    # smaller version of the same strategy.
    sides = (spec.get("strategy") or {}).get("sides")
    if not isinstance(sides, list) or not sides:
        raise SystemExit("HOLD: strategy.sides is not declared")
    outside = [side for side in sides if side not in CAPABILITIES["sides"]]
    if outside:
        raise SystemExit("HOLD: the engine has no execution primitive for side: " + ", ".join(map(str, outside)))
    source = Path(__file__).resolve()
    source_bytes = source.read_bytes()
    cells = []
    for identity in campaign_identities():
        cell_dir = Path(root) / identity["cell_id"]
        if cell_dir.exists() and not force:
            raise SystemExit("refuse overwrite: " + str(cell_dir))
        cell_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, cell_dir / ENTRYPOINT_NAME)
        shutil.copyfile(artifact_source, cell_dir / SIGNAL_ARTIFACT_NAME)
        config = build_effective_config(spec, identity, artifact_sha256)
        (cell_dir / CONFIG_NAME).write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        entrypoint_sha256 = hashlib.sha256(source_bytes).hexdigest()
        config_sha256 = hashlib.sha256((cell_dir / CONFIG_NAME).read_bytes()).hexdigest()
        out_dir = "%s/n8n-live-%s" % (
            spec["materialize"]["results_root"],
            out_dir_short(
                {
                    "effective_config_sha256": config_sha256,
                    "entrypoint_sha256": entrypoint_sha256,
                    "qlib_runtime_commit": spec["runtime"]["commit"],
                    "source_blob_sha": spec["candidate"]["git_blob_sha"],
                }
            ),
        )
        cells.append(
            {
                "cell_id": identity["cell_id"],
                "symbol": identity["symbol"],
                "timeframe": identity["timeframe"],
                "research_leverage": identity["research_leverage"],
                "source_commit": spec["candidate"]["source_commit"],
                "candidate_path": spec["candidate"]["path"],
                "source_blob_sha": spec["candidate"]["git_blob_sha"],
                "entrypoint_ref": config["execution"]["entrypoint"],
                "entrypoint_sha256": entrypoint_sha256,
                "strategy_artifact_ref": config["strategy"]["artifact"]["ref"],
                "strategy_artifact_sha256": config["strategy"]["artifact"]["sha256"],
                "effective_config_ref": config["execution"]["entrypoint"].rsplit("/", 1)[0] + "/" + CONFIG_NAME,
                "effective_config_sha256": config_sha256,
                "qlib_runtime_commit": spec["runtime"]["commit"],
                "capabilities": {key: list(values) for key, values in config["capabilities"].items()},
                "evidence_refs": {
                    "candidate": spec["candidate"]["blob_api_ref"],
                    "manifest": out_dir + "/manifest.json",
                    "mlflow_readback": out_dir + "/mlflow_readback.json",
                },
                "research_cost_model": dict(spec["research_cost_model"]),
                "run_inputs": {"symbol": identity["symbol"], "timeframe": identity["timeframe"],
                               "research_leverage": identity["research_leverage"]},
            }
        )
    manifest = {
        "schema": "qlib-formal-campaign-manifest-v1",
        "generated_from": str(Path(spec_path).resolve()),
        "cell_count": len(cells),
        "cells": cells,
    }
    (Path(root) / "campaign-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_cell(cell_dir, out_dir, simulator=None):
    """Execute one cell and write its evidence into out_dir. Return the contract dict.

    The frozen evaluation window is replayed whole: every entry/exit transition of
    the artifact becomes one long cycle, each cycle runs the existing house DCA
    overlay, and one fee-inclusive account equity series spans every window bar.
    A zero-cycle run is valid evidence, but it never fabricates performance.
    """
    import importlib.metadata

    import pandas as pd
    import qlib
    from mlflow.tracking import MlflowClient
    from qlib.data import D

    cell_dir = Path(cell_dir)
    out_dir = Path(out_dir)
    cfg = json.loads((cell_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    problems = problems_in_config(cfg)
    if problems:
        raise SystemExit("HOLD: cell config is not runnable: " + "; ".join(problems))
    if out_dir.exists():
        raise SystemExit("refuse overwrite " + str(out_dir))
    strategy = cfg["strategy"]
    symbol, timeframe, freq = strategy["symbol"], strategy["timeframe"], QLIB_FREQ[strategy["timeframe"]]
    window_start = pd.Timestamp(strategy["window_start_utc"]).to_pydatetime()
    window_end = pd.Timestamp(strategy["window_end_utc_inclusive"]).to_pydatetime()
    run_end = pd.Timestamp(strategy["force_flat_bar_utc"]).to_pydatetime() if strategy.get("force_flat_bar_utc") else window_end
    warmup_start = pd.Timestamp(window_start) - pd.Timedelta(float(strategy.get("warmup_days", 0)), unit="D")
    signal = load_signal_artifact(cell_dir, strategy["artifact"])

    qlib.init(provider_uri=cfg["data"]["provider_ref"], region="us",
              expression_cache=None, dataset_cache=None)
    fields = ["$high", "$low", "$close"]
    try:
        features = D.features([symbol], fields + ["$volume"],
                              start_time=str(warmup_start), end_time=str(run_end), freq=freq).sort_index()
    except Exception:  # ponytail: a frozen provider without volume is price-only; volume artifacts fail closed
        features = D.features([symbol], fields,
                              start_time=str(warmup_start), end_time=str(run_end), freq=freq).sort_index()
    if features.empty:
        raise SystemExit("HOLD: frozen provider returned no rows")
    times = [ts.to_pydatetime() for ts in features.index.get_level_values("datetime")]
    rows, entry_ts = signal(
        high=features["$high"].astype(float).to_numpy(),
        low=features["$low"].astype(float).to_numpy(),
        close=features["$close"].astype(float).to_numpy(),
        volume=None if "$volume" not in features else features["$volume"].astype(float).to_numpy(),
        times=times,
        params=dict(strategy.get("params") or {}),
        window_start=window_start,
        window_end=window_end,
    )
    row_problems = problems_in_signal_rows(rows)
    if row_problems:
        raise SystemExit("HOLD: strategy artifact output is not runnable: " + "; ".join(row_problems))
    eval_indices = [index for index, row in enumerate(rows) if window_start <= row["datetime"] <= run_end]
    if not eval_indices:
        raise SystemExit("HOLD: frozen signal produced no rows inside the window")
    eval_rows = [rows[index] for index in eval_indices]
    # The artifact ABI still returns one entry_ts. It is backward-compatibility
    # evidence only: it must name one of the transition bars the rows carry, but it
    # is never the set of cycles that ran and the committed fixtures disagree about
    # whether it is the first or the last entry. The rows decide the replay.
    entries = [index for index, row in enumerate(rows)
               if row["entry_transition"] == 1 and row["datetime"] <= run_end]
    if entries and entry_ts is None:
        raise SystemExit("HOLD: the signal artifact reports no entry_ts but its rows carry an entry transition")
    if not entries and entry_ts is not None:
        raise SystemExit("HOLD: the signal artifact reports an entry_ts its rows do not carry")
    entry_ts_position = None
    if entry_ts is not None:
        stamp = pd.Timestamp(entry_ts)
        entry_stamps = [pd.Timestamp(rows[index]["datetime"]) for index in entries]
        if stamp == entry_stamps[0]:
            entry_ts_position = "first"
        elif stamp == entry_stamps[-1]:
            entry_ts_position = "last"
        elif stamp in entry_stamps:
            entry_ts_position = "intermediate"
        else:
            raise SystemExit("HOLD: the signal artifact entry_ts is not one of its entry transition bars")

    overlay = cfg["house_overlay"]
    cycles, skipped_cycles = plan_cycles(rows, eval_indices[0], eval_indices[-1])
    records = replay_cycles(rows, cycles, overlay, cfg["research_leverage"], simulator=simulator)
    projection = equity_projection(eval_indices, records, float(overlay["initial_capital_quote"]))
    evaluation_times = [rows[index]["datetime"] for index in eval_indices]
    realized = [record["realized_pnl_quote"] for record in records]
    metrics = metrics_from_equity(evaluation_times, [bar["equity_quote"] for bar in projection], realized,
                                  PERIODS_PER_YEAR[timeframe])
    cost_matches, cost_reason = applied_cost_matches_model(cfg)
    claimable = performance_claimable(cfg, metrics)
    close_reasons = {}
    for record in records:
        close_reasons[record["close_reason"]] = close_reasons.get(record["close_reason"], 0) + 1
    summary = {
        "engine": "qlib.backtest.research_cell / backtest_hummingbot_dca (house DCA overlay)",
        "qlib_version": importlib.metadata.version("pyqlib"),
        "symbol": symbol,
        "timeframe": timeframe,
        "frequency": freq,
        "research_leverage": cfg["research_leverage"],
        "evaluation_window_start_utc": evaluation_times[0].isoformat(),
        "evaluation_window_end_utc": evaluation_times[-1].isoformat(),
        "evaluation_rows_including_force_flat": len(eval_rows),
        "entry_ts_utc": None if entry_ts is None else str(entry_ts),
        "entry_ts_semantics": "backward-compatible evidence only: the single entry_ts the artifact ABI "
                              "returns; the replay uses every entry/exit transition row",
        "entry_ts_position": entry_ts_position,
        "cycles_replayed": len(records),
        "cycles_before_evaluation_window": skipped_cycles,
        "close_reason_counts": close_reasons,
        # Kept for readability of the single-cycle case; per-cycle close facts
        # are in ``cycles``. A multi-cycle cell has no single close type.
        "close_type": records[0]["dca_close_type"] if len(records) == 1 else None,
        "max_exposure_quote": max((record["max_exposure_quote"] for record in records), default=0.0),
        "net_pnl_quote": float(sum(realized)),
        "ledger_rows": int(sum(record["ledger_rows"] for record in records)),
        "cycles": [
            {
                "cycle": number,
                "entry_ts_utc": record["entry_ts"].isoformat(),
                "exit_ts_utc": record["exit_ts"].isoformat(),
                "close_ts_utc": record["close_ts"].isoformat(),
                "close_reason": record["close_reason"],
                "dca_close_type": record["dca_close_type"],
                "forced_flat": record["forced"],
                "entry_price": record["entry_price"],
                "stage_prices": record["stage_prices"],
                "stage_quote_amounts": record["stage_quote_amounts"],
                "stages": record["stages"],
                "realized_pnl_quote": record["realized_pnl_quote"],
                "max_exposure_quote": record["max_exposure_quote"],
                "ledger_rows": record["ledger_rows"],
            }
            for number, record in enumerate(records, start=1)
        ],
        "strategy_artifact_sha256": strategy["artifact"]["sha256"],
        "strategy_params": dict(strategy.get("params") or {}),
        "economics_boundary": dict(cfg["economics_boundary"], research_cost_model=cfg["research_cost_model"],
                                   cost_model_approved=cost_model_claimable(cfg),
                                   performance_claimable=claimable),
        "applied_economics": {
            "applied": "2 * house_overlay.trade_cost of each filled DCA stage's quote amount, from that "
                       "stage's entry bar onward (frozen primitive semantics; not per-fill event timing)",
            "slippage_applied": False,
            "funding_applied": False,
            "declared_model_matches_applied": cost_matches,
            "declared_model_match_reason": cost_reason,
        },
        "metrics_scope": "metrics.json holds the fee-inclusive account-equity metrics and their definitions",
        # MLflow cannot store a null metric, so an undefined metric is absent from the
        # MLflow run and null in metrics.json. Listing it here keeps that visible.
        "mlflow_metrics_omitted": sorted(
            key for key in ("net_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct", "win_rate_pct",
                            "total_trades", "profit_factor", "net_profit_usdt")
            if metrics[key] is None
        ),
    }
    out_dir.mkdir(parents=True)
    evidence = write_evidence(out_dir, summary, eval_rows, records, evaluation_times, projection, metrics)
    summary_path = out_dir / "summary.json"
    mlruns = out_dir / "mlruns"
    mlruns.mkdir()
    tracking_uri = mlruns.resolve().as_uri()
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = client.create_experiment("formal-cell-" + cell_dir.name)
    client.create_run(exp_id, tags={"engine": "pyqlib", "cell_id": cell_dir.name,
                                    "research_leverage": str(cfg["research_leverage"]),
                                    "performance_claimable": "true" if claimable else "false"})
    run_id = client.search_runs(exp_id, order_by=["attributes.start_time DESC"], max_results=1)[0].info.run_id
    for key, value in {"qlib_version": summary["qlib_version"], "symbol": symbol, "frequency": freq,
                       "timeframe": timeframe,
                       "research_leverage": str(cfg["research_leverage"]),
                       "strategy_artifact_sha256": strategy["artifact"]["sha256"],
                       "strategy_params": json.dumps(dict(strategy.get("params") or {}), sort_keys=True),
                       "max_drawdown_definition": metrics["max_drawdown_definition"],
                       "bar_returns_ddof": str(metrics["bar_returns_ddof"]),
                       "periods_per_year": str(metrics["periods_per_year"]),
                       "close_reason_precedence": REPLAY_DEFINITIONS["close_reason_precedence"]}.items():
        client.log_param(run_id, key, value)
    logged = {
        "net_return_pct": metrics["net_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "sharpe": metrics["sharpe"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "win_rate_pct": metrics["win_rate_pct"],
        "total_trades": metrics["total_trades"],
        "profit_factor": metrics["profit_factor"],
        "net_profit_usdt": metrics["net_profit_usdt"],
        "max_exposure_quote": summary["max_exposure_quote"],
        "net_pnl_quote": summary["net_pnl_quote"],
        "ledger_rows": summary["ledger_rows"],
        "cycles_replayed": len(records),
        "evaluation_bars": metrics["evaluation_bars"],
    }
    for key, value in logged.items():
        if value is not None:
            client.log_metric(run_id, key, float(value))
    for name in evidence:
        client.log_artifact(run_id, str(out_dir / name), artifact_path="backtest")
    for artifact in (cell_dir / CONFIG_NAME, Path(__file__), cell_dir / SIGNAL_ARTIFACT_NAME):
        client.log_artifact(run_id, str(artifact), artifact_path="repro")
    client.set_terminated(run_id, status="FINISHED")
    readback = client.get_run(run_id)
    readback_payload = {
        "tracking_uri": tracking_uri,
        "experiment_id": exp_id,
        "run_id": run_id,
        "status": readback.info.status,
        "params": readback.data.params,
        "metrics": readback.data.metrics,
        "artifact_names": sorted(x.path for x in client.list_artifacts(run_id, "backtest"))
        + sorted(x.path for x in client.list_artifacts(run_id, "repro")),
    }
    (out_dir / "mlflow_readback.json").write_text(
        json.dumps(readback_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = write_manifest(out_dir)
    return {"summary": summary, "metrics": metrics, "mlflow": readback_payload, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", nargs="?", default="run", choices=("run", "materialize"))
    parser.add_argument("--cell-dir", default=None)
    parser.add_argument("--spec", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.verb == "run":
        cell_dir = Path(args.cell_dir) if args.cell_dir else Path(__file__).resolve().parent
        out_dir = Path(os.environ.get("QLIB_OUT_DIR") or (cell_dir / "out"))
        print(json.dumps(run_cell(cell_dir, out_dir), indent=2, sort_keys=True))
        return 0
    if not args.spec or not args.root:
        parser.error("materialize needs --spec and --root")
    manifest = materialize(args.spec, Path(args.root), force=args.force)
    print(json.dumps({"materialized": manifest["cell_count"], "cells": [c["cell_id"] for c in manifest["cells"]]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
