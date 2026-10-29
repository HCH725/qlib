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
  allocation x research_leverage). It is a research exposure multiplier: it is
  not a Binance liquidation or isolated-margin account model, and it is not an
  account-equity model. Funding is a separate charge this run applies from a
  verified source identity (see Funding, below); canonical execution economics
  and certification stay with the Node13 Hummingbot certification lane.
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

    summary.json   cycle plan, per-cycle stage facts, applied funding economics
    signals.csv    the artifact's rows inside the frozen evaluation window
    trades.csv     one row per replayed cycle: entry/exit/close, stages, fees, funding, net PnL
    events.csv     one row per replayed stage (the pinned transport reads it)
    funding_events.csv  one row per charged funding event (empty header when nothing was held)
    equity.csv     the whole evaluation window, flat bars included
    metrics.json   fee-and-funding-inclusive performance metrics plus their definitions
    manifest.json  sha256 of every non-MLflow evidence file in this directory

Funding, deliberately narrow::

    A long perpetual position pays funding, so a fee-only replay cannot honestly
    claim net performance. ``research_cost_model.funding`` is either the explicit
    ``not_modeled`` declaration or one structured source identity
    (``binance-usdm-funding-v1``). An identity is verified against the exact bytes
    on disk before it can reach PnL or claimability, and a modeled
    (``modeled_funding``) row reconstructs its mark from the 1h mark bar OPEN of
    the UTC hour containing the event. Every funding comparison happens at a bar
    CLOSE boundary derived from the timeframe, never at a Qlib bar label.

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
import math
import os
import re
import shutil
import sys
from datetime import datetime, timezone
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

#: Binance USD-M funding: the one frozen source identity a run may apply. A
#: declaration is a proven-source claim, so it carries the exact bytes it used.
FUNDING_BASIS = "binance-usdm-funding-v1"
FUNDING_MODELED_MARK_BASIS = "binance-usdm-mark-1h-open-v1"
#: The only ``truth_status`` values the replay may read. Anything else is a HOLD:
#: an unregistered estimate must never be mistaken for a modeled or official row.
FUNDING_TRUTH_STATUS = ("official", "modeled_funding")
FUNDING_IDENTITY_KEYS = (
    "basis",
    "source_ref",
    "source_sha256",
    "truth_status",
    "modeled_mark_basis",
    "modeled_mark_snapshot_sha256",
)
#: The explicit "this cell does not model funding" declaration. It keeps a cell
#: runnable, and it keeps ``performance_claimable`` false: nothing is zero-filled.
FUNDING_NOT_MODELED = (None, "not_modeled")
#: Milliseconds per frozen timeframe. A Qlib row label is the bar OPEN, so the
#: close boundary is label + this interval; funding ordering is defined at close.
TIMEFRAME_MS = {"1h": 3_600_000, "4h": 4 * 3_600_000, "1d": 86_400_000}
FUNDING_HOUR_MS = 3_600_000


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


def problems_in_funding_identity(identity):
    """Return the fail-closed problems in one declared funding source identity.

    The identity is the only thing that can prove which bytes a net-of-funding
    number came from, so every field is checked by name and by literal value
    before the series is read. A partial identity is a declaration, not a proof.
    """
    if not isinstance(identity, dict):
        return ["funding is not a structured source identity"]
    bad = ["funding identity is missing %s" % key for key in FUNDING_IDENTITY_KEYS if key not in identity]
    if identity.get("basis") != FUNDING_BASIS:
        bad.append("funding identity basis is not %s" % FUNDING_BASIS)
    if identity.get("modeled_mark_basis") != FUNDING_MODELED_MARK_BASIS:
        bad.append("funding identity modeled_mark_basis is not %s" % FUNDING_MODELED_MARK_BASIS)
    if list(identity.get("truth_status") or []) != list(FUNDING_TRUTH_STATUS):
        bad.append("funding identity truth_status is not exactly %s" % (list(FUNDING_TRUTH_STATUS),))
    source_ref = identity.get("source_ref")
    if not isinstance(source_ref, str) or not Path(source_ref).is_absolute():
        bad.append("funding identity source_ref is not an absolute path")
    for key in ("source_sha256", "modeled_mark_snapshot_sha256"):
        if not HEX64.match(str(identity.get(key))):
            bad.append("funding identity %s is not a 64-hex digest" % key)
    return bad


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
        if "funding" in cost:
            funding = cost["funding"]
            if isinstance(funding, dict):
                bad.extend(problems_in_funding_identity(funding))
            elif funding not in FUNDING_NOT_MODELED:
                bad.append("research cost model funding is neither a proven source identity nor the "
                           "explicit not_modeled declaration")
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
#: Every definition is persisted with the metrics, so no reader has to guess.
REPLAY_DEFINITIONS = {
    "cycle": "one long cycle: an entry_transition bar through the next exit_transition bar. The house DCA overlay runs on that frame only, so its own exit is a close boundary and the source state decides when the next cycle may start.",
    "close_reason_precedence": "TP > SL > trailing > strategy exit on the same bar: the existing DCA primitive reports TAKE_PROFIT before STOP_LOSS before TRAILING_STOP. Otherwise the close bar is the frame boundary = STRATEGY_EXIT, or FORCE_FLAT for a cycle still open on the last evaluation bar.",
    "stop_loss_scope": "unchanged existing primitive semantics: stop_loss is evaluated on the last DCA stage only, i.e. after the safety orders filled.",
    "time_limit": "no overlay time limit is declared, so the primitive's TIME_LIMIT close is the frozen frame boundary itself, not an invented house parameter.",
    "bar_boundary": "a Qlib row label is the bar OPEN. Every funding comparison uses the bar CLOSE boundary = label + the frozen timeframe interval (1h/4h/1d), so an event at a bar's open never charges a stage first filled on that bar.",
    "fee_model": "unchanged existing primitive semantics: 2 * house_overlay.trade_cost of each filled stage's quote amount is subtracted from that stage's return, from the stage's entry bar onward. Fees are therefore a round-trip estimate charged for the whole open path, not per-fill exchange event timing, and no slippage is applied. Funding is a separate charge: see funding_model.",
    "funding_model": "Binance USD-M perpetual funding, long only. A positive funding_rate is paid (it reduces PnL) and a negative one is received. The payment at one event is held_base_qty * funding_mark_price * funding_rate, where held_base_qty is the sum of the DCA stages whose stage_quote/stage_price is held immediately before the event. Every comparison uses the event's normalized UTC-hour boundary (funding_time_ms floored to its UTC hour, which is also the 1h mark bar open), never the raw jittered funding_time_ms; the raw value stays in the evidence as source data. A stage filled at bar close T is NOT charged for an event whose boundary is T (eligibility: fill_close_ms < funding_boundary_ms), because it was not held during the interval the event pays for; a position closing at bar close T IS charged for an event whose boundary is T (eligibility: funding_boundary_ms <= close_boundary_ms), because it was held through that interval. Official rows use the official mark_price; modeled_funding rows use the registered rate and reconstruct the mark from the OPEN of the Binance USD-M 1h mark bar at the event boundary, and are never described as official. No event is charged while the source is source_long and the account flat, and no event after a cycle's close boundary is charged to that cycle. Funding is charged into the same bar's equity as the rest of that bar's PnL.",
    "position_state": "flat = no source cycle and no account exposure; long = source cycle open and the DCA overlay holds quote exposure; source_long = source cycle open, the overlay already closed its own exit, account flat.",
    "equity": "initial_capital_quote + realized net PnL of finished cycles + the open cycle's own DCA ledger net_pnl_quote - the funding charged so far by the open cycle. filled_amount_quote is a quote allocation (doubled on its final bar by the primitive) and is never treated as account equity.",
    "net_return_pct": "100 * (end_equity / start_equity - 1): account equity net of fees and funding.",
    "cagr_pct": "100 * ((end_equity / start_equity) ** (365*24*3600 / elapsed_seconds) - 1): true geometric CAGR over the actual elapsed evaluation time on fee-and-funding-inclusive equity. Null when either end is not positive or elapsed time is not positive.",
    "sharpe": "sqrt(periods_per_year) * mean(bar_return) / stdev(bar_return, ddof=1), zero risk-free, every evaluation-window bar including flat bars, computed on fee-and-funding-inclusive returns. Null when fewer than 2 bars or the stdev is zero.",
    "max_drawdown_pct": "100 * max over bars of (running peak equity - equity) / running peak equity, on fee-and-funding-inclusive equity; the running peak is initialized from frozen initial capital before the first evaluation bar, so first-bar costs are included; 0.0 when equity never falls.",
    "bar_return": "equity[t] / equity[t-1] - 1 for t>0; on the first evaluation bar, equity[0] / frozen_initial_capital - 1 so first-bar fees, funding and PnL are not dropped. A non-positive prior/base equity is a HOLD.",
    "win_rate_pct": "100 * cycles with positive realized net PnL (after fees and funding) / total cycles. Null when there is no cycle.",
    "total_trades": "completed or forced long cycles replayed inside the evaluation window.",
    "profit_factor": "gross positive realized cycle net PnL after fees and funding / abs(gross negative realized cycle net PnL after fees and funding). Null when there is no cycle, or when no cycle lost money (undefined/infinite).",
    "net_profit_usdt": "sum of realized cycle net PnL after fees and funding; equals end_equity - frozen initial capital after the evaluation window is forced flat.",
    "trades_csv": "one row per replayed cycle. dca_net_pnl_quote is the house DCA primitive's own fee-inclusive result, funding_paid_quote is every funding event charged to that cycle, and realized_pnl_quote = dca_net_pnl_quote - funding_paid_quote.",
    "funding_events_csv": "one row per charged funding event, in charge order: the cycle it hit, the registered funding_time, the UTC-hour boundary used, the rate, the mark price and where that mark came from, held_base_qty at that instant, and payment_quote (positive = paid out of PnL). A run that held nothing during the window writes the header only.",
    "performance_claimable": "true only when the applied economics are proven equivalent to an approved research cost model - every declared cost is an explicit finite JSON number (null, empty, boolean, NaN/Infinity and string costs are refused, never read as zero), the declared fee equals the overlay's trade_cost, the declared slippage is exactly 0.0, funding is applied from a source identity that the runtime itself re-verified including both sha256 digests, and funding reconciles with cycle PnL and account equity - AND every required promotion metric is defined (CAGR, Sharpe) AND at least one cycle was replayed. Approval alone is not enough, and an unverified or absent funding source is never treated as zero. A declared not_modeled funding block therefore keeps this false.",
}


def bar_returns(equity, initial_capital):
    """Return per-bar simple returns, including the first bar versus frozen initial capital."""
    base = float(initial_capital)
    if base <= 0:
        raise SystemExit("HOLD: frozen initial capital must be positive")
    out = []
    prior = base
    for value in equity:
        value = float(value)
        if prior <= 0:
            raise SystemExit("HOLD: equity return base became non-positive")
        out.append(value / prior - 1.0)
        prior = value
    return out


def max_drawdown_pct(equity, initial_capital):
    """Return peak-to-trough drawdown with the frozen initial capital as the first peak."""
    peak = float(initial_capital)
    if peak <= 0:
        raise SystemExit("HOLD: frozen initial capital must be positive")
    worst = 0.0
    for value in equity:
        value = float(value)
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return worst


def utc_ms(moment):
    """Return epoch milliseconds of one naive-UTC datetime, immune to the host timezone.

    Every timestamp in this engine is naive UTC: the frozen provider rows, the
    window bounds and the Binance funding series. ``datetime.timestamp()`` on a
    naive value reads it as local time, which on a UTC+8 host would shift every
    funding comparison by 8h, so the offset is stated explicitly instead.
    """
    return int(round(moment.replace(tzinfo=timezone.utc).timestamp() * 1000))


def bar_close_ms(rows, index, timeframe):
    """Return the CLOSE boundary in ms of one artifact row.

    A Qlib row label is the bar OPEN. The funding ordering is defined at the bar
    CLOSE, so every eligibility comparison in this module goes through this
    boundary and never through ``rows[index]["datetime"]`` directly. Deriving it
    from the timeframe is also what makes a 4h or 1d bar correct: an event inside
    a bar is charged on that bar's close, not on its label.
    """
    if timeframe not in TIMEFRAME_MS:
        raise SystemExit("HOLD: timeframe %r has no frozen bar interval" % (timeframe,))
    return utc_ms(rows[index]["datetime"]) + TIMEFRAME_MS[timeframe]


def funding_iso(moment_ms):
    """Return one epoch-millisecond instant as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(moment_ms / 1000.0, tz=timezone.utc).isoformat()


def funding_boundary_ms(funding_time_ms):
    """Return the UTC-hour boundary one funding event belongs to.

    Registered ``modeled_funding`` rows carry sub-second jitter (+6 ms observed)
    and official rows are exact. Both name a UTC hour open, which is also the
    ``open_time_ms`` of the 1h mark bar the modeled reconstruction must read, so
    the jitter is floored here once instead of being compared raw.
    """
    if not isinstance(funding_time_ms, int) or isinstance(funding_time_ms, bool):
        raise SystemExit("HOLD: funding row has no integer funding_time_ms")
    return funding_time_ms - funding_time_ms % FUNDING_HOUR_MS


def funding_mark_root(source_ref, symbol):
    """Return the 1h mark root that belongs to one funding source path.

    The registered layout puts the two series side by side::

        <usdm>/funding/<SYMBOL>/<SYMBOL>-funding.jsonl.gz
        <usdm>/mark/<SYMBOL>/1h/<SYMBOL>-1h-YYYY-MM.jsonl.gz

    so the mark root is derived from the source path the identity already pins
    rather than from a second config knob that could disagree with it.
    """
    return Path(source_ref).parents[2] / "mark" / symbol / "1h"


def read_jsonl_gz(path, what):
    """Return the objects of one gzipped JSONL file. Fail closed on any bad line."""
    import gzip

    path = Path(path)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise SystemExit("HOLD: %s is not readable: %s (%s)" % (what, path, exc))
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise SystemExit("HOLD: %s line %d is not JSON: %s (%s)" % (what, number, path, exc))
    if not rows:
        raise SystemExit("HOLD: %s holds no rows: %s" % (what, path))
    return rows


def finite_float(value, what):
    """Return one finite float, or fail closed naming the exact field.

    A JSON boolean is a malformed financial value, never 0.0/1.0: ``float``
    silently converts it, so it is refused *before* any conversion. A decimal
    string stays accepted and unambiguous, because the registered source format
    carries rates as strings and a string is not a falsy cost assumption.
    """
    if isinstance(value, bool):
        raise SystemExit("HOLD: funding %s is a JSON boolean, not a number: %r" % (what, value))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise SystemExit("HOLD: funding %s is not numeric: %r" % (what, value))
    if not math.isfinite(number):
        raise SystemExit("HOLD: funding %s is not finite: %r" % (what, value))
    return number


def positive_float(value, what):
    """Return one finite positive float, or fail closed naming the exact field."""
    number = finite_float(value, what)
    if number <= 0.0:
        raise SystemExit("HOLD: funding %s is not positive: %r" % (what, value))
    return number


def modeled_mark_snapshot(files):
    """Return the deterministic digest of the exact monthly mark files used.

    ``files`` maps a POSIX path relative to ``<usdm>/mark`` to that file's
    sha256. The digest is the sha256 of the sorted ``"<relpath>:<sha256>"`` lines
    joined by newlines, so it depends on exactly which months were read and on
    their bytes, and on nothing about where they were mounted.
    """
    lines = ["%s:%s" % (name, files[name]) for name in sorted(files)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def load_funding_series(identity, symbol, window_start_ms, window_end_ms):
    """Return the verified Binance USD-M funding series inside the evaluation window.

    ``window_start_ms``/``window_end_ms`` are the first and last evaluation bar
    CLOSE boundaries, so the window is exactly the set of boundaries a replay
    could charge. Fail closed, never zero-fill: a wrong source sha256, a wrong
    modeled-mark snapshot, a duplicate timestamp, an unregistered ``truth_status``,
    a non-finite rate or price, and a modeled row whose 1h mark bar is absent are
    all HOLDs.

    The full in-window series is returned, modeled segment included; nothing is
    dropped because it is not official, and a gap is never read as zero funding.
    """
    problems = problems_in_funding_identity(identity)
    if problems:
        raise SystemExit("HOLD: funding identity is not usable: " + "; ".join(problems))
    source = Path(identity["source_ref"])
    if not source.is_file():
        raise SystemExit("HOLD: funding source is not readable: " + str(source))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != identity["source_sha256"]:
        raise SystemExit("HOLD: funding source sha256 mismatch: %s != %s" % (actual, identity["source_sha256"]))
    seen_stamps, seen_boundaries, staged = set(), set(), []
    for row in read_jsonl_gz(source, "funding source"):
        if not isinstance(row, dict):
            raise SystemExit("HOLD: funding source row is not an object")
        if row.get("symbol") != symbol:
            raise SystemExit(
                "HOLD: funding source symbol %r does not match the cell symbol %r" % (row.get("symbol"), symbol))
        stamp = row.get("funding_time_ms")
        boundary = funding_boundary_ms(stamp)
        if stamp in seen_stamps:
            raise SystemExit("HOLD: duplicate funding timestamp in the source: %d" % stamp)
        if boundary in seen_boundaries:
            raise SystemExit("HOLD: two funding events share the UTC hour boundary %d" % boundary)
        seen_stamps.add(stamp)
        seen_boundaries.add(boundary)
        status = row.get("truth_status")
        if status not in FUNDING_TRUTH_STATUS:
            raise SystemExit("HOLD: funding row %d has truth_status %r, outside the frozen %s"
                             % (stamp, status, list(FUNDING_TRUTH_STATUS)))
        rate = finite_float(row.get("funding_rate"), "funding_rate at %d" % stamp)
        if not (window_start_ms <= boundary <= window_end_ms):
            continue
        staged.append({
            "funding_time_ms": stamp,
            "boundary_ms": boundary,
            "funding_rate": rate,
            "truth_status": status,
            "mark_price": row.get("mark_price"),
            "mark_price_source": row.get("funding_price_source"),
        })
    mark_root = funding_mark_root(source, symbol)
    month_opens, required, events = {}, {}, []
    for event in staged:
        if event["truth_status"] == "official":
            mark = positive_float(event["mark_price"], "official mark_price at %d" % event["funding_time_ms"])
            source_note = event["mark_price_source"] or "binance_funding_history_mark_price"
        else:
            month = datetime.fromtimestamp(event["boundary_ms"] / 1000.0, tz=timezone.utc)
            name = "%s-1h-%04d-%02d.jsonl.gz" % (symbol, month.year, month.month)
            path = mark_root / name
            if name not in month_opens:
                if not path.is_file():
                    raise SystemExit("HOLD: modeled funding has no 1h mark source: " + str(path))
                opens = {}
                for bar in read_jsonl_gz(path, "1h mark source"):
                    try:
                        stamp_ms, open_price = int(bar["open_time_ms"]), bar["open"]
                    except (KeyError, TypeError, ValueError):
                        raise SystemExit("HOLD: 1h mark row is malformed in " + str(path))
                    if stamp_ms in opens:
                        raise SystemExit("HOLD: duplicate 1h mark open_time %d in %s" % (stamp_ms, path))
                    opens[stamp_ms] = open_price
                month_opens[name] = opens
                required[str(path.relative_to(mark_root.parents[1]))] = hashlib.sha256(path.read_bytes()).hexdigest()
            if event["boundary_ms"] not in month_opens[name]:
                raise SystemExit("HOLD: modeled funding has no 1h mark bar at %d (%s)"
                                 % (event["boundary_ms"], path))
            mark = positive_float(month_opens[name][event["boundary_ms"]],
                                  "modeled mark open at %d" % event["funding_time_ms"])
            source_note = FUNDING_MODELED_MARK_BASIS
        events.append(dict(event, mark_price=mark, mark_price_source=source_note))
    digest = modeled_mark_snapshot(required)
    if digest != identity["modeled_mark_snapshot_sha256"]:
        raise SystemExit("HOLD: modeled mark snapshot sha256 mismatch: %s != %s"
                         % (digest, identity["modeled_mark_snapshot_sha256"]))
    events.sort(key=lambda item: item["boundary_ms"])
    counts = {}
    for event in events:
        counts[event["truth_status"]] = counts.get(event["truth_status"], 0) + 1
    return {
        "identity": {
            "basis": FUNDING_BASIS,
            "source_ref": str(source),
            "source_sha256": actual,
            "truth_status": list(FUNDING_TRUTH_STATUS),
            "modeled_mark_basis": FUNDING_MODELED_MARK_BASIS,
            "modeled_mark_snapshot_sha256": digest,
        },
        "events": events,
        "truth_status_counts": counts,
        "mark_snapshot": dict(sorted(required.items())),
    }


def declared_cost_number(block, key, where):
    """Return (number, None) for one declared cost, or (None, reason).

    The value must be an explicit finite JSON number. ``None``, an empty or
    non-numeric string, a JSON boolean, NaN/Infinity and a numeric string are all
    declarations that carry no number, and this replay never invents 0.0 to fill
    one: a falsy value is not a cost assumption, it is a missing measurement.
    """
    if not isinstance(block, dict) or key not in block:
        return None, "%s does not declare %s" % (where, key)
    value = block[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "%s.%s is not an explicit finite JSON number: %r" % (where, key, value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, "%s.%s is not a finite JSON number: %r" % (where, key, value)
    if not math.isfinite(number):
        return None, "%s.%s is not finite: %r" % (where, key, value)
    return number, None


def applied_cost_matches_model(cfg, applied=None):
    """Return (matches, reason) for the applied economics against the declared model.

    Two costs are applied: the existing DCA primitive's frozen semantics
    (``2 * house_overlay.trade_cost`` of each filled stage's quote amount, no
    slippage anywhere), and the funding series a verified source identity names.

    Every clause is a proof obligation, and none of them is satisfied by a
    declaration alone:

    - the funding model must be a structured identity, not a zero or a label;
    - the run must report the funding as applied AND reconciled;
    - the identity the run actually resolved must equal the declared one key by
      key, including both sha256 digests, so a stale manifest cannot claim a run
      that read different bytes;
    - the declared per-side fee must be the fee the overlay really charges;
    - every declared cost must be an explicit finite JSON number, and the only
      slippage the replay can match is exactly 0.0: an absent, null, empty,
      boolean or string cost is a missing measurement, never an implied zero.

    Do not weaken these to make a claim true: a claim that cannot survive them is
    not a smaller claim, it is a false one.
    """
    cost = cfg.get("research_cost_model")
    overlay = cfg.get("house_overlay")
    if not isinstance(cost, dict) or not isinstance(overlay, dict):
        return False, "research cost model or house overlay is not declared"
    if cost.get("approved") is not True:
        return False, "research cost model is not approved"
    declared = cost.get("funding")
    declared_problems = problems_in_funding_identity(declared)
    if declared_problems or not isinstance(declared, dict):
        return False, ("the declared funding model is not a proven source identity: "
                       + (declared_problems[0] if declared_problems else "not a JSON object"))
    if not isinstance(applied, dict):
        return False, ("funding is not applied to PnL, so no declared model can be proven equivalent "
                       "(no applied funding evidence was supplied)")
    if applied.get("funding_applied") is not True:
        return False, "funding is not applied to PnL, so no declared model can be proven equivalent"
    if applied.get("reconciled") is not True:
        return False, "funding was charged but does not reconcile with cycle PnL and account equity"
    runtime = applied.get("identity")
    runtime_problems = problems_in_funding_identity(runtime)
    if runtime_problems or not isinstance(runtime, dict):
        return False, ("the runtime funding identity is not usable: "
                       + (runtime_problems[0] if runtime_problems else "not a JSON object"))
    mismatch = [key for key in FUNDING_IDENTITY_KEYS if runtime[key] != declared[key]]
    if mismatch:
        return False, ("the declared funding identity does not match the source this run applied: "
                       + ",".join(mismatch))
    declared_costs = {}
    for where, block, key in (
        ("research_cost_model", cost, "fees_per_side"),
        ("research_cost_model", cost, "slippage"),
        ("house_overlay", overlay, "trade_cost"),
    ):
        number, problem = declared_cost_number(block, key, where)
        if problem:
            return False, problem
        declared_costs[key] = number
    if declared_costs["slippage"] != 0.0:
        return False, "slippage is declared but the replay applies none"
    if declared_costs["fees_per_side"] != declared_costs["trade_cost"]:
        return False, "the declared per-side fee is not the fee the overlay applies"
    return True, (
        "the declared model equals the applied economics: 2 * house_overlay.trade_cost per filled stage and "
        "%d verified funding event(s) totalling %.12g quote from %s"
        % (int(applied.get("event_count") or 0), float(applied.get("total_payment_quote") or 0.0),
           runtime.get("basis"))
    )


def performance_claimable(cfg, metrics, applied=None):
    """True only when applied economics match an approved model and metrics are defined."""
    matches, _ = applied_cost_matches_model(cfg, applied)
    return bool(
        matches
        and metrics.get("total_trades", 0) > 0
        and metrics.get("cagr_pct") is not None
        and metrics.get("sharpe") is not None
    )


def funding_reconciles(records, metrics, initial_capital, total_funding):
    """Return True only when every funding payment is visible in cycle PnL and equity.

    Four identities must hold at once, so a payment that lands in the evidence but
    not in the PnL (or the other way round) can never pass as reconciled:

    - each cycle paid exactly the sum of the events attributed to it;
    - each cycle's net realized PnL is its own DCA result minus exactly those events;
    - the run total is the sum of every individual payment;
    - net profit is the sum of cycle net PnL, and end equity is initial + net profit.
    """
    tolerance = 1e-9 * max(1.0, abs(float(initial_capital)))
    charged = 0.0
    for record in records:
        own = float(sum(charge["payment_quote"] for charge in record["funding_events"]))
        charged += own
        if abs(own - float(record["funding_paid_quote"])) > tolerance:
            return False
        if abs(float(record["realized_pnl_quote"]) - (float(record["dca_net_pnl_quote"]) - own)) > tolerance:
            return False
    if abs(charged - float(total_funding)) > tolerance:
        return False
    realized_total = float(sum(float(record["realized_pnl_quote"]) for record in records))
    if abs(float(metrics["net_profit_usdt"]) - realized_total) > tolerance:
        return False
    if abs(float(metrics["end_equity_quote"]) - (float(initial_capital) + realized_total)) > tolerance:
        return False
    return True


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
    """Return the DCA input frame for one cycle: entry bar through close boundary.

    ``timestamp`` is the primitive's own epoch-seconds column and stays exactly as
    PR #12 shipped it. Funding never reads it: stage fills are mapped back to a row
    offset and priced at ``bar_close_ms``, so a naive ``.timestamp()`` (which reads
    a naive UTC datetime as host-local) can never leak into a funding boundary.
    """
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


def cycle_funding_charges(rows, timeframe, prices, amounts, ledger, entry_index, close_index, funding):
    """Return the funding events one cycle must pay, in boundary order.

    Position state is reconstructed from the DCA ledger itself: a stage column
    becomes non-zero on the bar whose close triggered that stage, so its first
    non-zero offset is the fill bar and every quantity and boundary is derived from
    that bar rather than from the ledger's epoch column.

    Pinned ordering (both sides at bar CLOSEs):

    - a stage first filled at close T is not charged for an event at T, because it
      was not held during the interval the event pays for;
    - a cycle closing at close T is charged for an event at T, because it was held
      through that interval;
    - an event after the cycle's close boundary is never charged: the account is
      flat from that boundary on, whatever the source state says.
    """
    if not funding or not funding.get("events"):
        return []
    close_boundary = bar_close_ms(rows, close_index, timeframe)
    fill_closes = []
    for level in range(len(prices)):
        column = ledger.get("filled_amount_quote_%d" % level)
        offset = None
        if column is not None:
            for candidate in range(len(column)):
                if float(column.iloc[candidate]) > 0.0:
                    offset = candidate
                    break
        fill_closes.append(None if offset is None else bar_close_ms(rows, entry_index + offset, timeframe))
    charges = []
    for event in funding["events"]:
        boundary = event["boundary_ms"]
        if boundary > close_boundary:
            continue
        held_base = 0.0
        for level, (price, amount) in enumerate(zip(prices, amounts)):
            if fill_closes[level] is None or fill_closes[level] >= boundary:
                continue
            held_base += float(amount / price)
        if held_base <= 0.0:
            continue
        charge_index = None
        for candidate in range(entry_index, close_index + 1):
            if bar_close_ms(rows, candidate, timeframe) >= boundary:
                charge_index = candidate
                break
        if charge_index is None:
            raise SystemExit("HOLD: funding event at %d has no boundary inside its own cycle" % boundary)
        charges.append(
            {
                "funding_time_ms": event["funding_time_ms"],
                "funding_time_utc": funding_iso(event["funding_time_ms"]),
                "boundary_ms": boundary,
                "boundary_utc": funding_iso(boundary),
                "funding_rate": event["funding_rate"],
                "mark_price": event["mark_price"],
                "mark_price_source": event["mark_price_source"],
                "truth_status": event["truth_status"],
                "held_base_qty": held_base,
                "payment_quote": held_base * event["mark_price"] * event["funding_rate"],
                "charge_index": charge_index,
                "charge_bar_ts": rows[charge_index]["datetime"],
            }
        )
    return charges


def replay_cycles(rows, cycles, overlay, leverage, timeframe=None, simulator=None, funding=None):
    """Run one house DCA overlay per source cycle and return the cycle records.

    ``timeframe`` is required whenever funding is charged: the bar CLOSE boundary
    comes from the frozen interval, and guessing the interval would silently move
    every payment. ``record["realized_pnl_quote"]`` is net of the funding the cycle
    paid; ``record["dca_net_pnl_quote"]`` keeps the primitive's own number.
    """
    if funding:
        bar_close_ms(rows, 0, timeframe)  # fail closed on a missing/unknown timeframe
    records = []
    for number, cycle in enumerate(cycles, start=1):
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
        dca_realized = float(ledger["net_pnl_quote"].iloc[-1])
        overlay_exit = dca_close_type in OVERLAY_CLOSE_TYPES
        dca_pnl_by_index = {
            entry_index + offset: float(ledger["net_pnl_quote"].iloc[offset]) for offset in range(len(ledger))
        }
        # The overlay is flat from its own exit bar on; the source stays long
        # until its exit, so the cycle's realized PnL is simply carried.
        for index in range(close_index + 1, exit_index + 1):
            dca_pnl_by_index[index] = dca_realized
        charges = cycle_funding_charges(
            rows, timeframe, prices, amounts, ledger, entry_index, close_index, funding
        )
        funding_paid = float(sum(charge["payment_quote"] for charge in charges))
        funding_by_index = {}
        for charge in charges:
            funding_by_index[charge["charge_index"]] = (
                funding_by_index.get(charge["charge_index"], 0.0) + charge["payment_quote"]
            )
        pnl_by_index, funding_cum_by_index, running = {}, {}, 0.0
        for index in range(entry_index, exit_index + 1):
            running += funding_by_index.get(index, 0.0)
            funding_cum_by_index[index] = running
            pnl_by_index[index] = dca_pnl_by_index[index] - running
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
                "cycle": number,
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
                "dca_net_pnl_quote": dca_realized,
                "funding_paid_quote": funding_paid,
                "realized_pnl_quote": dca_realized - funding_paid,
                "max_exposure_quote": float(max(ledger["filled_amount_quote"])),
                "ledger_rows": int(len(ledger)),
                "stage_prices": [float(value) for value in prices],
                "stage_quote_amounts": [float(value) for value in amounts],
                "stages": stages,
                "funding_events": charges,
                "funding_cum_by_index": funding_cum_by_index,
                "pnl_by_index": pnl_by_index,
            }
        )
    return records


def equity_projection(eval_indices, records, initial_capital):
    """Return one account-equity bar per evaluation-window bar.

    Realized net PnL of finished cycles carries forward, the open cycle contributes
    its own DCA ledger mark to market minus the funding it has been charged so far,
    and every bar of the window is present so a reader can see flat periods.
    ``filled_amount_quote`` never enters here.
    """
    projection, carried, carried_funding, cursor = [], 0.0, 0.0, 0
    for bar in eval_indices:
        while cursor < len(records) and records[cursor]["exit_index"] < bar:
            carried += records[cursor]["realized_pnl_quote"]
            carried_funding += records[cursor]["funding_paid_quote"]
            cursor += 1
        position, number, mark, funding = "flat", None, 0.0, 0.0
        if cursor < len(records) and records[cursor]["entry_index"] <= bar:
            record = records[cursor]
            number = cursor + 1
            mark = record["pnl_by_index"].get(bar, record["realized_pnl_quote"])
            funding = record["funding_cum_by_index"].get(bar, record["funding_paid_quote"])
            position = "long" if bar <= record["close_index"] else "source_long"
        projection.append(
            {
                "equity_quote": initial_capital + carried + mark,
                "position_state": position,
                "cycle_number": number,
                "realized_pnl_quote": carried,
                "mark_pnl_quote": mark,
                "funding_paid_quote": carried_funding + funding,
            }
        )
    for bar, value in zip(
        projection,
        bar_returns([item["equity_quote"] for item in projection], initial_capital),
    ):
        bar["bar_return"] = value
    return projection


def metrics_from_equity(timestamps, equity, realized_pnls, periods_per_year, initial_capital,
                        funding_paid_quote=None):
    """Return required fee-and-funding-inclusive metrics using frozen initial capital as the baseline."""
    if not equity or len(timestamps) != len(equity):
        raise SystemExit("HOLD: the evaluation window produced no usable equity series")
    start = float(initial_capital)
    end = float(equity[-1])
    if start <= 0:
        raise SystemExit("HOLD: frozen initial capital must be positive")
    elapsed = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0.0
    returns = bar_returns(equity, start)
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
    metrics = {
        "net_return_pct": (end / start - 1.0) * 100.0,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "max_drawdown_pct": max_drawdown_pct(equity, start),
        "win_rate_pct": (100.0 * len(winners) / total) if total else None,
        "total_trades": total,
        "profit_factor": (sum(winners) / gross_loss) if losers and gross_loss > 0 else None,
        "net_profit_usdt": sum(realized_pnls),
        "start_equity_quote": start,
        "first_observed_equity_quote": float(equity[0]),
        "end_equity_quote": end,
        "elapsed_seconds": elapsed,
        "evaluation_bars": len(equity),
        "periods_per_year": periods_per_year,
        "bar_returns_ddof": 1,
        "max_drawdown_definition": REPLAY_DEFINITIONS["max_drawdown_pct"],
        "definitions": dict(REPLAY_DEFINITIONS),
    }
    if funding_paid_quote is not None:
        metrics["funding_paid_quote"] = float(funding_paid_quote)
    return metrics


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
        "dca_net_pnl_quote",
        "funding_paid_quote",
        "realized_pnl_quote",
        "max_exposure_quote",
        "ledger_rows",
    ]
    for level in range(stage_count):
        names += ["stage_%d_price" % level, "stage_%d_quote" % level, "stage_%d_filled_quote" % level]
    return names


def funding_event_fieldnames():
    """Return the funding_events.csv header, in the order the evidence is read."""
    return [
        "cycle",
        "funding_time_utc",
        "funding_time_ms",
        "funding_boundary_utc",
        "funding_boundary_ms",
        "funding_rate",
        "mark_price",
        "mark_price_source",
        "truth_status",
        "held_base_qty",
        "payment_quote",
        "charge_bar_utc",
    ]


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
                "dca_net_pnl_quote": record["dca_net_pnl_quote"],
                "funding_paid_quote": record["funding_paid_quote"],
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
                "funding_paid_quote",
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
                    "funding_paid_quote": bar["funding_paid_quote"],
                    "position_state": bar["position_state"],
                    "cycle_number": "" if bar["cycle_number"] is None else bar["cycle_number"],
                }
            )
    # One row per charged funding event. A cell that held nothing during the window
    # writes the header only, so the file is never mistaken for a zero-fill.
    with (out_dir / "funding_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=funding_event_fieldnames())
        writer.writeheader()
        for number, record in enumerate(records, start=1):
            for charge in record["funding_events"]:
                writer.writerow(
                    {
                        "cycle": number,
                        "funding_time_utc": charge["funding_time_utc"],
                        "funding_time_ms": charge["funding_time_ms"],
                        "funding_boundary_utc": charge["boundary_utc"],
                        "funding_boundary_ms": charge["boundary_ms"],
                        "funding_rate": charge["funding_rate"],
                        "mark_price": charge["mark_price"],
                        "mark_price_source": charge["mark_price_source"],
                        "truth_status": charge["truth_status"],
                        "held_base_qty": charge["held_base_qty"],
                        "payment_quote": charge["payment_quote"],
                        "charge_bar_utc": charge["charge_bar_ts"].isoformat(),
                    }
                )
    return ["summary.json", "signals.csv", "trades.csv", "events.csv", "funding_events.csv",
            "equity.csv", "metrics.json"]


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
            # Every scope statement here is persisted in effective_config.json, in
            # summary.json and (via the repro artifacts) in MLflow, so it must
            # describe exactly what this replay does. Funding is applied, not
            # "unmodeled": only the isolated-margin account mechanics and slippage
            # are absent from the Qlib replay.
            "research_leverage_scope": "exposure multiplier only: research_leverage scales every DCA stage's "
                                       "quote allocation, so it scales exposure and PnL; it is not a margin "
                                       "balance, a leverage maintenance rule or a liquidation model",
            "research_leverage_is_exposure_multiplier": True,
            "isolated_margin": "not modeled in this Qlib replay: no isolated-margin balance, maintenance "
                               "margin, liquidation price or position limit is simulated, so a replay can "
                               "never be read as a margin-account result",
            "slippage": "not applied: every fill uses the frozen bar-close reference price of the DCA "
                        "primitive, so a declared research cost model is equivalent only when its "
                        "slippage is exactly 0.0 and no other slippage is ever assumed",
            "funding": "applied from a verified structured source identity: when research_cost_model."
                       "funding is the binance-usdm-funding-v1 identity, every held-through event is "
                       "charged into the same bar's equity and the identity is re-verified against the "
                       "source bytes; when it is the explicit not_modeled declaration the cell stays "
                       "runnable with funding absent from PnL and performance_claimable false, and "
                       "funding is never zero-filled",
            "funding_scope": "conditional_verified_identity_else_not_modeled_unclaimable",
            "final_execution_and_certification": "Node13 Hummingbot owns canonical execution economics, "
                                                 "live fee/funding rates and certification; this replay is "
                                                 "research evidence, not the certified economics",
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
    # Funding: the frozen window is the set of bar CLOSE boundaries a cycle can be
    # charged at, so the series is read once for exactly that interval. An identity
    # must resolve or the run holds; the explicit ``not_modeled`` declaration keeps
    # a legacy cell runnable, unclaimable, and honest about it.
    declared_funding = (cfg.get("research_cost_model") or {}).get("funding")
    funding = None
    if isinstance(declared_funding, dict):
        funding = load_funding_series(
            declared_funding,
            symbol,
            bar_close_ms(rows, eval_indices[0], timeframe),
            bar_close_ms(rows, eval_indices[-1], timeframe),
        )
    records = replay_cycles(rows, cycles, overlay, cfg["research_leverage"], timeframe,
                            simulator=simulator, funding=funding)
    projection = equity_projection(eval_indices, records, float(overlay["initial_capital_quote"]))
    evaluation_times = [rows[index]["datetime"] for index in eval_indices]
    realized = [record["realized_pnl_quote"] for record in records]
    total_funding = float(sum(record["funding_paid_quote"] for record in records))
    initial_capital = float(overlay["initial_capital_quote"])
    metrics = metrics_from_equity(
        evaluation_times,
        [bar["equity_quote"] for bar in projection],
        realized,
        PERIODS_PER_YEAR[timeframe],
        initial_capital,
        funding_paid_quote=total_funding,
    )
    applied = {
        "funding_applied": funding is not None,
        "identity": None if funding is None else funding["identity"],
        "event_count": int(sum(len(record["funding_events"]) for record in records)),
        "total_payment_quote": total_funding,
        "reconciled": funding_reconciles(records, metrics, initial_capital, total_funding),
    }
    if not applied["reconciled"]:
        raise SystemExit("HOLD: funding was charged but does not reconcile with cycle PnL and equity")
    cost_matches, cost_reason = applied_cost_matches_model(cfg, applied)
    claimable = performance_claimable(cfg, metrics, applied)
    close_reasons = {}
    for record in records:
        close_reasons[record["close_reason"]] = close_reasons.get(record["close_reason"], 0) + 1
    charged_status = {}
    for record in records:
        for charge in record["funding_events"]:
            charged_status[charge["truth_status"]] = charged_status.get(charge["truth_status"], 0) + 1
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
        "funding_paid_quote": total_funding,
        "funding_charged_events": applied["event_count"],
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
                "dca_net_pnl_quote": record["dca_net_pnl_quote"],
                "funding_paid_quote": record["funding_paid_quote"],
                "funding_events": len(record["funding_events"]),
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
                       "stage's entry bar onward (frozen primitive semantics; not per-fill event timing), "
                       "plus every verified Binance USD-M funding event this cycle held through",
            "slippage_applied": False,
            "funding_applied": applied["funding_applied"],
            "funding_declaration": declared_funding if not isinstance(declared_funding, dict)
            else "structured source identity",
            "funding_total_payment_quote": total_funding,
            "funding_charged_event_count": applied["event_count"],
            "funding_reconciled": applied["reconciled"],
            "funding_truth_status_counts": {} if funding is None else funding["truth_status_counts"],
            "funding_charged_truth_status_counts": charged_status,
            "funding_basis": None if funding is None else funding["identity"]["basis"],
            "funding_source_ref": None if funding is None else funding["identity"]["source_ref"],
            "funding_source_sha256": None if funding is None else funding["identity"]["source_sha256"],
            "funding_modeled_mark_basis": None if funding is None else funding["identity"]["modeled_mark_basis"],
            "funding_modeled_mark_snapshot_sha256": None
            if funding is None else funding["identity"]["modeled_mark_snapshot_sha256"],
            "funding_modeled_mark_files": {} if funding is None else funding["mark_snapshot"],
            "declared_model_matches_applied": cost_matches,
            "declared_model_match_reason": cost_reason,
        },
        "metrics_scope": "metrics.json holds the fee-and-funding-inclusive account-equity metrics and their "
                         "definitions",
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
                       "funding_basis": str(summary["applied_economics"]["funding_basis"]),
                       "funding_source_sha256": str(summary["applied_economics"]["funding_source_sha256"]),
                       "funding_modeled_mark_snapshot_sha256":
                           str(summary["applied_economics"]["funding_modeled_mark_snapshot_sha256"]),
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
        "funding_paid_quote": total_funding,
        "funding_charged_events": applied["event_count"],
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
