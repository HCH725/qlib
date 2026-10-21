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
        # entry_ts: datetime of the single entry this run evaluates, or None.

``volume`` is a float array, or None when the frozen provider has no volume
field; an artifact that needs volume must fail closed on None.

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
    """Execute one cell and write its evidence into out_dir. Return the contract dict."""
    import importlib.metadata
    import csv

    import numpy as np
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
    symbol, freq = strategy["symbol"], QLIB_FREQ[strategy["timeframe"]]
    window_start = pd.Timestamp(strategy["window_start_utc"])
    window_end = pd.Timestamp(strategy["window_end_utc_inclusive"])
    run_end = pd.Timestamp(strategy["force_flat_bar_utc"]) if strategy.get("force_flat_bar_utc") else window_end
    warmup_start = window_start - pd.Timedelta(float(strategy.get("warmup_days", 0)), unit="D")
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
        window_start=window_start.to_pydatetime(),
        window_end=window_end.to_pydatetime(),
    )
    row_problems = problems_in_signal_rows(rows)
    if row_problems:
        raise SystemExit("HOLD: strategy artifact output is not runnable: " + "; ".join(row_problems))
    eval_rows = [row for row in rows if window_start.to_pydatetime() <= row["datetime"] <= run_end.to_pydatetime()]
    if not eval_rows:
        raise SystemExit("HOLD: frozen signal produced no rows inside the window")
    if entry_ts is None:
        raise SystemExit("HOLD: frozen signal produced no entry inside the window")
    overlay = cfg["house_overlay"]
    entry_price = next(row["close"] for row in rows if row["entry_transition"] == 1)
    prices = stage_prices(entry_price, overlay["so_spacing_pct"], overlay["safety_order_levels"])
    amounts = house_dca_amounts(
        overlay["base_order_quote_pct"],
        overlay["safety_order_quote_pct"],
        overlay["safety_order_levels"],
        overlay["initial_capital_quote"],
        cfg["research_leverage"],
    )
    frame_rows = [row for row in rows if row["datetime"] >= entry_ts]
    frame = pd.DataFrame(
        {
            "timestamp": [row["datetime"].timestamp() for row in frame_rows],
            "close": [row["close"] for row in frame_rows],
            "low": [row["low"] for row in frame_rows],
            "high": [row["high"] for row in frame_rows],
        }
    )
    frame.index = frame["timestamp"].to_numpy(dtype=float)
    ledger, close_type = dca_overlay_ledger(
        frame, prices, amounts,
        None if overlay["take_profit_pct"] is None else Decimal(str(overlay["take_profit_pct"])),
        None if overlay["stop_loss_pct"] is None else Decimal(str(overlay["stop_loss_pct"])),
        float(overlay["trade_cost"] or 0.0),
        simulator=simulator,
    )
    exposure_quote = float(max(ledger["filled_amount_quote"]))
    net_pnl_quote = float(ledger["net_pnl_quote"].iloc[-1])
    summary = {
        "engine": "qlib.backtest.research_cell / backtest_hummingbot_dca (house DCA overlay)",
        "qlib_version": importlib.metadata.version("pyqlib"),
        "symbol": symbol,
        "frequency": freq,
        "research_leverage": cfg["research_leverage"],
        "stage_prices": [float(value) for value in prices],
        "stage_quote_amounts": [float(value) for value in amounts],
        "max_exposure_quote": exposure_quote,
        "net_pnl_quote": net_pnl_quote,
        "close_type": close_type,
        "entry_ts_utc": str(entry_ts),
        "strategy_artifact_sha256": strategy["artifact"]["sha256"],
        "strategy_params": dict(strategy.get("params") or {}),
        "evaluation_rows_including_force_flat": len(eval_rows),
        "ledger_rows": int(len(ledger)),
        "economics_boundary": dict(cfg["economics_boundary"], research_cost_model=cfg["research_cost_model"],
                                   cost_model_approved=cost_model_claimable(cfg),
                                   performance_claimable=False),
    }
    out_dir.mkdir(parents=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "signals.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(eval_rows[0])
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(dict(row, datetime=pd.Timestamp(row["datetime"]).isoformat()))
    events = [{"stage": level, "stage_price": float(price), "stage_quote_amount": float(amount),
               "filled_quote": float(ledger["filled_amount_quote_%d" % level].iloc[-1])}
              for level, (price, amount) in enumerate(zip(prices, amounts))]
    with (out_dir / "events.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    mlruns = out_dir / "mlruns"
    mlruns.mkdir()
    tracking_uri = mlruns.resolve().as_uri()
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    client = MlflowClient(tracking_uri=tracking_uri)
    exp_id = client.create_experiment("formal-cell-" + cell_dir.name)
    client.create_run(exp_id, tags={"engine": "pyqlib", "cell_id": cell_dir.name,
                                    "research_leverage": str(cfg["research_leverage"]),
                                    "performance_claimable": "false"})
    run_id = client.search_runs(exp_id, order_by=["attributes.start_time DESC"], max_results=1)[0].info.run_id
    for key, value in {"qlib_version": summary["qlib_version"], "symbol": symbol, "frequency": freq,
                       "research_leverage": str(cfg["research_leverage"]),
                       "strategy_artifact_sha256": strategy["artifact"]["sha256"],
                       "strategy_params": json.dumps(dict(strategy.get("params") or {}), sort_keys=True)}.items():
        client.log_param(run_id, key, value)
    for key, value in {"max_exposure_quote": exposure_quote, "net_pnl_quote": net_pnl_quote,
                       "ledger_rows": summary["ledger_rows"]}.items():
        client.log_metric(run_id, key, float(value))
    for artifact in (summary_path, out_dir / "events.csv", out_dir / "signals.csv"):
        client.log_artifact(run_id, str(artifact), artifact_path="backtest")
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
    manifest = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and "mlruns" not in path.parts and path.name != "manifest.json":
            manifest[str(path.relative_to(out_dir))] = {
                "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "mlflow": readback_payload, "manifest": manifest}


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
