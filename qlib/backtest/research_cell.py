"""Config-driven formal research cell: frozen signal + house DCA overlay x research_leverage.

One cell is one frozen identity (symbol, timeframe, research_leverage) of the
round0 Formal Baseline. This file is the frozen cell entrypoint: the
materializer copies it byte-identically next to each cell's
effective_config.json, and the thin transport runs it inside the fixed Qlib
runtime as ``python <cell>/run_cell.entrypoint.py`` with ``QLIB_OUT_DIR`` set.

Boundary, deliberately narrow:

- ``research_leverage`` multiplies the house quote allocation only (base quote
  allocation x research_leverage). It is a research exposure multiplier. It is
  not a Binance liquidation, isolated-margin, funding or account-equity model;
  canonical economics stay with the Node13 Hummingbot certification lane.
- DCA stage semantics reuse the exported Qlib <-> Hummingbot DCA path
  (``qlib.backtest.backtest_hummingbot_dca``) over the frozen provider.
- Every overlay parameter comes from the cell config. This file invents none:
  a missing approved value is a fail-closed error, never a default.

Verbs::

    python research_cell.py run                     # run this cell (default)
    python research_cell.py materialize --spec S.json --root /results/_cells

ponytail: stdlib only at import time. numpy/pandas/qlib/mlflow load inside the
run path, so the deterministic planner (identities, config validation, stage
scaling, materialization) stays testable without the runtime.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from decimal import Decimal
from pathlib import Path

SCHEMA = "qlib-formal-research-cell-v1"
#: Frozen round0 coverage. Symbols/timeframes/leverages are the 42-cell set.
CELL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT")
CELL_TIMEFRAMES = ("1h", "4h", "1d")
CELL_LEVERAGES = (3, 5)
QLIB_FREQ = {"1h": "60min", "4h": "240min", "1d": "day"}
ENTRYPOINT_NAME = "run_cell.entrypoint.py"
CONFIG_NAME = "effective_config.json"

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
        for key in ("entry_len", "exit_len"):
            value = strategy.get(key)
            if not isinstance(value, int) or value <= 0:
                bad.append("strategy %s must be a positive integer" % key)
        if not strategy.get("window_start_utc") or not strategy.get("window_end_utc_inclusive"):
            bad.append("strategy window bounds are missing")
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


def donchian_state(high, low, close, times, entry_len, exit_len, window_start, window_end):
    """Return the frozen Donchian state machine rows over aligned OHLC arrays.

    Completed-bar decision: the prior-N channel excludes the current bar, which
    keeps the rule identical to the pinned source semantics.
    """
    state = False
    entry_ts = None
    rows = []
    last = len(close)
    for i in range(last):
        prior_high = max(high[i - entry_len:i]) if i >= entry_len else None
        prior_low = min(low[i - exit_len:i]) if i >= exit_len else None
        in_window = window_start <= times[i] <= window_end
        break_up = prior_high is not None and close[i] > prior_high
        break_down = prior_low is not None and close[i] < prior_low
        if break_up and break_down:
            raise ValueError("channel collision at bar %d" % i)
        entry = exit_ = False
        if not state and in_window and break_up:
            state, entry, entry_ts = True, True, times[i]
        elif state and break_down:
            state, exit_ = False, True
        rows.append(
            {
                "datetime": times[i],
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "prior_high": None if prior_high is None else float(prior_high),
                "prior_low": None if prior_low is None else float(prior_low),
                "in_window": int(in_window),
                "break_up": int(break_up),
                "break_down": int(break_down),
                "state": int(state),
                "entry_transition": int(entry),
                "exit_transition": int(exit_),
            }
        )
    return rows, entry_ts


def build_effective_config(spec, identity):
    """Return one cell's effective_config.json content from the campaign spec."""
    entry_dir = "%s/%s" % (spec["materialize"]["cells_dir"], identity["cell_id"])
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
        "strategy": dict(spec["strategy"], symbol=identity["symbol"], timeframe=identity["timeframe"]),
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
    source = Path(__file__).resolve()
    source_bytes = source.read_bytes()
    cells = []
    for identity in campaign_identities():
        cell_dir = Path(root) / identity["cell_id"]
        if cell_dir.exists() and not force:
            raise SystemExit("refuse overwrite: " + str(cell_dir))
        cell_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, cell_dir / ENTRYPOINT_NAME)
        config = build_effective_config(spec, identity)
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
                "effective_config_ref": config["execution"]["entrypoint"].rsplit("/", 1)[0] + "/" + CONFIG_NAME,
                "effective_config_sha256": config_sha256,
                "qlib_runtime_commit": spec["runtime"]["commit"],
                "capabilities": {"execution": ["qlib-native-backtest"], "data": ["btcusdt-1h"], "timing": ["bar-close-1h"]},
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
    warmup_start = window_start - pd.Timedelta(strategy["entry_len"], unit="D")

    qlib.init(provider_uri=cfg["data"]["provider_ref"], region="us",
              expression_cache=None, dataset_cache=None)
    features = D.features([symbol], ["$high", "$low", "$close"],
                          start_time=str(warmup_start), end_time=str(run_end), freq=freq).sort_index()
    if features.empty:
        raise SystemExit("HOLD: frozen provider returned no rows")
    times = [ts.to_pydatetime() for ts in features.index.get_level_values("datetime")]
    rows, entry_ts = donchian_state(
        features["$high"].astype(float).to_numpy(),
        features["$low"].astype(float).to_numpy(),
        features["$close"].astype(float).to_numpy(),
        times,
        strategy["entry_len"],
        strategy["exit_len"],
        window_start.to_pydatetime(),
        window_end.to_pydatetime(),
    )
    eval_rows = [row for row in rows if window_start.to_pydatetime() <= row["datetime"] <= run_end.to_pydatetime()]
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
                       "entry_len": str(strategy["entry_len"]), "exit_len": str(strategy["exit_len"])}.items():
        client.log_param(run_id, key, value)
    for key, value in {"max_exposure_quote": exposure_quote, "net_pnl_quote": net_pnl_quote,
                       "ledger_rows": summary["ledger_rows"]}.items():
        client.log_metric(run_id, key, float(value))
    for artifact in (summary_path, out_dir / "events.csv", out_dir / "signals.csv"):
        client.log_artifact(run_id, str(artifact), artifact_path="backtest")
    for artifact in (cell_dir / CONFIG_NAME, Path(__file__)):
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
