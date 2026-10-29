"""Checks for the config-driven formal research cell (shared shell + artifacts).

Run: python3 tests/backtest/test_research_cell.py   (or pytest tests/backtest)

These are implementation checks, not a campaign. They prove, on an isolated
local fixture:
- the frozen 42 identities are exactly 7 x 3 x 2, unique, and stable;
- a cell config with an unapproved overlay/cost value fails closed;
- materialization writes 42 bindable cell dirs, copies the one frozen strategy
  artifact into each of them, and emits a hash-consistent manifest;
- the generic engine carries no strategy rule: the same engine bytes run a
  Donchian artifact, a VZO band-recross artifact and a Bollinger upper-band
  breakout artifact, changing only the frozen artifact and its parameters;
- a missing, unreadable, out-of-cell-dir, malformed or hash-mismatched strategy
  artifact fails closed before any signal semantics are used;
- research_leverage scales the house quote exposure and the realized ledger by
  exactly 5/3 from 3x to 5x, reproducibly, through the exported
  Qlib <-> Hummingbot DCA path;
- the BO + 2 SO stage semantics of that path are not broken;
- a materialized cell replays the artifact's whole long-only transition series:
  every entry/exit pair becomes one house DCA cycle, the strategy exit is a real
  close boundary, an early overlay exit never invents a new source entry, and an
  impossible transition sequence fails closed;
- one fee-inclusive account equity series spans the fixed evaluation window with
  flat bars included, and its final value is initial capital plus net profit;
- cagr/sharpe/max drawdown/win rate/profit factor follow their persisted
  definitions exactly, including the zero-trade and zero-volatility cases;
- an approved research cost model only claims performance when the applied
  economics are mechanically equivalent to it, and every declared cost has to be an
  explicit finite JSON number: null, empty, boolean, non-finite and string costs are
  refused instead of being read as an implied 0.0;
- a malformed boolean funding rate, official mark or modeled mark open HOLDs
  instead of being converted to 0.0/1.0, while the registered decimal-string form
  keeps working;
- the persisted economics boundary describes applied funding, unmodeled isolated
  margin and unapplied slippage separately, so no persisted definition contradicts
  a funding_applied=true run;
- the evidence files and the manifest cover every one of those bars and cycles.

No Qlib provider, no campaign run, no network. The DCA module, the cell module
and the strategy artifacts are loaded by path so the math is testable in any
interpreter. The VZO and Bollinger artifacts are written into a temp dir on
purpose: the repo keeps exactly one committed strategy fixture (Donchian) to
prove a strategy rule survives outside the engine, not a strategy library.
"""

import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DCA_PATH = ROOT / "qlib" / "backtest" / "hummingbot_dca.py"
CELL_PATH = ROOT / "qlib" / "backtest" / "research_cell.py"
DONCHIAN_PATH = ROOT / "tests" / "fixtures" / "strategies" / "donchian_signal.py"


def load_module(name, path):
    """Load one repo module by path (no package import, no heavy dependencies)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cell = load_module("research_cell_under_test", CELL_PATH)
dca = load_module("hummingbot_dca_under_test", DCA_PATH)
ENGINE_BYTES = CELL_PATH.read_bytes()


def check(name, condition, extra=None):
    print(("PASS " if condition else "FAIL ") + name + ("" if condition else " " + str(extra)))
    if not condition:
        raise SystemExit("FAILED: " + name + " " + str(extra))


def artifact(path):
    """Return the (ref, sha256) identity a cell config carries for an artifact."""
    return {"ref": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def spec_for(cells_dir="/results/_validation/formal-cells-round0", artifact_path=None):
    return {
        "candidate": {
            "path": "donchian-20-10-breakout-btcusdt-1h-2026-10-06.md",
            "git_blob_sha": "8a09f6e642261714ef72872b601b8f679e1f0d30",
            "source_commit": "890da11cb50daadc4d59b7fd58a2e96638b17833",
            "repository": "HCH725/alpha-strategy-research",
            "blob_api_ref": "https://api.github.com/repos/HCH725/alpha-strategy-research/git/blobs/"
            "8a09f6e642261714ef72872b601b8f679e1f0d30",
        },
        "runtime": {"commit": "bee09f23d938b96c2c1ef82d891aa0abfa171cd0"},
        "data": {"provider_ref": "/results/_validation/formal-cells-round0/provider"},
        "strategy": {
            "family": "S01509",
            "sides": ["long"],
            "params": {"entry_len": 20, "exit_len": 10},
            "warmup_days": 20,
            "window_start_utc": "2025-04-19 16:00:00",
            "window_end_utc_inclusive": "2026-04-19 16:00:00",
        },
        "strategy_artifact": {"source": str(artifact_path or DONCHIAN_PATH)},
        "house_overlay": {
            "base_order_quote_pct": 0.06,
            "safety_order_quote_pct": 0.06,
            "safety_order_levels": 2,
            "so_spacing_pct": 0.04,
            "take_profit_pct": 0.05,
            "stop_loss_pct": None,
            "trade_cost": 0.0004,
            "initial_capital_quote": 100000.0,
        },
        "research_cost_model": {
            "fees_per_side": 0.0004,
            "slippage": 0.0,
            "funding": "not_modeled",
            "approved": False,
            "note": "round0 research cost model is not approved yet; campaign claimability stays false",
        },
        "materialize": {"cells_dir": cells_dir, "results_root": "/results/_validation"},
    }


def cell_config(spec=None, index=0):
    """Return one materializable cell config for the given campaign spec."""
    spec = spec or spec_for()
    artifact_bytes = Path(spec["strategy_artifact"]["source"]).read_bytes()
    return cell.build_effective_config(
        spec, cell.campaign_identities()[index], hashlib.sha256(artifact_bytes).hexdigest()
    )


def frame(rows):
    """Build the DCA input frame: float epoch second index == timestamp column."""
    import pandas as pd

    stamps = [row[0] for row in rows]
    frame_df = pd.DataFrame(
        {
            "timestamp": [stamp.timestamp() for stamp in stamps],
            "close": [float(row[1]) for row in rows],
            "low": [float(row[1]) * 0.999 for row in rows],
            "high": [float(row[1]) * 1.001 for row in rows],
        }
    )
    frame_df.index = frame_df["timestamp"].to_numpy(dtype=float)
    return frame_df


def stage_path():
    """Price path that triggers the base order, both safety orders, then TP."""
    start = datetime(2025, 5, 1)
    closes = [100.0, 99.0, 96.0, 95.0, 92.0, 91.5, 93.0, 96.0, 101.0, 101.5]
    return frame([(start + timedelta(hours=i), close) for i, close in enumerate(closes)])


def ledger_for(leverage, fixture=None, overlay=None):
    overlay = overlay or spec_for()["house_overlay"]
    prices = cell.stage_prices(100.0, overlay["so_spacing_pct"], overlay["safety_order_levels"])
    amounts = cell.house_dca_amounts(
        overlay["base_order_quote_pct"],
        overlay["safety_order_quote_pct"],
        overlay["safety_order_levels"],
        overlay["initial_capital_quote"],
        leverage,
    )
    ledger, close_type = cell.dca_overlay_ledger(
        fixture if fixture is not None else stage_path(),
        prices,
        amounts,
        Decimal(str(overlay["take_profit_pct"])) if overlay["take_profit_pct"] is not None else None,
        Decimal(str(overlay["stop_loss_pct"])) if overlay["stop_loss_pct"] is not None else None,
        float(overlay["trade_cost"]),
        simulator=dca.simulate_hummingbot_dca,
    )
    return ledger, close_type, prices, amounts


# ------------------------------------------------- strategy artifact fixtures

VZO_ARTIFACT = '''
"""Frozen strategy artifact: VZO band recross (volume oscillator family)."""


def _ema(values, length):
    """Return the exponential moving average of one numeric list."""
    if length <= 0:
        raise ValueError("ema length must be positive")
    alpha = 2.0 / (length + 1.0)
    out = []
    state = None
    for value in values:
        state = value * alpha + state * (1.0 - alpha) if state is not None else value
        out.append(state)
    return out


def signal(*, high, low, close, volume, times, params, window_start, window_end):
    """Long on a recross above the band; flat on a drop below the exit band."""
    if volume is None:
        raise ValueError("vzo artifact requires a frozen-provider volume column")
    length = params.get("vzo_len")
    band = params.get("band")
    exit_band = params.get("exit_band")
    for name, value in (("vzo_len", length), ("band", band), ("exit_band", exit_band)):
        if not isinstance(value, (int, float)):
            raise ValueError("vzo artifact requires a numeric %s" % name)
    signed = [float(volume[0])]
    for i in range(1, len(close)):
        signed.append(float(volume[i]) if close[i] > close[i - 1] else -float(volume[i]))
    numerator = _ema(signed, int(length))
    denominator = _ema([float(value) for value in volume], int(length))
    vzo = [100.0 * n / d if d else 0.0 for n, d in zip(numerator, denominator)]
    state = False
    entry_ts = None
    rows = []
    for i in range(len(close)):
        cross_up = i > 0 and vzo[i] > band >= vzo[i - 1]
        cross_down = i > 0 and vzo[i] < exit_band <= vzo[i - 1]
        in_window = window_start <= times[i] <= window_end
        entry = exit_ = False
        if not state and in_window and cross_up:
            state, entry, entry_ts = True, True, times[i]
        elif state and cross_down:
            state, exit_ = False, True
        rows.append({
            "datetime": times[i],
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "vzo": vzo[i],
            "entry_transition": int(entry),
            "exit_transition": int(exit_),
        })
    return rows, entry_ts
'''


BOLLINGER_ARTIFACT = '''
"""Frozen strategy artifact: Bollinger upper-band breakout (trend family)."""


def _sma(values, length):
    """Return the simple moving average of the last `length` values."""
    return sum(values[-length:]) / float(length)


def _stdev(values, length):
    """Return the population standard deviation of the last `length` values."""
    window = values[-length:]
    mean = sum(window) / float(length)
    return (sum((value - mean) ** 2 for value in window) / float(length)) ** 0.5


def signal(*, high, low, close, volume, times, params, window_start, window_end):
    """Long on a close above the prior upper band; flat below the middle band."""
    length = params.get("bb_len")
    multiplier = params.get("bb_k")
    for name, value in (("bb_len", length), ("bb_k", multiplier)):
        if not isinstance(value, (int, float)):
            raise ValueError("bollinger artifact requires a numeric %s" % name)
    length = int(length)
    state = False
    entry_ts = None
    rows = []
    prior = []
    for i in range(len(close)):
        middle = _sma(prior, length) if len(prior) >= length else None
        spread = _stdev(prior, length) if len(prior) >= length else None
        upper = None if middle is None else middle + multiplier * spread
        in_window = window_start <= times[i] <= window_end
        break_up = upper is not None and close[i] > upper
        break_down = middle is not None and close[i] < middle
        entry = exit_ = False
        if not state and in_window and break_up:
            state, entry, entry_ts = True, True, times[i]
        elif state and break_down:
            state, exit_ = False, True
        rows.append({
            "datetime": times[i],
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "upper_band": upper,
            "middle_band": middle,
            "entry_transition": int(entry),
            "exit_transition": int(exit_),
        })
        prior.append(float(close[i]))
    return rows, entry_ts
'''


def synthetic_series():
    """Return one OHLCV fixture that all three mechanisms read unchanged."""
    start = datetime(2025, 5, 1)
    closes = [100.0 - i * 0.25 for i in range(10)]
    closes += [110.0] + [110.0 + i * 0.75 for i in range(1, 21)]
    volumes = [1000.0 + i * 5.0 for i in range(10)] + [8000.0] + [3000.0 + i * 20.0 for i in range(1, 21)]
    times = [start + timedelta(hours=i) for i in range(len(closes))]
    return {
        "high": [value * 1.001 for value in closes],
        "low": [value * 0.999 for value in closes],
        "close": closes,
        "volume": volumes,
        "times": times,
        "window_start": start - timedelta(days=1),
        "window_end": times[-1],
    }


def write_strategy_dir(directory, name, source):
    """Write one strategy artifact into its own run-scoped cell dir."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return directory, path


def run_artifact(directory, name, params):
    """Load an artifact through the engine loader and run it on the fixture."""
    function = cell.load_signal_artifact(directory, artifact(Path(directory) / name))
    series = synthetic_series()
    rows, entry_ts = function(
        high=series["high"],
        low=series["low"],
        close=series["close"],
        volume=series["volume"],
        times=series["times"],
        params=params,
        window_start=series["window_start"],
        window_end=series["window_end"],
    )
    return series, rows, entry_ts


def dca_from_rows(rows, entry_ts, leverage=3):
    """Run the shared DCA overlay over an artifact's own signal rows."""
    overlay = spec_for()["house_overlay"]
    entry_price = next(row["close"] for row in rows if row["entry_transition"] == 1)
    prices = cell.stage_prices(entry_price, overlay["so_spacing_pct"], overlay["safety_order_levels"])
    amounts = cell.house_dca_amounts(
        overlay["base_order_quote_pct"], overlay["safety_order_quote_pct"],
        overlay["safety_order_levels"], overlay["initial_capital_quote"], leverage,
    )
    frame_rows = [row for row in rows if row["datetime"] >= entry_ts]
    ledger, close_type = cell.dca_overlay_ledger(
        frame([(row["datetime"], row["close"]) for row in frame_rows]),
        prices,
        amounts,
        Decimal(str(overlay["take_profit_pct"])),
        None,
        float(overlay["trade_cost"]),
        simulator=dca.simulate_hummingbot_dca,
    )
    return ledger, close_type, prices, amounts


def test_frozen_identities_are_exactly_42_unique_cells():
    identities = cell.campaign_identities()
    ids = [item["cell_id"] for item in identities]
    check("42 identities", len(identities) == 42, len(identities))
    check("identities are unique", len(set(ids)) == 42)
    expected = {
        (symbol, timeframe, leverage)
        for symbol in cell.CELL_SYMBOLS
        for timeframe in cell.CELL_TIMEFRAMES
        for leverage in cell.CELL_LEVERAGES
    }
    check("identities are the frozen cross product",
          {(i["symbol"], i["timeframe"], i["research_leverage"]) for i in identities} == expected)
    check("identity list is stable across calls",
          [i["cell_id"] for i in cell.campaign_identities()] == ids)
    check("no 43rd cell and no leverage outside 3/5",
          {i["research_leverage"] for i in identities} == {3, 5} and len(ids) == 42)
    check("timeframe maps to a qlib frequency",
          {i["timeframe"] for i in identities} == set(cell.QLIB_FREQ))


def test_cell_config_fails_closed_without_approved_values():
    good = cell_config()
    check("materialized cell config is runnable", cell.problems_in_config(good) == [],
          cell.problems_in_config(good))
    check("an unapproved cost model is declared but not claimable",
          cell.cost_model_claimable(good) is False)
    claimable = cell_config()
    claimable["research_cost_model"]["approved"] = True
    check("an approved cost model is claimable and still runnable",
          cell.cost_model_claimable(claimable) is True and cell.problems_in_config(claimable) == [])
    for label, mutate in (
        ("missing so_spacing", lambda cfg: cfg["house_overlay"].pop("so_spacing_pct")),
        ("missing take_profit declaration", lambda cfg: cfg["house_overlay"].pop("take_profit_pct")),
        ("missing initial capital", lambda cfg: cfg["house_overlay"].pop("initial_capital_quote")),
        ("undeclared cost model approval", lambda cfg: cfg["research_cost_model"].pop("approved")),
        ("leverage outside the frozen set", lambda cfg: cfg.update(research_leverage=4)),
        ("string leverage", lambda cfg: cfg.update(research_leverage="3")),
        ("symbol outside the frozen set", lambda cfg: cfg["strategy"].update(symbol="PEPEUSDT")),
        ("empty symbol", lambda cfg: cfg["strategy"].update(symbol="")),
        ("timeframe outside the frozen set", lambda cfg: cfg["strategy"].update(timeframe="7h")),
        ("missing strategy params object", lambda cfg: cfg["strategy"].update(params=["entry_len"])),
        ("negative warmup", lambda cfg: cfg["strategy"].update(warmup_days=-1)),
        ("missing artifact", lambda cfg: cfg["strategy"].pop("artifact")),
        ("artifact ref is not python", lambda cfg: cfg["strategy"]["artifact"].update(ref="/tmp/x.txt")),
        ("artifact hash is not a digest", lambda cfg: cfg["strategy"]["artifact"].update(sha256="abc")),
        ("missing provider", lambda cfg: cfg["data"].pop("provider_ref")),
        ("missing execution sides", lambda cfg: cfg["strategy"].pop("sides")),
        ("short side the shell cannot execute", lambda cfg: cfg["strategy"].update(sides=["long", "short"])),
        ("missing capability declaration", lambda cfg: cfg.pop("capabilities")),
        ("capability outside the frozen set", lambda cfg: cfg["capabilities"].update(execution=["qlib-native-backtest", "grid-runner"])),
        ("capability sides diverging from the strategy", lambda cfg: cfg["capabilities"].update(sides=["long", "short"])),
    ):
        broken = cell_config()
        mutate(broken)
        check("fail closed: " + label, cell.problems_in_config(broken) != [])


def test_a_two_sided_strategy_holds_before_any_cell_is_written():
    """A frozen two-sided record must not be shipped as a long-only artifact.

    The shell executes one long leg. Declaring the missing side is how the
    engineering flow states the true capability gap, so materialize() refuses
    before writing a single cell instead of producing a smaller strategy under
    the same candidate identity.
    """
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-two-sided-"))
    spec = spec_for()
    spec["strategy"]["sides"] = ["long", "short"]
    spec_path = tmp / "campaign-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    root = tmp / "cells"
    held = None
    try:
        cell.materialize(spec_path, root)
    except SystemExit as exc:
        held = str(exc)
    check("a two-sided strategy holds with the missing execution primitive",
          held is not None and "execution primitive for side: short" in held, held)
    check("no cell is written for the unsupported strategy", not root.exists())
    undeclared = spec_for()
    undeclared["strategy"].pop("sides")
    undeclared_path = tmp / "undeclared.json"
    undeclared_path.write_text(json.dumps(undeclared), encoding="utf-8")
    held = None
    try:
        cell.materialize(undeclared_path, tmp / "cells2")
    except SystemExit as exc:
        held = str(exc)
    check("an artifact that does not declare its sides is refused",
          held is not None and "sides is not declared" in held, held)


def test_materialize_writes_42_bindable_cells():
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-"))
    spec = spec_for()
    spec_path = tmp / "campaign-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    root = tmp / "cells"
    manifest = cell.materialize(spec_path, root)
    entry_bytes = CELL_PATH.read_bytes()
    artifact_bytes = DONCHIAN_PATH.read_bytes()
    artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()
    check("materialized exactly 42 cells", manifest["cell_count"] == 42 == len(manifest["cells"]))
    check("campaign manifest is written", (root / "campaign-manifest.json").is_file())
    good = 0
    for entry in manifest["cells"]:
        cell_dir = root / entry["cell_id"]
        config_path = cell_dir / "effective_config.json"
        entrypoint_path = cell_dir / "run_cell.entrypoint.py"
        artifact_path = cell_dir / "strategy_signal.py"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        same_source = entrypoint_path.read_bytes() == entry_bytes
        same_artifact = artifact_path.is_file() and artifact_path.read_bytes() == artifact_bytes
        bindable = (
            config.get("candidate", {}).get("git_blob_sha") == entry["source_blob_sha"]
            and config.get("runtime", {}).get("commit") == entry["qlib_runtime_commit"]
            and config.get("execution", {}).get("entrypoint") == entry["entrypoint_ref"]
            and entry["effective_config_ref"] == entry["entrypoint_ref"].replace(
                "run_cell.entrypoint.py", "effective_config.json")
            and hashlib.sha256(entrypoint_path.read_bytes()).hexdigest() == entry["entrypoint_sha256"]
            and hashlib.sha256(config_path.read_bytes()).hexdigest() == entry["effective_config_sha256"]
            and cell.problems_in_config(config) == []
            and config["research_leverage"] == entry["research_leverage"]
            and config["strategy"]["artifact"] == {
                "ref": spec["materialize"]["cells_dir"] + "/" + entry["cell_id"] + "/strategy_signal.py",
                "sha256": artifact_sha}
            and entry["strategy_artifact_sha256"] == artifact_sha
            and entry["strategy_artifact_ref"] == config["strategy"]["artifact"]["ref"]
            and config["strategy"]["params"] == spec["strategy"]["params"]
        )
        good += int(same_source and same_artifact and bindable)
    check("every cell is byte-identical, hash-consistent and bindable", good == 42, good)
    out_dirs = {entry["evidence_refs"]["manifest"].rsplit("/", 1)[0] for entry in manifest["cells"]}
    check("every cell gets its own transport run location", len(out_dirs) == 42, len(out_dirs))
    check("evidence refs are the transport run location the readback reports",
          all(entry["evidence_refs"]["manifest"].endswith("/manifest.json")
              and entry["evidence_refs"]["mlflow_readback"]
              == entry["evidence_refs"]["manifest"].replace("manifest.json", "mlflow_readback.json")
              and entry["evidence_refs"]["manifest"].startswith("/results/_validation/n8n-live-")
              for entry in manifest["cells"]))
    check("the run-location formula reproduces the deployed representative evidence dir",
          cell.out_dir_short({
              "source_blob_sha": "8a09f6e642261714ef72872b601b8f679e1f0d30",
              "entrypoint_sha256": "000e23b739eb13a7c3b1077af5501556bd73fd2ec0e9d439ce1e6d5113d14c05",
              "effective_config_sha256": "7a976cce5a6d418c635af073e9df7f553b6893259c79e59722f49b368c28d1d2",
              "qlib_runtime_commit": "bee09f23d938b96c2c1ef82d891aa0abfa171cd0",
          }) == "8b37e4ca2140")
    try:
        cell.materialize(spec_path, root)
        check("materialize refuses overwrite", False)
    except SystemExit as exc:
        check("materialize refuses overwrite", "refuse overwrite" in str(exc), exc)
    missing = spec_for()
    missing["house_overlay"].pop("so_spacing_pct")
    spec_path.write_text(json.dumps(missing), encoding="utf-8")
    try:
        cell.materialize(spec_path, tmp / "cells2")
        check("materialize holds without an approved overlay value", False)
    except SystemExit as exc:
        check("materialize holds without an approved overlay value",
              "no approved value" in str(exc), exc)
    missing_capital = spec_for()
    missing_capital["house_overlay"].pop("initial_capital_quote")
    spec_path.write_text(json.dumps(missing_capital), encoding="utf-8")
    held_before_write = False
    try:
        cell.materialize(spec_path, tmp / "cells3")
    except SystemExit as exc:
        held_before_write = "no approved value" in str(exc) and not (tmp / "cells3").exists()
    check("materialize holds without an initial capital allocation before writing", held_before_write)
    no_artifact = spec_for()
    no_artifact.pop("strategy_artifact")
    spec_path.write_text(json.dumps(no_artifact), encoding="utf-8")
    held_no_artifact = False
    try:
        cell.materialize(spec_path, tmp / "cells4")
    except SystemExit as exc:
        held_no_artifact = "strategy_artifact" in str(exc) and not (tmp / "cells4").exists()
    check("materialize holds before writing when no strategy artifact is declared", held_no_artifact)


def test_capability_declaration_is_one_reusable_primitive_set():
    """Every cell declares the same reusable primitives, never its own identity.

    A per-cell name (``dogeusdt-1d``) would turn the capability gate into a
    42-entry allowlist that cannot fail, and would let one hardcoded symbol
    stand in for the other 41. Coverage is measured provider readiness instead.
    """
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-caps-"))
    spec_path = tmp / "campaign-spec.json"
    spec_path.write_text(json.dumps(spec_for()), encoding="utf-8")
    manifest = cell.materialize(spec_path, tmp / "cells")
    declarations = [entry["capabilities"] for entry in manifest["cells"]]
    check("every cell declares the reusable primitive set",
          all(decl == cell.CAPABILITIES for decl in declarations),
          sorted({json.dumps(d, sort_keys=True) for d in declarations}))
    check("the declaration is identical across all 42 cells",
          len({json.dumps(d, sort_keys=True) for d in declarations}) == 1)
    check("the declaration names no symbol, timeframe or per-cell identity",
          not any(
              token in json.dumps(declarations).lower()
              for token in [symbol.lower() for symbol in cell.CELL_SYMBOLS]
              + list(cell.CELL_TIMEFRAMES)
              + ["btcusdt-1h", "bar-close-1h"]
          ), json.dumps(declarations[0], sort_keys=True))
    check("the declaration buckets are exactly execution/data/timing/sides",
          set(cell.CAPABILITIES) == {"execution", "data", "timing", "sides"}, sorted(cell.CAPABILITIES))
    check("the shell declares only the long side it actually executes",
          cell.CAPABILITIES["sides"] == ["long"], cell.CAPABILITIES["sides"])
    mutated = cell_config(index=41)
    mutated["strategy"]["symbol"] = "DOGEUSDT"
    mutated["strategy"]["timeframe"] = "1d"
    check("a non-BTC/1h cell config still validates on the same primitives",
          cell.problems_in_config(mutated) == [], cell.problems_in_config(mutated))


def test_engine_carries_no_strategy_rule():
    """The shared engine must not name, gate or special-case any strategy."""
    source = ENGINE_BYTES.decode("utf-8").lower()
    for token in ("donchian", "s01509", "vzo", "bollinger", "entry_len", "exit_len",
                  "break_up", "break_down", "channel collision"):
        check("engine does not carry strategy token: " + token, token not in source)
    for token in ('strategy.get("family")', "strategy['family']", 'strategy.get("name")',
                  "strategy['name']", "family_id", 'symbol == "', "timeframe == \""):
        check("engine does not gate on a strategy identity: " + token, token not in source)


def test_three_strategies_share_one_engine():
    """Donchian, VZO and Bollinger run through the same engine bytes."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-strategies-"))
    donchian_dir, donchian = write_strategy_dir(
        tmp / "donchian", "strategy_signal.py", DONCHIAN_PATH.read_text(encoding="utf-8"))
    vzo_dir, vzo = write_strategy_dir(tmp / "vzo", "strategy_signal.py", VZO_ARTIFACT)
    bollinger_dir, bollinger = write_strategy_dir(
        tmp / "bollinger", "strategy_signal.py", BOLLINGER_ARTIFACT)
    cases = (
        ("donchian", donchian_dir, donchian, {"entry_len": 5, "exit_len": 3}),
        ("vzo", vzo_dir, vzo, {"vzo_len": 5, "band": -20.0, "exit_band": -80.0}),
        ("bollinger", bollinger_dir, bollinger, {"bb_len": 5, "bb_k": 2.0}),
    )
    entry_times = []
    for label, directory, path, params in cases:
        series, rows, entry_ts = run_artifact(directory, path.name, params)
        check(label + " artifact satisfies the row contract",
              cell.problems_in_signal_rows(rows) == [], cell.problems_in_signal_rows(rows))
        check(label + " artifact produces exactly one entry transition",
              sum(row["entry_transition"] for row in rows) == 1,
              sum(row["entry_transition"] for row in rows))
        check(label + " artifact reports the entry timestamp", entry_ts is not None)
        entry_row = next(row for row in rows if row["entry_transition"] == 1)
        check(label + " entry is the bar after the breakout", entry_row["datetime"] == series["times"][10],
              entry_row["datetime"])
        ledger, close_type, prices, amounts = dca_from_rows(rows, entry_ts)
        check(label + " entries through the shared DCA overlay",
              len(ledger) > 0 and close_type in ("TAKE_PROFIT", "STOP_LOSS", "FORCE_FLAT", "END"), close_type)
        check(label + " stage prices come from the approved overlay",
              [float(p) for p in prices] == [110.0, 105.6, 101.2], prices)
        entry_times.append(entry_ts)
    check("all three mechanisms enter on the same frozen bar",
          len(set(entry_times)) == 1, entry_times)
    check("the engine bytes are the same for all three strategies",
          CELL_PATH.read_bytes() == ENGINE_BYTES)


def test_strategy_artifact_fails_closed():
    """A missing, foreign, malformed or swapped artifact must never run."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-artifact-"))
    cell_dir, path = write_strategy_dir(tmp, "strategy_signal.py", DONCHIAN_PATH.read_text(encoding="utf-8"))
    good = artifact(path)
    check("the good artifact loads", callable(cell.load_signal_artifact(cell_dir, good)))

    missing = dict(good, ref=str(Path(cell_dir) / "absent.py"))
    check("missing artifact fails closed", raised(lambda: cell.load_signal_artifact(cell_dir, missing), "not readable"))

    outside = Path(tempfile.mkdtemp(prefix="research-cell-foreign-")) / "strategy_signal.py"
    outside.write_text(DONCHIAN_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    foreign = {"ref": str(outside), "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}
    check("out-of-cell-dir artifact fails closed",
          raised(lambda: cell.load_signal_artifact(cell_dir, foreign), "outside the cell directory"))

    swapped = dict(good, sha256="0" * 64)
    check("hash mismatch fails closed",
          raised(lambda: cell.load_signal_artifact(cell_dir, swapped), "hash mismatch"))

    broken_dir, broken_path = write_strategy_dir(tmp / "syntax", "strategy_signal.py", "def signal(:\n")
    broken = {"ref": str(broken_path), "sha256": hashlib.sha256(broken_path.read_bytes()).hexdigest()}
    check("unparsable artifact fails closed",
          raised(lambda: cell.load_signal_artifact(broken_dir, broken), "import failed"))

    empty_dir, empty_path = write_strategy_dir(tmp / "nofn", "strategy_signal.py", "VALUE = 1\n")
    empty = {"ref": str(empty_path), "sha256": hashlib.sha256(empty_path.read_bytes()).hexdigest()}
    check("artifact without signal() fails closed",
          raised(lambda: cell.load_signal_artifact(empty_dir, empty), "no callable signal"))

    check("undeclared artifact fails closed",
          raised(lambda: cell.load_signal_artifact(cell_dir, None), "not declared"))
    check("a bad row contract is rejected",
          cell.problems_in_signal_rows([{"datetime": 1}]) != [])
    check("an empty row set is rejected", cell.problems_in_signal_rows([]) != [])
    check("a backwards timestamp is rejected",
          cell.problems_in_signal_rows([
              {"datetime": datetime(2025, 1, 2), "high": 1.0, "low": 1.0, "close": 1.0,
               "entry_transition": 0, "exit_transition": 0},
              {"datetime": datetime(2025, 1, 1), "high": 1.0, "low": 1.0, "close": 1.0,
               "entry_transition": 0, "exit_transition": 0},
          ]) != [])


def raised(function, token):
    """Return True when the call exits with a HOLD message carrying `token`."""
    try:
        function()
    except SystemExit as exc:
        return token in str(exc)
    return False


def hold_message(function):
    """Return the HOLD message a call raises, or None when the call does not hold."""
    try:
        function()
    except SystemExit as exc:
        return str(exc)
    return None


def test_research_leverage_scales_stage_exposure():
    base = spec_for()["house_overlay"]
    amounts_3 = cell.house_dca_amounts(base["base_order_quote_pct"], base["safety_order_quote_pct"],
                                       base["safety_order_levels"], base["initial_capital_quote"], 3)
    amounts_5 = cell.house_dca_amounts(base["base_order_quote_pct"], base["safety_order_quote_pct"],
                                       base["safety_order_levels"], base["initial_capital_quote"], 5)
    check("house overlay has one base order and two safety orders (3 stages)", len(amounts_3) == 3, amounts_3)
    check("bo/so quote split stays 6% of base capital",
          [float(a) for a in amounts_3] == [18000.0, 18000.0, 18000.0], amounts_3)
    check("every stage scales exactly by 5/3",
          all(b / a == Decimal(5) / Decimal(3) for a, b in zip(amounts_3, amounts_5)))
    prices = cell.stage_prices(100.0, base["so_spacing_pct"], base["safety_order_levels"])
    check("stage prices are approval-driven, not invented",
          [float(p) for p in prices] == [100.0, 96.0, 92.0], prices)


def test_dca_overlay_keeps_bo_and_two_so_semantics():
    ledger, close_type, prices, amounts = ledger_for(3)
    filled = [float(ledger["filled_amount_quote_%d" % level].iloc[-1]) for level in range(3)]
    check("base order fills the full stage allocation", filled[0] == float(amounts[0]), filled)
    check("both safety orders fill their stage allocation",
          filled[1] == float(amounts[1]) and filled[2] == float(amounts[2]), filled)
    check("stage prices are the DCA stage prices",
          [float(p) for p in prices] == [100.0, 96.0, 92.0], prices)
    check("close type comes from the take profit stage", close_type == "TAKE_PROFIT", close_type)
    check("ledger reports the summed exposure",
          float(ledger["filled_amount_quote"].iloc[-1]) >= sum(filled), ledger["filled_amount_quote"].iloc[-1])


def test_3x_vs_5x_exposure_pnl_and_reproducibility():
    ledger_3a, kind_3a, _, _ = ledger_for(3)
    ledger_3b, kind_3b, _, _ = ledger_for(3)
    ledger_5, kind_5, _, _ = ledger_for(5)
    exposure_3 = float(max(ledger_3a["filled_amount_quote"]))
    exposure_5 = float(max(ledger_5["filled_amount_quote"]))
    pnl_3 = float(ledger_3a["net_pnl_quote"].iloc[-1])
    pnl_5 = float(ledger_5["net_pnl_quote"].iloc[-1])
    check("3x and 5x produce different exposure", exposure_3 != exposure_5, (exposure_3, exposure_5))
    check("5x exposure is exactly 5/3 of 3x", abs(exposure_5 / exposure_3 - 5 / 3) < 1e-12,
          exposure_5 / exposure_3)
    check("5x modeled pnl is exactly 5/3 of 3x", abs(pnl_5 / pnl_3 - 5 / 3) < 1e-9, pnl_5 / pnl_3)
    check("close semantics are leverage-invariant", kind_3a == kind_5 == "TAKE_PROFIT", (kind_3a, kind_5))
    check("the same cell replays to the same ledger",
          ledger_3a["net_pnl_quote"].equals(ledger_3b["net_pnl_quote"])
          and kind_3a == kind_3b)
    check("leverage is not modeled as fees or funding",
          float(ledger_3a["cum_fees_quote"].iloc[-1]) > 0.0)



# ------------------------------------------------- full-strategy replay tests


def replay_overlay():
    """Return the house overlay exactly as a materialized cell config carries it.

    A hand-shaped dict is what let a wrong overlay key slip through before, so the
    replay fixtures read the canonical config block instead of restating it.
    """
    return cell_config()["house_overlay"]


def transition_rows(closes, entries=(), exits=(), start=None):
    """Build artifact rows around an explicit close path plus its transitions."""
    start = start or datetime(2025, 5, 1)
    return [
        {
            "datetime": start + timedelta(hours=index),
            "high": float(close) * 1.001,
            "low": float(close) * 0.999,
            "close": float(close),
            "entry_transition": int(index in entries),
            "exit_transition": int(index in exits),
        }
        for index, close in enumerate(closes)
    ]


def replay(rows, first=None, last=None, overlay=None, leverage=3, simulator=None, funding=None,
           timeframe="1h"):
    """Plan, replay and project one synthetic transition series end to end."""
    overlay = overlay or replay_overlay()
    first = 0 if first is None else first
    last = len(rows) - 1 if last is None else last
    cycles, skipped = cell.plan_cycles(rows, first, last)
    records = cell.replay_cycles(rows, cycles, overlay, leverage, timeframe,
                                 simulator=simulator or dca.simulate_hummingbot_dca, funding=funding)
    indices = list(range(first, last + 1))
    projection = cell.equity_projection(indices, records, float(overlay["initial_capital_quote"]))
    total_funding = float(sum(record["funding_paid_quote"] for record in records))
    metrics = cell.metrics_from_equity(
        [rows[index]["datetime"] for index in indices],
        [bar["equity_quote"] for bar in projection],
        [record["realized_pnl_quote"] for record in records],
        cell.PERIODS_PER_YEAR[timeframe],
        float(overlay["initial_capital_quote"]),
        funding_paid_quote=total_funding,
    )
    return {"cycles": cycles, "skipped": skipped, "records": records, "projection": projection,
            "metrics": metrics, "indices": indices, "overlay": overlay, "funding": funding,
            "funding_paid_quote": total_funding}


def two_cycle_series():
    """Two take-profit cycles on two different entry prices, with flat gaps."""
    return replay(transition_rows(
        [100.0, 100.0, 103.0, 106.0, 104.0, 104.0, 200.0, 205.0, 212.0, 210.0, 209.0, 209.0],
        entries={1, 6}, exits={4, 10},
    ))


def test_replay_uses_the_frozen_config_overlay_keys():
    """The replay must read the canonical overlay keys a real cell config carries."""
    overlay = replay_overlay()
    check("the replay fixture is the config-shaped overlay, not a hand-shaped dict",
          set(overlay) == set(cell.OVERLAY_KEYS), sorted(overlay))
    check("the safety-order allocation key is the canonical one",
          "safety_order_quote_pct" in overlay and "safety_order_pct" not in overlay, sorted(overlay))
    run = replay(transition_rows([100.0, 100.0, 103.0, 106.0, 104.0], entries={1}, exits={4}), overlay=overlay)
    record = run["records"][0]
    check("replaying through the canonical overlay keys produces the stage plan",
          record["stage_quote_amounts"] == [18000.0, 18000.0, 18000.0], record["stage_quote_amounts"])
    check("and realizes the cycle without a KeyError",
          abs(record["realized_pnl_quote"] - 1065.6) < 1e-9, record["realized_pnl_quote"])
    wrong = dict(overlay)
    wrong["safety_order_pct"] = wrong.pop("safety_order_quote_pct")
    failed = None
    try:
        replay(transition_rows([100.0, 100.0, 103.0], entries={1}), overlay=wrong)
    except KeyError as exc:
        failed = str(exc)
    check("a wrong overlay key is a hard failure, never a silent default",
          failed is not None and "safety_order_quote_pct" in failed, failed)


def test_cost_model_equivalence_gates_performance_claimable():
    """An approved cost declaration alone must not make metrics claimable."""
    frozen = cell_config()
    matches, reason = cell.applied_cost_matches_model(frozen)
    check("the frozen canonical cost model is not approved", matches is False and "not approved" in reason,
          reason)
    check("the replay carries the frozen Binance USD-M funding identity constants",
          cell.FUNDING_BASIS == "binance-usdm-funding-v1"
          and cell.FUNDING_MODELED_MARK_BASIS == "binance-usdm-mark-1h-open-v1"
          and cell.FUNDING_TRUTH_STATUS == ("official", "modeled_funding"),
          (cell.FUNDING_BASIS, cell.FUNDING_MODELED_MARK_BASIS, cell.FUNDING_TRUTH_STATUS))
    approved = cell_config()
    approved["research_cost_model"]["approved"] = True
    matches, reason = cell.applied_cost_matches_model(approved)
    check("an approved fee-only model still fails on unmodeled funding", matches is False, matches)
    check("the reason names funding as the gap", "funding" in reason.lower(), reason)
    check("approved + fees match + slippage 0 + funding not_modeled is NOT claimable",
          cell.performance_claimable(approved, two_cycle_series()["metrics"]) is False)
    for label, mutate in (
        ("declared slippage is not applied", lambda cost: cost.update(slippage=0.0002)),
        ("declared funding is not applied", lambda cost: cost.update(funding="8h")),
        ("declared fee is not the fee the overlay applies", lambda cost: cost.update(fees_per_side=0.0002)),
        ("funding declared as not modeled", lambda cost: cost.update(funding="not_modeled")),
    ):
        broken = cell_config()
        broken["research_cost_model"]["approved"] = True
        mutate(broken["research_cost_model"])
        check("fail closed: " + label, cell.applied_cost_matches_model(broken)[0] is False,
              cell.applied_cost_matches_model(broken))
    traded = two_cycle_series()["metrics"]
    check("the frozen cost model keeps good evidence unclaimable",
          cell.performance_claimable(frozen, traded) is False)
    check("no trade keeps the claim false",
          cell.performance_claimable(approved, replay(transition_rows([100.0, 101.0]))["metrics"]) is False)
    check("undefined metrics keep the claim false",
          cell.performance_claimable(approved, dict(traded, cagr_pct=None)) is False)
    check("a malformed cost block keeps the claim false",
          cell.performance_claimable({"research_cost_model": "approved", "house_overlay": {}}, traded) is False)


def test_two_complete_cycles_are_both_replayed():
    """Every entry/exit pair in the window is replayed, not just the first."""
    run = two_cycle_series()
    records = run["records"]
    check("two entry/exit pairs produce two cycles", len(records) == 2, records)
    check("no cycle is skipped inside the window", run["skipped"] == 0, run["skipped"])
    first, second = records[0], records[1]
    check("cycle 1 opens on its own entry bar", (first["entry_index"], first["exit_index"]) == (1, 4),
          (first["entry_index"], first["exit_index"]))
    check("cycle 2 opens on the second entry bar", (second["entry_index"], second["exit_index"]) == (6, 10),
          (second["entry_index"], second["exit_index"]))
    check("each cycle prices its stage from its own entry close",
          first["stage_prices"] == [100.0, 96.0, 92.0] and second["stage_prices"] == [200.0, 192.0, 184.0],
          (first["stage_prices"], second["stage_prices"]))
    check("both cycles close on their overlay take profit",
          [record["close_reason"] for record in records] == ["TAKE_PROFIT", "TAKE_PROFIT"],
          [record["close_reason"] for record in records])
    check("both cycles realize their own fee-inclusive PnL",
          all(abs(record["realized_pnl_quote"] - 1065.6) < 1e-9 for record in records),
          [record["realized_pnl_quote"] for record in records])
    check("the aggregate PnL is the sum of the replayed cycles",
          abs(run["metrics"]["net_profit_usdt"] - 2131.2) < 1e-9, run["metrics"]["net_profit_usdt"])
    check("both cycles count as trades", run["metrics"]["total_trades"] == 2, run["metrics"]["total_trades"])
    positions = [bar["position_state"] for bar in run["projection"]]
    check("each cycle is long while the overlay holds, source_long after it closes, flat after the exit",
          positions == ["flat", "long", "long", "long", "source_long", "flat",
                        "long", "long", "long", "source_long", "source_long", "flat"], positions)
    check("the overlay's doubled last-bar allocation is not equity",
          abs(records[0]["max_exposure_quote"] - 2 * 18000.0) < 1e-9, records[0]["max_exposure_quote"])


def test_entry_ts_is_evidence_only_and_every_transition_is_replayed():
    """The committed artifact's entry_ts is its LAST entry; the rows must drive replay."""
    closes = []
    price = 100.0
    for _ in range(6):
        for _ in range(8):
            price += 2.0
            closes.append(round(price, 4))
        for _ in range(8):
            price -= 3.0
            closes.append(round(price, 4))
    start = datetime(2025, 5, 1)
    times = [start + timedelta(hours=index) for index in range(len(closes))]
    series = {
        "high": [value * 1.004 for value in closes],
        "low": [value * 0.996 for value in closes],
        "close": closes,
        "volume": [1000.0 for _ in closes],
        "times": times,
        "window_start": times[10],
        "window_end": times[-1],
    }
    directory, path = write_strategy_dir(
        Path(tempfile.mkdtemp(prefix="research-cell-abi-")), "strategy_signal.py",
        DONCHIAN_PATH.read_text(encoding="utf-8"))
    function = cell.load_signal_artifact(directory, artifact(path))
    rows, entry_ts = function(
        high=series["high"], low=series["low"], close=series["close"], volume=series["volume"],
        times=series["times"], params={"entry_len": 5, "exit_len": 3},
        window_start=series["window_start"], window_end=series["window_end"],
    )
    entries = [row["datetime"] for row in rows if row["entry_transition"] == 1]
    check("the committed fixture produces several entry transitions", len(entries) > 2, entries)
    check("its single entry_ts is the last transition, not the first",
          entry_ts == entries[-1] and entry_ts != entries[0], (entry_ts, entries[0], entries[-1]))
    fast = replay(rows, first=10)
    check("the replay still uses every transition row",
          len(fast["records"]) == len(entries), (len(fast["records"]), len(entries)))
    check("each replayed cycle starts on its own entry bar",
          [record["entry_index"] for record in fast["records"]]
          == [index for index, row in enumerate(rows) if row["entry_transition"] == 1],
          [record["entry_index"] for record in fast["records"]])
    check("the metrics count every cycle, not the single entry_ts",
          fast["metrics"]["total_trades"] == len(entries), fast["metrics"]["total_trades"])


def test_strategy_exit_closes_the_cycle_without_an_overlay_exit():
    """A source exit is a real close boundary even when the overlay never fired."""
    run = replay(transition_rows([100.0, 100.0, 100.5, 101.0], entries={1}, exits={3}))
    record = run["records"][0]
    check("the strategy exit closes at the exit bar",
          (record["exit_index"], record["close_index"]) == (3, 3), (record["exit_index"], record["close_index"]))
    check("the close reason is the strategy exit, not an invented overlay exit",
          record["close_reason"] == "STRATEGY_EXIT", record["close_reason"])
    check("the primitive's own boundary close type is preserved for audit",
          record["dca_close_type"] in ("TIME_LIMIT", "FAILED"), record["dca_close_type"])
    check("the cycle realizes the mark to market at the exit bar",
          abs(record["realized_pnl_quote"] - 165.6) < 1e-9, record["realized_pnl_quote"])
    check("one cycle is reported", run["metrics"]["total_trades"] == 1, run["metrics"]["total_trades"])


def test_overlay_exit_before_the_source_exit_creates_no_phantom_entry():
    """TP before the source exit leaves the account flat but the source long."""
    run = replay(transition_rows([100.0, 100.0, 103.0, 106.0, 104.0, 103.0, 104.0, 104.0, 104.0],
                                 entries={1, 7}, exits={6}))
    records = run["records"]
    check("the early overlay exit and the later source exit are two cycles, not three",
          len(records) == 2, [record["entry_index"] for record in records])
    first, second = records[0], records[1]
    check("cycle 1 closes on take profit before its source exit",
          (first["close_index"], first["exit_index"], first["close_reason"]) == (3, 6, "TAKE_PROFIT"),
          (first["close_index"], first["exit_index"], first["close_reason"]))
    check("the account stays flat while the source is still long",
          [bar["position_state"] for bar in run["projection"][3:7]]
          == ["long", "source_long", "source_long", "source_long"],
          [bar["position_state"] for bar in run["projection"][3:7]])
    check("equity is carried, not re-marked, while the overlay is flat",
          len({round(run["projection"][index]["equity_quote"], 9) for index in (3, 4, 5, 6)}) == 1,
          [run["projection"][index]["equity_quote"] for index in (3, 4, 5, 6)])
    check("no phantom cycle starts inside the still-long source state",
          second["entry_index"] == 7, second["entry_index"])
    check("the cycle still open at window end is forced flat there",
          second["close_reason"] == "FORCE_FLAT" and second["forced"] is True, second["close_reason"])
    check("a forced flat on a single bar pays only the round-trip cost",
          abs(second["realized_pnl_quote"] + 14.4) < 1e-9, second["realized_pnl_quote"])


def test_full_equity_covers_flat_bars_and_reconciles():
    """equity.csv spans the whole window and final equity == initial + net profit."""
    run = two_cycle_series()
    projection, metrics = run["projection"], run["metrics"]
    initial = float(run["overlay"]["initial_capital_quote"])
    check("every evaluation bar carries an equity point", len(projection) == 12, len(projection))
    check("a flat bar before the first cycle is present and unchanged",
          projection[0]["position_state"] == "flat" and projection[0]["equity_quote"] == initial
          and projection[0]["bar_return"] == 0.0, projection[0])
    check("flat bars between cycles carry the realized PnL",
          projection[5]["position_state"] == "flat" and abs(projection[5]["equity_quote"] - 101065.6) < 1e-9,
          projection[5])
    check("flat bars have zero bar return",
          all(projection[index]["bar_return"] == 0.0 for index in (0, 4, 5, 9, 10, 11)),
          [projection[index]["bar_return"] for index in (0, 4, 5, 9, 10, 11)])
    check("the entry bar carries the round-trip cost immediately",
          abs(projection[1]["equity_quote"] - (initial - 14.4)) < 1e-9, projection[1])
    check("final equity equals initial capital plus net profit",
          abs(metrics["end_equity_quote"] - (initial + metrics["net_profit_usdt"])) < 1e-9,
          (metrics["end_equity_quote"], initial + metrics["net_profit_usdt"]))
    check("the equity series ends on the last evaluation bar",
          abs(metrics["end_equity_quote"] - 102131.2) < 1e-9, metrics["end_equity_quote"])
    check("the fee model is documented as the frozen primitive's round-trip estimate",
          "2 * house_overlay.trade_cost" in metrics["definitions"]["fee_model"],
          metrics["definitions"]["fee_model"])


def test_first_evaluation_bar_entry_cost_is_in_metrics_baseline():
    """First-bar fees must reduce return/CAGR and count in drawdown from frozen capital."""
    run = replay(
        transition_rows([100.0, 100.0, 100.0], entries={0}, exits={2}),
        first=0,
        last=2,
    )
    metrics = run["metrics"]
    projection = run["projection"]
    initial = float(run["overlay"]["initial_capital_quote"])
    fee_loss = run["records"][0]["realized_pnl_quote"]
    check("the synthetic cycle loses only the frozen primitive round-trip fee",
          abs(fee_loss + 14.4) < 1e-9, fee_loss)
    check("the first observed equity already includes the first-bar fee",
          abs(projection[0]["equity_quote"] - (initial - 14.4)) < 1e-9, projection[0])
    check("the first bar return is measured from frozen initial capital",
          abs(projection[0]["bar_return"] - (-14.4 / initial)) < 1e-12, projection[0]["bar_return"])
    check("reported start remains frozen initial capital, not post-fee first observed equity",
          metrics["start_equity_quote"] == initial
          and metrics["first_observed_equity_quote"] == projection[0]["equity_quote"],
          (metrics["start_equity_quote"], metrics["first_observed_equity_quote"]))
    check("net return includes the first-bar fee",
          abs(metrics["net_return_pct"] - (fee_loss / initial * 100.0)) < 1e-12,
          metrics["net_return_pct"])
    check("max drawdown includes the first-bar fee from the initial-capital peak",
          abs(metrics["max_drawdown_pct"] - (-fee_loss / initial * 100.0)) < 1e-12,
          metrics["max_drawdown_pct"])
    check("CAGR is negative rather than falsely zero", metrics["cagr_pct"] is not None and metrics["cagr_pct"] < 0,
          metrics["cagr_pct"])
    check("Sharpe includes the negative first bar rather than becoming undefined",
          metrics["sharpe"] is not None and metrics["sharpe"] < 0, metrics["sharpe"])
    check("final equity reconciles with frozen initial capital plus realized PnL",
          abs(metrics["end_equity_quote"] - (initial + metrics["net_profit_usdt"])) < 1e-9,
          (metrics["end_equity_quote"], initial + metrics["net_profit_usdt"]))


def test_cagr_sharpe_drawdown_win_rate_and_profit_factor_formulas():
    """The reported metrics match their persisted definitions, exactly."""
    import statistics

    equity = [100.0, 120.0, 90.0, 100.0]
    stamps = [datetime(2025, 1, 1) + timedelta(days=index) for index in range(4)]
    realized = [10.0, -5.0, 20.0]
    metrics = cell.metrics_from_equity(stamps, equity, realized, cell.PERIODS_PER_YEAR["1h"], 100.0)
    returns = [0.0, 0.2, -0.25, 100.0 / 90.0 - 1.0]
    expected_sharpe = statistics.mean(returns) / statistics.stdev(returns) * (8760 ** 0.5)
    check("net return is measured on fee-inclusive equity", metrics["net_return_pct"] == 0.0,
          metrics["net_return_pct"])
    check("sharpe is mean/stdev(ddof=1) annualized by the frozen periods per year",
          abs(metrics["sharpe"] - expected_sharpe) < 1e-12, (metrics["sharpe"], expected_sharpe))
    check("the persisted sharpe definition states ddof and the zero-variance rule",
          "ddof=1" in metrics["definitions"]["sharpe"] and "stdev is zero" in metrics["definitions"]["sharpe"],
          metrics["definitions"]["sharpe"])
    check("max drawdown is peak to trough on equity", abs(metrics["max_drawdown_pct"] - 25.0) < 1e-12,
          metrics["max_drawdown_pct"])
    check("the max drawdown definition is persisted as its own key",
          metrics["max_drawdown_definition"] == metrics["definitions"]["max_drawdown_pct"])
    check("win rate counts only positive completed cycles",
          abs(metrics["win_rate_pct"] - (200.0 / 3.0)) < 1e-12, metrics["win_rate_pct"])
    check("profit factor is gross profit over absolute gross loss", metrics["profit_factor"] == 6.0,
          metrics["profit_factor"])
    check("trade count is the number of replayed cycles", metrics["total_trades"] == 3,
          metrics["total_trades"])
    check("net profit is the sum of realized cycle PnL", metrics["net_profit_usdt"] == 25.0,
          metrics["net_profit_usdt"])
    annual = [datetime(2025, 1, 1), datetime(2026, 1, 1)]
    one_year = cell.metrics_from_equity(annual, [100.0, 110.0], [10.0], cell.PERIODS_PER_YEAR["1d"], 100.0)
    check("cagr is geometric over the actual elapsed year",
          abs(one_year["cagr_pct"] - 10.0) < 1e-9, one_year["cagr_pct"])
    half = cell.metrics_from_equity([annual[0], datetime(2025, 7, 2)], [100.0, 110.0], [10.0],
                                    cell.PERIODS_PER_YEAR["1d"], 100.0)
    check("cagr compounds the actual elapsed time instead of reusing the return",
          half["cagr_pct"] > 19.0 and abs(half["net_return_pct"] - 10.0) < 1e-9,
          (half["cagr_pct"], half["net_return_pct"]))
    no_loss = cell.metrics_from_equity(stamps, equity, [5.0, 3.0], cell.PERIODS_PER_YEAR["1h"], 100.0)
    check("profit factor is null when no cycle lost money", no_loss["profit_factor"] is None,
          no_loss["profit_factor"])
    empty = cell.metrics_from_equity(stamps, equity, [], cell.PERIODS_PER_YEAR["4h"], 100.0)
    check("profit factor and win rate are null with no cycle",
          empty["profit_factor"] is None and empty["win_rate_pct"] is None
          and empty["total_trades"] == 0, (empty["profit_factor"], empty["win_rate_pct"]))
    four_hour = cell.metrics_from_equity(stamps, equity, realized, cell.PERIODS_PER_YEAR["4h"], 100.0)
    check("annualization follows the frozen timeframe periods per year",
          abs(four_hour["sharpe"] / metrics["sharpe"] - (2190 / 8760) ** 0.5) < 1e-12,
          (four_hour["sharpe"], metrics["sharpe"]))


def test_zero_trade_and_zero_volatility_evidence():
    """A window with no cycle is complete evidence, never fabricated performance."""
    run = replay(transition_rows([100.0, 101.0, 99.0, 100.0]))
    metrics = run["metrics"]
    check("a flat window replays no cycle", run["records"] == [] and metrics["total_trades"] == 0,
          metrics["total_trades"])
    check("every bar still carries the initial equity",
          {bar["equity_quote"] for bar in run["projection"]} == {100000.0}
          and {bar["position_state"] for bar in run["projection"]} == {"flat"},
          {bar["equity_quote"] for bar in run["projection"]})
    check("zero-trade metrics use the explicit deterministic representation",
          (metrics["net_return_pct"], metrics["cagr_pct"], metrics["net_profit_usdt"],
           metrics["max_drawdown_pct"]) == (0.0, 0.0, 0.0, 0.0),
          (metrics["net_return_pct"], metrics["cagr_pct"], metrics["net_profit_usdt"]))
    check("undefined metrics stay null instead of being invented",
          metrics["sharpe"] is None and metrics["profit_factor"] is None
          and metrics["win_rate_pct"] is None,
          (metrics["sharpe"], metrics["profit_factor"], metrics["win_rate_pct"]))
    approved = cell_config()
    approved["research_cost_model"]["approved"] = True
    check("an approved cost model cannot claim performance without a trade",
          cell.performance_claimable(approved, metrics) is False)
    flat = cell.metrics_from_equity([datetime(2025, 1, 1), datetime(2025, 1, 2)], [100.0, 100.0], [],
                                    cell.PERIODS_PER_YEAR["1h"], 100.0)
    check("zero volatility is null sharpe, not a division", flat["sharpe"] is None, flat["sharpe"])


def test_malformed_transition_sequences_fail_closed():
    """Impossible transition sequences hold instead of being guessed."""
    rows = transition_rows([100.0, 100.0, 101.0, 102.0, 103.0, 104.0])
    orphan_exit = [dict(row) for row in rows]
    orphan_exit[2]["exit_transition"] = 1
    check("an exit without an open cycle fails closed",
          raised(lambda: cell.plan_cycles(orphan_exit, 0, 5), "without an open source cycle"))
    doubled_entry = [dict(row) for row in rows]
    doubled_entry[1]["entry_transition"] = 1
    doubled_entry[3]["entry_transition"] = 1
    check("an entry while the source is still long fails closed",
          raised(lambda: cell.plan_cycles(doubled_entry, 0, 5), "while the source is still long"))
    collision = [dict(row) for row in rows]
    collision[1]["entry_transition"] = 1
    collision[1]["exit_transition"] = 1
    check("a bar carrying both transitions fails closed",
          raised(lambda: cell.plan_cycles(collision, 0, 5), "both an entry and an exit"))
    late_window = [dict(row) for row in rows]
    late_window[1]["entry_transition"] = 1
    late_window[4]["exit_transition"] = 1
    check("a source long on the first evaluation bar fails closed",
          raised(lambda: cell.plan_cycles(late_window, 3, 5), "long on the first evaluation bar"))
    before_window = [dict(row) for row in rows]
    before_window[0]["entry_transition"] = 1
    before_window[1]["exit_transition"] = 1
    cycles, skipped = cell.plan_cycles(before_window, 2, 5)
    check("a cycle that completed before the window is skipped, not replayed",
          cycles == [] and skipped == 1, (cycles, skipped))
    open_cycle = [dict(row) for row in rows]
    open_cycle[2]["entry_transition"] = 1
    forced, skipped_open = cell.plan_cycles(open_cycle, 0, 5)
    check("a cycle still open at the last bar is forced flat there",
          len(forced) == 1 and forced[0]["forced"] is True and forced[0]["exit_index"] == 5, forced)
    check("a forced cycle is not counted as a pre-window cycle", skipped_open == 0, skipped_open)
    check("the fail-closed message is a HOLD, not a silent default",
          raised(lambda: cell.plan_cycles(orphan_exit, 0, 5), "HOLD:"))


def test_evidence_files_and_manifest_cover_the_whole_window():
    """write_evidence emits every required artifact and the manifest hashes them."""
    import csv

    tmp = Path(tempfile.mkdtemp(prefix="research-cell-evidence-"))
    run = two_cycle_series()
    rows = transition_rows(
        [100.0, 100.0, 103.0, 106.0, 104.0, 104.0, 200.0, 205.0, 212.0, 210.0, 209.0, 209.0],
        entries={1, 6}, exits={4, 10},
    )
    evaluation_times = [rows[index]["datetime"] for index in run["indices"]]
    names = cell.write_evidence(tmp, {"symbol": "BTCUSDT", "cycles_replayed": 2}, rows, run["records"],
                                evaluation_times, run["projection"], run["metrics"])
    manifest = cell.write_manifest(tmp)
    required = ["summary.json", "signals.csv", "trades.csv", "events.csv", "funding_events.csv",
                "equity.csv", "metrics.json"]
    check("every required evidence file is written",
          all((tmp / name).is_file() for name in required), sorted(p.name for p in tmp.iterdir()))
    check("write_evidence names exactly the non-MLflow evidence it wrote",
          names == ["summary.json", "signals.csv", "trades.csv", "events.csv", "funding_events.csv",
                    "equity.csv", "metrics.json"],
          names)
    with (tmp / "equity.csv").open(newline="", encoding="utf-8") as handle:
        equity_rows = list(csv.DictReader(handle))
    check("equity.csv spans every evaluation bar", len(equity_rows) == 12, len(equity_rows))
    check("equity.csv carries the minimum contract columns",
          {"datetime", "equity_quote", "bar_return", "position_state", "cycle_number"}
          <= set(equity_rows[0]), sorted(equity_rows[0]))
    check("equity.csv final row equals initial plus net profit",
          abs(float(equity_rows[-1]["equity_quote"]) - 102131.2) < 1e-9, equity_rows[-1]["equity_quote"])
    with (tmp / "trades.csv").open(newline="", encoding="utf-8") as handle:
        trades = list(csv.DictReader(handle))
    check("trades.csv has one row per replayed cycle", len(trades) == 2, trades)
    check("trades.csv carries entry/exit/close, close reason and stage fill facts",
          {"entry_ts_utc", "exit_ts_utc", "close_ts_utc", "close_reason", "realized_pnl_quote",
           "stage_0_filled_quote", "stage_2_filled_quote"} <= set(trades[0]), sorted(trades[0]))
    check("an unfilled safety order is recorded as zero, not invented",
          float(trades[0]["stage_2_filled_quote"]) == 0.0, trades[0]["stage_2_filled_quote"])
    with (tmp / "events.csv").open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    check("events.csv keeps one row per replayed stage, as the pinned transport counts",
          len(events) == 2 * 3, len(events))
    check("events.csv keeps the legacy stage columns and adds the cycle number",
          list(events[0]) == ["cycle", "stage", "stage_price", "stage_quote_amount", "filled_quote"],
          list(events[0]))
    with (tmp / "signals.csv").open(newline="", encoding="utf-8") as handle:
        signal_rows = list(csv.DictReader(handle))
    check("signals.csv holds the artifact's window rows", len(signal_rows) == 12, len(signal_rows))
    metrics_file = json.loads((tmp / "metrics.json").read_text(encoding="utf-8"))
    required_metrics = ["net_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct", "max_drawdown_definition",
                        "win_rate_pct", "total_trades", "profit_factor", "net_profit_usdt"]
    check("metrics.json exposes every required metric",
          all(key in metrics_file for key in required_metrics),
          [key for key in required_metrics if key not in metrics_file])
    check("the manifest hashes every non-MLflow evidence file",
          sorted(manifest) == sorted(names), (sorted(manifest), sorted(names)))
    check("the manifest does not hash itself",
          "manifest.json" not in manifest and "mlruns" not in json.dumps(manifest))
    check("every manifest digest matches the file on disk",
          all(hashlib.sha256((tmp / name).read_bytes()).hexdigest() == entry["sha256"]
              for name, entry in manifest.items()))


def test_zero_cycle_events_file_keeps_the_transport_shape():
    """A zero-cycle cell writes the events header and invents no stage row."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-zero-events-"))
    rows = transition_rows([100.0, 101.0, 100.0])
    run = replay(rows)
    cell.write_evidence(tmp, {"symbol": "BTCUSDT"}, rows, run["records"],
                        [row["datetime"] for row in rows], run["projection"], run["metrics"])
    lines = (tmp / "events.csv").read_text(encoding="utf-8").splitlines()
    check("a zero-cycle run writes the events header but no invented stage row", len(lines) == 1, lines)
    check("so the pinned transport's execution count cannot be satisfied by a no-trade cell",
          len(lines) < 2 and run["records"] == [], (len(lines), run["records"]))


# ------------------------------------------------------- funding model fixtures
#
# The funding source is registered raw data, so the fixtures use the registered
# layout (``<root>/funding/<SYMBOL>/...`` beside ``<root>/mark/<SYMBOL>/1h/...``)
# instead of a hand-built dict: the identity hash and the modeled-mark
# reconstruction are then exercised through the same code path a real run uses.

FUNDING_SYMBOL = "BTCUSDT"
FUNDING_START = datetime(2025, 5, 1)


def funding_boundary(index, interval_hours=1, start=FUNDING_START):
    """Return one synthetic bar's UTC-hour CLOSE boundary, computed independently."""
    moment = (start + timedelta(hours=index + interval_hours)).replace(tzinfo=timezone.utc)
    return int(moment.timestamp() * 1000)


def write_funding_fixture(root, rows, mark_bars=(), symbol=FUNDING_SYMBOL):
    """Write the registered funding + 1h mark layout and return its source identity.

    ``mark_bars`` carries ``(open_time_ms, open[, close])``. Where a test must prove
    the reconstruction reads the bar's OPEN, it passes a deliberately different
    close.
    """
    root = Path(root)
    funding_dir = root / "funding" / symbol
    funding_dir.mkdir(parents=True, exist_ok=True)
    source = funding_dir / ("%s-funding.jsonl.gz" % symbol)
    payloads = [
        json.dumps({
            "symbol": row.get("symbol", symbol),
            "venue": "BINANCE",
            "market_type": "usdm_perp",
            "funding_time_ms": row["funding_time_ms"],
            "funding_rate": row.get("funding_rate", "0.0001"),
            "mark_price": row.get("mark_price"),
            "rate_type": row.get("rate_type", "Regular"),
            "truth_status": row.get("truth_status", "official"),
            "funding_price_source": row.get("funding_price_source"),
        })
        for row in rows
    ]
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(payloads) + "\n")
    mark_dir = root / "mark" / symbol / "1h"
    mark_dir.mkdir(parents=True, exist_ok=True)
    per_month = {}
    for bar in mark_bars:
        stamp_ms, open_price = int(bar[0]), bar[1]
        close_price = bar[2] if len(bar) > 2 and bar[2] is not None else open_price
        month = datetime.fromtimestamp(stamp_ms / 1000.0, tz=timezone.utc)
        per_month.setdefault((month.year, month.month), []).append((stamp_ms, open_price, close_price))
    used = {}
    for (year, month), bars in sorted(per_month.items()):
        name = "%s-1h-%04d-%02d.jsonl.gz" % (symbol, year, month)
        path = mark_dir / name
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("\n".join(
                json.dumps({"open_time_ms": stamp, "open": str(open_price), "close": str(close_price)})
                for stamp, open_price, close_price in bars) + "\n")
        used["%s/1h/%s" % (symbol, name)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "basis": cell.FUNDING_BASIS,
        "source_ref": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "truth_status": list(cell.FUNDING_TRUTH_STATUS),
        "modeled_mark_basis": cell.FUNDING_MODELED_MARK_BASIS,
        "modeled_mark_snapshot_sha256": cell.modeled_mark_snapshot(used),
    }


def modeled_mark_month(root, bars, symbol=FUNDING_SYMBOL):
    """Write one or more months of 1h mark bars verbatim; return the snapshot digest map.

    ``bars`` carries ``(open_time_ms, open[, close])`` where a value may be any
    JSON value at all, so a malformed boolean or non-finite mark reaches the
    loader exactly as the registered source would carry it.
    """
    per_month = {}
    for bar in bars:
        stamp_ms, open_price = int(bar[0]), bar[1]
        close_price = bar[2] if len(bar) > 2 else open_price
        month = datetime.fromtimestamp(stamp_ms / 1000.0, tz=timezone.utc)
        per_month.setdefault((month.year, month.month), []).append((stamp_ms, open_price, close_price))
    used = {}
    mark_dir = Path(root) / "mark" / symbol / "1h"
    mark_dir.mkdir(parents=True, exist_ok=True)
    for (year, month), items in sorted(per_month.items()):
        name = "%s-1h-%04d-%02d.jsonl.gz" % (symbol, year, month)
        path = mark_dir / name
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("\n".join(
                json.dumps({"open_time_ms": stamp, "open": open_price, "close": close_price})
                for stamp, open_price, close_price in items) + "\n")
        used["%s/1h/%s" % (symbol, name)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return used


def official_event(boundary_ms, rate="0.0001", mark="100.0", **extra):
    """Return one official funding row on the frozen envelope."""
    row = {
        "funding_time_ms": boundary_ms,
        "funding_rate": rate,
        "mark_price": mark,
        "truth_status": "official",
        "funding_price_source": "binance_funding_history_mark_price",
    }
    row.update(extra)
    return row


def modeled_event(boundary_ms, rate="0.0001", **extra):
    """Return one modeled_funding row: a registered rate and no official mark."""
    row = {"funding_time_ms": boundary_ms, "funding_rate": rate, "mark_price": None,
           "truth_status": "modeled_funding", "funding_price_source": None}
    row.update(extra)
    return row


def malformed_event(boundary_ms, field, value):
    """Return one official row whose named financial field carries a malformed JSON value.

    Built this way on purpose: the field has to reach the loader exactly as a bad
    source row would, including a JSON boolean, so the fixture cannot lean on a
    helper's string formatting.
    """
    row = official_event(boundary_ms)
    row[field] = value
    return row


def load_fixture_funding(identity, rows, symbol=FUNDING_SYMBOL, first=0, last=None):
    """Load one fixture series over exactly the bar-close window a replay would charge."""
    last = (len(rows) - 1) if last is None else last
    return cell.load_funding_series(identity, symbol, funding_boundary(first), funding_boundary(last))


def flat_cycle_rows(count=8, entries=(1,), exits=None):
    """A window whose prices never move, so only the base DCA stage ever fills."""
    exits = (count - 2,) if exits is None else exits
    return transition_rows([100.0] * count, entries=set(entries), exits=set(exits))


def funding_applied_for(run, initial_capital):
    """Return the applied-economics object a run reports for its own funding charges."""
    return {
        "funding_applied": run["funding"] is not None,
        "identity": None if run["funding"] is None else run["funding"]["identity"],
        "event_count": int(sum(len(record["funding_events"]) for record in run["records"])),
        "total_payment_quote": run["funding_paid_quote"],
        "reconciled": cell.funding_reconciles(run["records"], run["metrics"], initial_capital,
                                              run["funding_paid_quote"]),
    }


def test_a_b_positive_and_negative_funding_move_pnl_exactly():
    """A paid funding event subtracts exactly; a received one adds exactly."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-ab-"))
    paid_identity = write_funding_fixture(tmp / "paid", [official_event(funding_boundary(2), "0.0001")])
    rows = flat_cycle_rows()
    paid = replay(rows, funding=load_fixture_funding(paid_identity, rows))
    record = paid["records"][0]
    initial = float(paid["overlay"]["initial_capital_quote"])
    check("the fixture series carries exactly one official event",
          len(paid["funding"]["events"]) == 1, paid["funding"]["events"])
    check("only the base DCA stage is held on a flat window",
          [stage["filled_quote"] for stage in record["stages"]] == [18000.0, 0.0, 0.0],
          record["stages"])
    check("held base quantity is stage_quote / stage_price",
          abs(record["funding_events"][0]["held_base_qty"] - 180.0) < 1e-12,
          record["funding_events"][0]["held_base_qty"])
    check("a positive funding rate is paid out of PnL",
          abs(record["funding_events"][0]["payment_quote"] - 1.8) < 1e-12,
          record["funding_events"][0]["payment_quote"])
    check("the DCA primitive's own result is preserved for audit",
          abs(record["dca_net_pnl_quote"] + 14.4) < 1e-9, record["dca_net_pnl_quote"])
    check("cycle realized PnL is the DCA result minus the funding it paid",
          abs(record["realized_pnl_quote"] - (record["dca_net_pnl_quote"] - 1.8)) < 1e-12,
          (record["realized_pnl_quote"], record["dca_net_pnl_quote"]))
    check("the payment lands on the bar whose close reaches the event boundary",
          record["funding_events"][0]["charge_index"] == 2, record["funding_events"][0]["charge_index"])
    check("equity is untouched before the charge bar",
          abs(paid["projection"][1]["equity_quote"] - (initial - 14.4)) < 1e-9, paid["projection"][1])
    check("equity on the charge bar drops by fee plus funding exactly",
          abs(paid["projection"][2]["equity_quote"] - (initial - 14.4 - 1.8)) < 1e-9,
          paid["projection"][2])
    check("net profit is the funded PnL", abs(paid["metrics"]["net_profit_usdt"] + 16.2) < 1e-9,
          paid["metrics"]["net_profit_usdt"])
    check("end equity is initial capital plus funded net profit",
          abs(paid["metrics"]["end_equity_quote"] - (initial - 16.2)) < 1e-9,
          paid["metrics"]["end_equity_quote"])
    check("metrics record the funding total",
          abs(paid["metrics"]["funding_paid_quote"] - 1.8) < 1e-12, paid["metrics"]["funding_paid_quote"])
    check("the funding reconciles against cycle PnL and equity",
          cell.funding_reconciles(paid["records"], paid["metrics"], initial, paid["funding_paid_quote"]))

    received_identity = write_funding_fixture(tmp / "received",
                                              [official_event(funding_boundary(2), "-0.0001")])
    received = replay(rows, funding=load_fixture_funding(received_identity, rows))
    credit = received["records"][0]
    check("a negative funding rate is received, exactly",
          abs(credit["funding_events"][0]["payment_quote"] + 1.8) < 1e-12,
          credit["funding_events"][0]["payment_quote"])
    check("a received amount raises cycle PnL by exactly that amount",
          abs(credit["realized_pnl_quote"] - (credit["dca_net_pnl_quote"] + 1.8)) < 1e-12,
          (credit["realized_pnl_quote"], credit["dca_net_pnl_quote"]))
    check("a received amount raises equity on the charge bar",
          abs(received["projection"][2]["equity_quote"] - (initial - 14.4 + 1.8)) < 1e-9,
          received["projection"][2])
    check("a received amount raises net profit",
          abs(received["metrics"]["net_profit_usdt"] + 12.6) < 1e-9,
          received["metrics"]["net_profit_usdt"])
    check("paid and received differ by exactly twice the payment",
          abs((paid["metrics"]["net_profit_usdt"] - received["metrics"]["net_profit_usdt"]) + 3.6) < 1e-9,
          (paid["metrics"]["net_profit_usdt"], received["metrics"]["net_profit_usdt"]))


def test_c_d_same_boundary_entry_does_not_pay_and_same_boundary_exit_does():
    """The two pinned boundary cases, both stated at bar CLOSEs."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-cd-"))
    rows = flat_cycle_rows()
    entry_identity = write_funding_fixture(tmp / "entry", [official_event(funding_boundary(1))])
    entry_run = replay(rows, funding=load_fixture_funding(entry_identity, rows))
    record = entry_run["records"][0]
    check("the base stage fills at the bar close that equals the event boundary",
          record["stage_prices"][0] == 100.0 and record["entry_index"] == 1, record["entry_index"])
    check("a stage filled at the event's own boundary pays nothing",
          record["funding_events"] == [] and record["funding_paid_quote"] == 0.0,
          record["funding_events"])
    check("the same-boundary entry leaves the cycle on its DCA-only PnL",
          abs(record["realized_pnl_quote"] - record["dca_net_pnl_quote"]) < 1e-12,
          (record["realized_pnl_quote"], record["dca_net_pnl_quote"]))
    check("the unfilled event is absent from the evidence rather than recorded as zero",
          entry_run["metrics"]["funding_paid_quote"] == 0.0, entry_run["metrics"]["funding_paid_quote"])

    exit_identity = write_funding_fixture(tmp / "exit", [official_event(funding_boundary(6))])
    exit_run = replay(rows, funding=load_fixture_funding(exit_identity, rows))
    closed = exit_run["records"][0]
    check("the cycle closes on the bar whose close equals the event boundary",
          closed["close_index"] == 6, closed["close_index"])
    check("a position closing at the event's own boundary does pay",
          len(closed["funding_events"]) == 1, closed["funding_events"])
    check("the exit-boundary payment is charged on the closing bar",
          closed["funding_events"][0]["charge_index"] == 6,
          closed["funding_events"][0]["charge_index"])
    check("the exit-boundary payment is the same held * mark * rate",
          abs(closed["funding_events"][0]["payment_quote"] - 1.8) < 1e-12,
          closed["funding_events"][0]["payment_quote"])


def test_e_a_second_stage_fill_only_affects_later_funding_events():
    """Held quantity steps up at the safety order's own fill boundary, never before."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-e-"))
    rows = transition_rows([100.0, 100.0, 95.0, 95.0, 95.0, 95.0, 95.0], entries={1}, exits={6})
    overlay = dict(replay_overlay(), take_profit_pct=None)
    identity = write_funding_fixture(
        tmp, [official_event(funding_boundary(2)), official_event(funding_boundary(3))])
    run = replay(rows, funding=load_fixture_funding(identity, rows), overlay=overlay)
    record = run["records"][0]
    fills = [stage["filled_quote"] for stage in record["stages"]]
    check("the base and the first safety order both fill", fills[0] > 0 and fills[1] > 0, fills)
    check("the second safety order never fills", fills[2] == 0.0, fills)
    charges = record["funding_events"]
    check("both funding events are charged", len(charges) == 2, charges)
    check("the event at the safety order's own fill boundary holds only the base stage",
          abs(charges[0]["held_base_qty"] - 180.0) < 1e-12, charges[0]["held_base_qty"])
    check("the next event holds base plus the filled safety order",
          abs(charges[1]["held_base_qty"] - (180.0 + 18000.0 / 96.0)) < 1e-12, charges[1]["held_base_qty"])
    check("so the second payment is larger than the first",
          charges[1]["payment_quote"] > charges[0]["payment_quote"],
          (charges[0]["payment_quote"], charges[1]["payment_quote"]))
    check("the payments sum to the cycle's funding total",
          abs(record["funding_paid_quote"] - sum(charge["payment_quote"] for charge in charges)) < 1e-12,
          record["funding_paid_quote"])


def test_f_overlay_early_exit_stops_funding_while_the_source_stays_long():
    """The account is flat after the overlay's own exit, so nothing more is charged."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-f-"))
    rows = transition_rows([100.0, 100.0, 103.0, 106.0, 104.0, 103.0, 104.0, 104.0, 104.0],
                           entries={1, 7}, exits={6})
    identity = write_funding_fixture(
        tmp, [official_event(funding_boundary(4)), official_event(funding_boundary(8))])
    run = replay(rows, funding=load_fixture_funding(identity, rows))
    first, second = run["records"][0], run["records"][1]
    check("the overlay closes cycle 1 before its source exit",
          (first["close_index"], first["exit_index"]) == (3, 6),
          (first["close_index"], first["exit_index"]))
    check("an event between the overlay exit and the source exit is not charged to the flat cycle",
          first["funding_events"] == [], first["funding_events"])
    check("the account stays flat and source_long across that span",
          [bar["position_state"] for bar in run["projection"][3:7]] == ["long", "source_long", "source_long", "source_long"],
          [bar["position_state"] for bar in run["projection"][3:7]])
    check("equity is carried, never re-marked, while the account is flat",
          len({round(run["projection"][index]["equity_quote"], 9) for index in (3, 4, 5, 6)}) == 1,
          [run["projection"][index]["equity_quote"] for index in (3, 4, 5, 6)])
    check("the only charged event belongs to the cycle that is actually holding",
          len(second["funding_events"]) == 1 and first["funding_paid_quote"] == 0.0,
          (first["funding_paid_quote"], second["funding_paid_quote"]))
    check("no funding is charged while the source state is source_long",
          all(charge["charge_index"] <= second["close_index"] for charge in second["funding_events"]),
          second["funding_events"])


def test_g_modeled_funding_uses_the_1h_mark_open_and_records_provenance():
    """A modeled row is charged from the 1h mark OPEN, and says so in the evidence."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-g-"))
    boundary = funding_boundary(2)
    identity = write_funding_fixture(
        tmp, [modeled_event(boundary)], mark_bars=[(boundary, "123.45", "999.0")])
    rows = flat_cycle_rows()
    funding = load_fixture_funding(identity, rows)
    run = replay(rows, funding=funding)
    charge = run["records"][0]["funding_events"][0]
    check("the modeled row keeps its registered rate",
          abs(charge["funding_rate"] - 0.0001) < 1e-15, charge["funding_rate"])
    check("the modeled mark is the 1h mark bar OPEN, not its close",
          abs(charge["mark_price"] - 123.45) < 1e-12, charge["mark_price"])
    check("the modeled mark price source is the reconstruction basis",
          charge["mark_price_source"] == cell.FUNDING_MODELED_MARK_BASIS, charge["mark_price_source"])
    check("the row is never relabeled as official", charge["truth_status"] == "modeled_funding",
          charge["truth_status"])
    check("the payment uses the reconstructed mark",
          abs(charge["payment_quote"] - 180.0 * 123.45 * 0.0001) < 1e-12, charge["payment_quote"])
    check("the series reports one modeled row in the window",
          funding["truth_status_counts"] == {"modeled_funding": 1}, funding["truth_status_counts"])
    check("the resolved identity names the modeled mark basis",
          funding["identity"]["modeled_mark_basis"] == cell.FUNDING_MODELED_MARK_BASIS,
          funding["identity"])
    check("the resolved identity reproduces the declared snapshot digest",
          funding["identity"]["modeled_mark_snapshot_sha256"] == identity["modeled_mark_snapshot_sha256"],
          (funding["identity"]["modeled_mark_snapshot_sha256"], identity["modeled_mark_snapshot_sha256"]))
    check("the snapshot names exactly the monthly file that was read",
          list(funding["mark_snapshot"]) == ["BTCUSDT/1h/BTCUSDT-1h-2025-05.jsonl.gz"],
          list(funding["mark_snapshot"]))
    check("the raw jittered stamp is preserved as source evidence",
          charge["funding_time_ms"] == boundary and charge["boundary_ms"] == boundary,
          (charge["funding_time_ms"], charge["boundary_ms"]))


def test_g2_a_jittered_modeled_stamp_floors_to_its_utc_hour():
    """+6 ms of registered jitter must not move the event across a boundary."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-g2-"))
    boundary = funding_boundary(2)
    identity = write_funding_fixture(
        tmp, [modeled_event(boundary + 6)], mark_bars=[(boundary, "123.45", "999.0")])
    rows = flat_cycle_rows()
    funding = load_fixture_funding(identity, rows)
    check("the jittered stamp floors onto the UTC hour of its mark bar",
          funding["events"][0]["boundary_ms"] == boundary, funding["events"][0])
    check("the raw millisecond stamp is kept untouched",
          funding["events"][0]["funding_time_ms"] == boundary + 6, funding["events"][0])
    run = replay(rows, funding=funding)
    charge = run["records"][0]["funding_events"][0]
    check("the jittered event still charges exactly one payment",
          abs(charge["payment_quote"] - 180.0 * 123.45 * 0.0001) < 1e-12, charge["payment_quote"])
    check("the charge lands on the boundary bar, not on a later one",
          charge["charge_index"] == 2, charge["charge_index"])


def test_h_missing_modeled_mark_holds():
    """A modeled row with no reconstructable mark must HOLD, never zero-fill."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-h-"))
    rows = flat_cycle_rows()
    no_bar = write_funding_fixture(
        tmp / "no-bar", [modeled_event(funding_boundary(2))], mark_bars=[(funding_boundary(3), "123.45")])
    check("a modeled row whose 1h mark bar is absent holds",
          raised(lambda: load_fixture_funding(no_bar, rows), "no 1h mark bar"))
    no_file = write_funding_fixture(tmp / "no-file", [modeled_event(funding_boundary(2))], mark_bars=[])
    check("a modeled row whose monthly mark file is absent holds",
          raised(lambda: load_fixture_funding(no_file, rows), "no 1h mark source"))
    non_positive = write_funding_fixture(
        tmp / "zero", [modeled_event(funding_boundary(2))], mark_bars=[(funding_boundary(2), "0")])
    check("a non-positive reconstructed mark holds",
          raised(lambda: load_fixture_funding(non_positive, rows), "not positive"))
    duplicate = write_funding_fixture(
        tmp / "dup", [modeled_event(funding_boundary(2))],
        mark_bars=[(funding_boundary(2), "123.45"), (funding_boundary(2), "124.0")])
    check("a duplicate 1h mark open_time holds instead of overwriting a bar",
          raised(lambda: load_fixture_funding(duplicate, rows), "duplicate 1h mark open_time"))


def test_i_a_wrong_source_or_mark_snapshot_digest_holds():
    """The identity is a proof obligation: a stale digest must not run."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-i-"))
    rows = flat_cycle_rows()
    boundary = funding_boundary(2)
    good = write_funding_fixture(tmp, [modeled_event(boundary)], mark_bars=[(boundary, "123.45")])
    check("the untouched fixture loads", len(load_fixture_funding(good, rows)["events"]) == 1)
    wrong_source = dict(good, source_sha256="0" * 64)
    check("a wrong funding source sha256 holds",
          raised(lambda: load_fixture_funding(wrong_source, rows), "funding source sha256 mismatch"))
    wrong_snapshot = dict(good, modeled_mark_snapshot_sha256="0" * 64)
    check("a wrong modeled mark snapshot sha256 holds",
          raised(lambda: load_fixture_funding(wrong_snapshot, rows), "modeled mark snapshot sha256 mismatch"))
    tampered_mark = Path(good["source_ref"]).parents[2] / "mark" / FUNDING_SYMBOL / "1h"
    with gzip.open(tampered_mark / "BTCUSDT-1h-2025-05.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"open_time_ms": boundary, "open": "124.0", "close": "124.0"}) + "\n")
    check("a mark file whose bytes changed holds, even with the declared digest",
          raised(lambda: load_fixture_funding(good, rows), "modeled mark snapshot sha256 mismatch"))
    check("an unreadable mark file also holds rather than being skipped",
          (tampered_mark / "BTCUSDT-1h-2025-05.jsonl.gz").write_bytes(b"not gzip\n")
          and raised(lambda: load_fixture_funding(good, rows), "not readable"))


def test_j_duplicate_bad_status_and_malformed_rate_hold():
    """Every structurally broken source is a HOLD, never a silent zero."""
    root = Path(tempfile.mkdtemp(prefix="research-cell-funding-j-"))
    rows = flat_cycle_rows()
    duplicate = write_funding_fixture(
        root / "dup", [official_event(funding_boundary(2)), official_event(funding_boundary(2))])
    check("a duplicate funding timestamp holds",
          raised(lambda: load_fixture_funding(duplicate, rows), "duplicate funding timestamp"))
    same_hour = write_funding_fixture(
        root / "hour", [official_event(funding_boundary(2)), official_event(funding_boundary(2) + 6)])
    check("two events in one UTC hour hold instead of double-charging a boundary",
          raised(lambda: load_fixture_funding(same_hour, rows), "share the UTC hour boundary"))
    bad_status = write_funding_fixture(
        root / "status", [official_event(funding_boundary(2), truth_status="estimated")])
    check("an unregistered truth_status holds",
          raised(lambda: load_fixture_funding(bad_status, rows), "truth_status"))
    bad_rate = write_funding_fixture(root / "rate", [official_event(funding_boundary(2), "abc")])
    check("a non-numeric funding rate holds",
          raised(lambda: load_fixture_funding(bad_rate, rows), "funding_rate"))
    infinite = write_funding_fixture(root / "inf", [official_event(funding_boundary(2), "inf")])
    check("a non-finite funding rate holds",
          raised(lambda: load_fixture_funding(infinite, rows), "not finite"))
    bad_mark = write_funding_fixture(root / "mark", [official_event(funding_boundary(2), mark="-1")])
    check("a non-positive official mark holds",
          raised(lambda: load_fixture_funding(bad_mark, rows), "not positive"))
    bad_stamp = write_funding_fixture(root / "stamp", [official_event(str(funding_boundary(2)))])
    check("a non-integer funding timestamp holds",
          raised(lambda: load_fixture_funding(bad_stamp, rows), "integer funding_time_ms"))
    other_symbol = write_funding_fixture(
        root / "symbol", [official_event(funding_boundary(2))], symbol="ETHUSDT")
    check("a source for another symbol holds",
          raised(lambda: load_fixture_funding(other_symbol, rows, symbol="BTCUSDT"), "does not match the cell symbol"))
    unreadable = {
        "basis": cell.FUNDING_BASIS,
        "source_ref": str(root / "absent" / "funding" / FUNDING_SYMBOL / "BTCUSDT-funding.jsonl.gz"),
        "source_sha256": "1" * 64,
        "truth_status": list(cell.FUNDING_TRUTH_STATUS),
        "modeled_mark_basis": cell.FUNDING_MODELED_MARK_BASIS,
        "modeled_mark_snapshot_sha256": "2" * 64,
    }
    check("a missing funding source holds", raised(lambda: load_fixture_funding(unreadable, rows), "not readable"))
    for label, broken in (
        ("missing key", {key: value for key, value in unreadable.items() if key != "source_sha256"}),
        ("wrong basis", dict(unreadable, basis="binance-spot-funding")),
        ("wrong modeled mark basis", dict(unreadable, modeled_mark_basis="mark-close")),
        ("wrong truth status set", dict(unreadable, truth_status=["official"])),
        ("relative source path", dict(unreadable, source_ref="funding/BTCUSDT/BTCUSDT-funding.jsonl.gz")),
    ):
        check("a malformed funding identity holds: " + label,
              raised(lambda broken=broken: load_fixture_funding(broken, rows), "funding identity is not usable"))


def test_k_a_zero_trade_run_still_has_zero_funding_and_valid_evidence():
    """Nothing was held, so nothing is charged, and the evidence says exactly that."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-k-"))
    rows = transition_rows([100.0, 101.0, 100.0])
    identity = write_funding_fixture(
        tmp, [official_event(funding_boundary(0)), official_event(funding_boundary(1))])
    funding = load_fixture_funding(identity, rows)
    check("the series still carries its in-window events", len(funding["events"]) == 2, funding["events"])
    run = replay(rows, funding=funding)
    check("a flat window replays no cycle", run["records"] == [], run["records"])
    check("a zero-trade run charges zero funding", run["funding_paid_quote"] == 0.0, run["funding_paid_quote"])
    check("zero-trade metrics report a zero funding total rather than omitting it",
          run["metrics"]["funding_paid_quote"] == 0.0, run["metrics"].get("funding_paid_quote"))
    initial = float(run["overlay"]["initial_capital_quote"])
    check("a zero-trade run still reconciles",
          cell.funding_reconciles(run["records"], run["metrics"], initial, run["funding_paid_quote"]))
    out = tmp / "evidence"
    cell.write_evidence(out, {"symbol": FUNDING_SYMBOL}, rows, run["records"],
                        [row["datetime"] for row in rows], run["projection"], run["metrics"])
    check("the funding evidence is written even with nothing charged",
          (out / "funding_events.csv").is_file())
    lines = (out / "funding_events.csv").read_text(encoding="utf-8").splitlines()
    check("a zero-charge run writes the funding header and no invented row", len(lines) == 1, lines)
    check("the header still names every required funding column",
          lines[0].split(",") == cell.funding_event_fieldnames(), lines[0])
    approved = cell_config()
    approved["research_cost_model"]["approved"] = True
    approved["research_cost_model"]["funding"] = dict(funding["identity"])
    applied = funding_applied_for(run, initial)
    check("a zero-trade run can never claim performance, however clean its funding",
          cell.performance_claimable(approved, run["metrics"], applied) is False)


def test_l_end_equity_reconciles_and_the_first_bar_baseline_survives_funding():
    """Net profit, funding and end equity must agree, first-bar baseline included."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-l-"))
    rows = transition_rows([100.0, 100.0, 103.0, 106.0, 104.0, 104.0, 200.0, 205.0, 212.0, 210.0, 209.0, 209.0],
                           entries={0, 6}, exits={4, 10})
    identity = write_funding_fixture(
        tmp, [official_event(funding_boundary(2)), official_event(funding_boundary(7))])
    run = replay(rows, funding=load_fixture_funding(identity, rows))
    initial = float(run["overlay"]["initial_capital_quote"])
    records, metrics = run["records"], run["metrics"]
    check("both cycles are replayed", len(records) == 2, len(records))
    dca_total = float(sum(record["dca_net_pnl_quote"] for record in records))
    funding_total = float(sum(record["funding_paid_quote"] for record in records))
    check("each cycle paid exactly its own events",
          all(abs(record["funding_paid_quote"] - sum(charge["payment_quote"] for charge in record["funding_events"])) < 1e-12
              for record in records), [record["funding_paid_quote"] for record in records])
    check("net profit is the fee result minus the funding",
          abs(metrics["net_profit_usdt"] - (dca_total - funding_total)) < 1e-9,
          (metrics["net_profit_usdt"], dca_total, funding_total))
    check("net profit is the sum of realized cycle net PnL",
          abs(metrics["net_profit_usdt"] - sum(record["realized_pnl_quote"] for record in records)) < 1e-12,
          metrics["net_profit_usdt"])
    check("end equity is initial capital plus net profit, funding included",
          abs(metrics["end_equity_quote"] - (initial + metrics["net_profit_usdt"])) < 1e-9,
          (metrics["end_equity_quote"], initial + metrics["net_profit_usdt"]))
    check("the equity evidence closes on the same end equity",
          abs(float(run["projection"][-1]["equity_quote"]) - metrics["end_equity_quote"]) < 1e-9,
          (run["projection"][-1]["equity_quote"], metrics["end_equity_quote"]))
    check("the reported funding total is the sum of the charges",
          abs(metrics["funding_paid_quote"] - funding_total) < 1e-12,
          (metrics["funding_paid_quote"], funding_total))
    check("the first bar return is still measured from frozen initial capital",
          abs(run["projection"][0]["bar_return"]
              - (run["projection"][0]["equity_quote"] - initial) / initial) < 1e-12,
          run["projection"][0])
    check("the first bar carries the entry fee and can carry no funding",
          run["projection"][0]["funding_paid_quote"] == 0.0, run["projection"][0])
    check("the cumulative funding column is monotone and ends on the total",
          [bar["funding_paid_quote"] for bar in run["projection"]][-1] == funding_total
          and all(a <= b + 1e-12 for a, b in zip([bar["funding_paid_quote"] for bar in run["projection"]],
                                                 [bar["funding_paid_quote"] for bar in run["projection"]][1:])),
          [bar["funding_paid_quote"] for bar in run["projection"]])
    check("the whole run reconciles", cell.funding_reconciles(records, metrics, initial, funding_total))
    check("a broken identity still fails to reconcile",
          cell.funding_reconciles([dict(records[0], funding_paid_quote=records[0]["funding_paid_quote"] + 1.0)]
                                  + records[1:], metrics, initial, funding_total) is False)


def test_l2_funding_boundaries_never_depend_on_the_host_timezone():
    """A naive-UTC row must not be re-read as host-local: TZ cannot move a payment."""
    import os
    import time as clock

    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-l2-"))
    rows = flat_cycle_rows()
    identity = write_funding_fixture(tmp, [official_event(funding_boundary(2))])
    payments, indexes = [], []
    original = os.environ.get("TZ")
    try:
        for zone in ("UTC", "Asia/Taipei", "America/New_York"):
            os.environ["TZ"] = zone
            clock.tzset()
            run = replay(rows, funding=load_fixture_funding(identity, rows))
            payments.append(round(run["records"][0]["funding_paid_quote"], 12))
            indexes.append(run["records"][0]["funding_events"][0]["charge_index"])
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        clock.tzset()
    check("the same series charges the same payment in every host timezone", len(set(payments)) == 1, payments)
    check("the charge lands on the same bar in every host timezone", len(set(indexes)) == 1, indexes)
    check("and the payment is the exact hand-computed amount", payments[0] == 1.8, payments)


def test_m_performance_claimable_requires_a_verified_applied_funding_model():
    """The claim opens only for an approved model the runtime itself re-verified."""
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-funding-m-"))
    rows = flat_cycle_rows()
    identity = write_funding_fixture(tmp, [official_event(funding_boundary(2))])
    funding = load_fixture_funding(identity, rows)
    run = replay(rows, funding=funding)
    initial = float(run["overlay"]["initial_capital_quote"])
    applied = funding_applied_for(run, initial)
    cfg = cell_config()
    cfg["research_cost_model"]["approved"] = True
    cfg["research_cost_model"]["funding"] = dict(funding["identity"])
    check("an approved, fully verified model makes the config runnable",
          cell.problems_in_config(cfg) == [], cell.problems_in_config(cfg))
    check("the applied economics match the declared model", cell.applied_cost_matches_model(cfg, applied)[0] is True,
          cell.applied_cost_matches_model(cfg, applied))
    check("the match reason names fees and the funding events",
          "funding event" in cell.applied_cost_matches_model(cfg, applied)[1],
          cell.applied_cost_matches_model(cfg, applied)[1])
    check("a verified model with a real trade and defined metrics IS claimable",
          cell.performance_claimable(cfg, run["metrics"], applied) is True,
          (run["metrics"]["total_trades"], run["metrics"]["cagr_pct"], run["metrics"]["sharpe"]))
    for label, mutate in (
        ("unapproved cost model", lambda c, a: c["research_cost_model"].update(approved=False)),
        ("declared slippage", lambda c, a: c["research_cost_model"].update(slippage=0.0002)),
        ("fee mismatch", lambda c, a: c["research_cost_model"].update(fees_per_side=0.0002)),
        ("legacy not_modeled funding", lambda c, a: c["research_cost_model"].update(funding="not_modeled")),
        ("malformed declared identity", lambda c, a: c["research_cost_model"]["funding"].pop("source_sha256")),
        ("wrong declared source sha", lambda c, a: c["research_cost_model"]["funding"].update(source_sha256="0" * 64)),
        ("wrong declared snapshot sha",
         lambda c, a: c["research_cost_model"]["funding"].update(modeled_mark_snapshot_sha256="0" * 64)),
        ("funding reported as not applied", lambda c, a: a.update(funding_applied=False)),
        ("funding reported as not reconciled", lambda c, a: a.update(reconciled=False)),
        ("no applied evidence at all", lambda c, a: a.clear()),
        ("runtime identity missing", lambda c, a: a.update(identity=None)),
        ("runtime identity for another source", lambda c, a: a.update(identity=dict(a["identity"], source_sha256="0" * 64))),
    ):
        broken_cfg, broken_applied = cell_config(), dict(applied)
        broken_cfg["research_cost_model"]["approved"] = True
        broken_cfg["research_cost_model"]["funding"] = dict(funding["identity"])
        broken_applied["identity"] = dict(applied["identity"])
        broken_cfg["research_cost_model"]["funding"] = dict(funding["identity"])
        mutate(broken_cfg, broken_applied)
        check("claim stays false: " + label,
              cell.performance_claimable(broken_cfg, run["metrics"], broken_applied) is False,
              cell.applied_cost_matches_model(broken_cfg, broken_applied))
    check("claim stays false with no trade",
          cell.performance_claimable(cfg, dict(run["metrics"], total_trades=0), applied) is False)
    check("claim stays false with an undefined CAGR",
          cell.performance_claimable(cfg, dict(run["metrics"], cagr_pct=None), applied) is False)
    check("claim stays false with an undefined Sharpe",
          cell.performance_claimable(cfg, dict(run["metrics"], sharpe=None), applied) is False)
    legacy = cell_config()
    legacy["research_cost_model"]["approved"] = True
    legacy_run = replay(rows)
    check("a legacy not_modeled run is valid evidence but never claimable",
          cell.performance_claimable(legacy, legacy_run["metrics"],
                                     funding_applied_for(legacy_run, initial)) is False,
          cell.applied_cost_matches_model(legacy, funding_applied_for(legacy_run, initial)))


def test_n_declared_costs_must_be_explicit_finite_json_numbers():
    """B1: an unknown or invalid declared cost is never invented as 0.0.

    The independent audit reproduced a ``slippage: null`` config opening a
    performance claim on a real provider run, and empty/boolean values behaved the
    same way. Every declared cost is therefore an explicit finite JSON number, and
    the only slippage this replay can match is exactly 0.0. No production fee or
    slippage number is chosen or approved here: the fixture values are the ones the
    committed synthetic spec already carries.
    """
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-costs-n-"))
    rows = flat_cycle_rows()
    identity = write_funding_fixture(tmp, [official_event(funding_boundary(2))])
    funding = load_fixture_funding(identity, rows)
    run = replay(rows, funding=funding)
    initial = float(run["overlay"]["initial_capital_quote"])
    applied = funding_applied_for(run, initial)

    def approved_cfg():
        """Return the synthetic approved config whose costs the run really charges."""
        cfg = cell_config()
        cfg["research_cost_model"]["approved"] = True
        cfg["research_cost_model"]["funding"] = dict(funding["identity"])
        return cfg

    cfg = approved_cfg()
    check("the explicit-number positive control stays runnable", cell.problems_in_config(cfg) == [],
          cell.problems_in_config(cfg))
    matches, reason = cell.applied_cost_matches_model(cfg, applied)
    check("an explicit 0.0 slippage with a matching fee still matches", matches is True, reason)
    check("that control is still claimable on a real cycle",
          cell.performance_claimable(cfg, run["metrics"], applied) is True,
          cell.applied_cost_matches_model(cfg, applied))
    near_zero = dict(cfg, research_cost_model=dict(cfg["research_cost_model"], slippage=1e-7))
    check("exactly 0.0 is the only slippage value the replay can match",
          cell.applied_cost_matches_model(near_zero, applied)[0] is False,
          cell.applied_cost_matches_model(near_zero, applied))

    invalid = (
        (None, "null"),
        ("", "empty string"),
        ("0.0", "numeric string"),
        ("abc", "non-numeric string"),
        (False, "false"),
        (True, "true"),
        (float("nan"), "NaN"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
    )
    for where, key in (
        ("research_cost_model", "slippage"),
        ("research_cost_model", "fees_per_side"),
        ("house_overlay", "trade_cost"),
    ):
        for value, label in invalid:
            broken = approved_cfg()
            broken[where][key] = value
            matches, reason = cell.applied_cost_matches_model(broken, applied)
            check("%s.%s as %s never matches" % (where, key, label), matches is False, (value, reason))
            check("... the reason names the exact field", key in reason, reason)
            check("... and the claim stays closed",
                  cell.performance_claimable(broken, run["metrics"], applied) is False, reason)
        absent = approved_cfg()
        absent[where].pop(key)
        matches, reason = cell.applied_cost_matches_model(absent, applied)
        check("%s.%s absent never matches" % (where, key), matches is False and key in reason, reason)

    null_slippage = approved_cfg()
    null_slippage["research_cost_model"]["slippage"] = None
    matches, reason = cell.applied_cost_matches_model(null_slippage, applied)
    check("the audited null-slippage case is refused by the explicit-number rule",
          matches is False and "slippage" in reason and "explicit finite JSON number" in reason, reason)
    check("the null-slippage reason names the null value, not an implied zero",
          "None" in reason and "or 0.0" not in reason, reason)


def test_o_boolean_funding_numbers_hold_instead_of_zero_filling():
    """B2: a JSON boolean is not a funding number and never becomes 0.0 or 1.0.

    ``float(False) == 0.0`` and ``float(True) == 1.0``, so a malformed boolean rate
    silently charged zero funding and a boolean mark was read as a price of 1.0.
    Both now HOLD before any conversion, while the registered decimal-string form
    the source actually carries keeps working.
    """
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-bool-o-"))
    for value, label in ((False, "false"), (True, "true")):
        rate_root = tmp / ("rate-" + label)
        identity = write_funding_fixture(rate_root, [malformed_event(funding_boundary(2), "funding_rate", value)])
        message = hold_message(lambda: load_fixture_funding(identity, flat_cycle_rows()))
        check("a %s funding_rate is a HOLD, never a zero-filled charge" % label,
              message is not None and "funding_rate" in message and "boolean" in message, message)
        mark_root = tmp / ("mark-" + label)
        mark_identity = write_funding_fixture(mark_root, [malformed_event(funding_boundary(2), "mark_price", value)])
        message = hold_message(lambda: load_fixture_funding(mark_identity, flat_cycle_rows()))
        check("a %s official mark_price is a HOLD" % label,
              message is not None and "mark_price" in message and "boolean" in message, message)
        modeled_root = tmp / ("modeled-" + label)
        modeled_identity = write_funding_fixture(modeled_root, [modeled_event(funding_boundary(2))])
        marks = modeled_mark_month(modeled_root, [(funding_boundary(2), value)])
        modeled_identity["modeled_mark_snapshot_sha256"] = cell.modeled_mark_snapshot(marks)
        message = hold_message(lambda: load_fixture_funding(modeled_identity, flat_cycle_rows()))
        check("a %s modeled 1h mark open is a HOLD" % label,
              message is not None and "mark open" in message and "boolean" in message, message)
    for label, value in (("NaN", float("nan")), ("Infinity", float("inf"))):
        message = hold_message(lambda: cell.finite_float(value, "funding_rate at 0"))
        check("a %s funding_rate is a HOLD, not a number" % label,
              message is not None and "not finite" in message, message)
    accepted = write_funding_fixture(tmp / "string-form",
                                     [official_event(funding_boundary(2), rate="0.0001", mark="100.0")])
    series = load_fixture_funding(accepted, flat_cycle_rows())
    check("the registered decimal-string rate and mark stay accepted",
          series["events"][0]["funding_rate"] == 0.0001 and series["events"][0]["mark_price"] == 100.0,
          series["events"])
    run = replay(flat_cycle_rows(), funding=series)
    check("and that string-form event still charges exactly 1.8 quote",
          abs(run["records"][0]["funding_paid_quote"] - 1.8) < 1e-12, run["records"][0]["funding_paid_quote"])


def test_p_persisted_economics_boundary_matches_the_applied_funding_scope():
    """N1: the persisted boundary may not call funding unmodeled while it is applied.

    The same block is written into every ``effective_config.json``, copied into the
    run's ``summary.json`` and (through the repro artifacts) into MLflow, so it must
    describe exactly what this replay does: leverage is an exposure multiplier,
    isolated-margin mechanics and slippage are absent, funding is applied only from
    a verified structured identity and is never zero-filled, and Node13 Hummingbot
    keeps final execution economics and certification.
    """
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-boundary-p-"))
    spec_path = tmp / "campaign-spec.json"
    spec_path.write_text(json.dumps(spec_for()), encoding="utf-8")
    root = tmp / "cells"
    manifest = cell.materialize(spec_path, root)
    persisted_path = root / manifest["cells"][0]["cell_id"] / "effective_config.json"
    persisted_text = persisted_path.read_text(encoding="utf-8")
    boundary = json.loads(persisted_text)["economics_boundary"]
    check("the effective config persists the boundary block",
          isinstance(boundary, dict) and bool(boundary), boundary)
    for phrase in ("isolated_margin_funding_slippage", "not modeled here", "funding not modeled",
                   "funding is not modeled", "funding: not modeled"):
        check("the persisted config never says %r" % phrase, phrase not in persisted_text, phrase)
    check("the conflated single scope key is replaced by separate scopes",
          all(key in boundary for key in ("research_leverage_scope", "isolated_margin", "slippage", "funding",
                                          "funding_scope", "final_execution_and_certification")),
          sorted(boundary))
    check("research leverage is persisted as an exposure multiplier",
          boundary["research_leverage_is_exposure_multiplier"] is True
          and "exposure multiplier" in boundary["research_leverage_scope"],
          boundary["research_leverage_scope"])
    check("isolated-margin mechanics are the thing that is not modeled",
          "not modeled" in boundary["isolated_margin"] and "liquidation" in boundary["isolated_margin"],
          boundary["isolated_margin"])
    check("slippage is persisted as not applied", "not applied" in boundary["slippage"], boundary["slippage"])
    check("funding is persisted as conditional on a verified structured identity",
          "verified" in boundary["funding"] and "not_modeled" in boundary["funding"]
          and "zero-filled" in boundary["funding"], boundary["funding"])
    check("the machine-readable funding scope pins that conditionality",
          boundary["funding_scope"] == "conditional_verified_identity_else_not_modeled_unclaimable",
          boundary["funding_scope"])
    check("Node13 Hummingbot stays the final execution/certification owner",
          "Node13 Hummingbot" in boundary["final_execution_and_certification"]
          and "certification" in boundary["final_execution_and_certification"],
          boundary["final_execution_and_certification"])
    for key, value in boundary.items():
        lowered = str(value).lower()
        if "fund" in lowered:
            check("the funding scope %r never calls funding unmodeled" % key,
                  "not modeled" not in lowered and "not modelled" not in lowered, value)
    summary_boundary = dict(boundary, research_cost_model=json.loads(persisted_text)["research_cost_model"],
                            cost_model_approved=False, performance_claimable=False)
    check("the summary copy cannot re-word the boundary",
          all(summary_boundary[key] == boundary[key] for key in boundary), sorted(summary_boundary))
    check("a funding_applied run's summary wording stays compatible with the persisted scope",
          "when research_cost_model" in boundary["funding"] and "charged into the same bar's equity"
          in boundary["funding"], boundary["funding"])


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print("ALL RESEARCH CELL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
