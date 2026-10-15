"""Checks for the config-driven formal research cell (42-cell round0 mechanics).

Run: python3 tests/backtest/test_research_cell.py   (or pytest tests/backtest)

These are implementation checks, not a campaign. They prove, on an isolated
local fixture:
- the frozen 42 identities are exactly 7 x 3 x 2, unique, and stable;
- a cell config with an unapproved overlay/cost value fails closed;
- materialization writes 42 bindable cell dirs and a hash-consistent manifest;
- research_leverage scales the house quote exposure and the realized ledger by
  exactly 5/3 from 3x to 5x, reproducibly, through the exported
  Qlib <-> Hummingbot DCA path;
- the BO + 2 SO stage semantics of that path are not broken.

No Qlib provider, no campaign run, no network. The DCA module and the cell
module are loaded by path so the math is testable in any interpreter.
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DCA_PATH = ROOT / "qlib" / "backtest" / "hummingbot_dca.py"
CELL_PATH = ROOT / "qlib" / "backtest" / "research_cell.py"


def load_module(name, path):
    """Load one repo module by path (no package import, no heavy dependencies)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cell = load_module("research_cell_under_test", CELL_PATH)
dca = load_module("hummingbot_dca_under_test", DCA_PATH)


def check(name, condition, extra=None):
    print(("PASS " if condition else "FAIL ") + name + ("" if condition else " " + str(extra)))
    if not condition:
        raise SystemExit("FAILED: " + name + " " + str(extra))


def spec_for(cells_dir="/results/_validation/formal-cells-round0"):
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
            "entry_len": 20,
            "exit_len": 10,
            "window_start_utc": "2025-04-19 16:00:00",
            "window_end_utc_inclusive": "2026-04-19 16:00:00",
        },
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
    good = cell.build_effective_config(spec_for(), cell.campaign_identities()[0])
    check("materialized cell config is runnable", cell.problems_in_config(good) == [],
          cell.problems_in_config(good))
    check("an unapproved cost model is declared but not claimable",
          cell.cost_model_claimable(good) is False)
    claimable = cell.build_effective_config(spec_for(), cell.campaign_identities()[0])
    claimable["research_cost_model"]["approved"] = True
    check("an approved cost model is claimable and still runnable",
          cell.cost_model_claimable(claimable) is True and cell.problems_in_config(claimable) == [])
    for label, mutate in (
        ("missing so_spacing", lambda cfg: cfg["house_overlay"].pop("so_spacing_pct")),
        ("missing take_profit declaration", lambda cfg: cfg["house_overlay"].pop("take_profit_pct")),
        ("undeclared cost model approval", lambda cfg: cfg["research_cost_model"].pop("approved")),
        ("leverage outside the frozen set", lambda cfg: cfg.update(research_leverage=4)),
        ("symbol outside the frozen set", lambda cfg: cfg["strategy"].update(symbol="PEPEUSDT")),
        ("missing provider", lambda cfg: cfg["data"].pop("provider_ref")),
    ):
        broken = cell.build_effective_config(spec_for(), cell.campaign_identities()[0])
        mutate(broken)
        check("fail closed: " + label, cell.problems_in_config(broken) != [])


def test_materialize_writes_42_bindable_cells():
    tmp = Path(tempfile.mkdtemp(prefix="research-cell-"))
    spec = spec_for()
    spec_path = tmp / "campaign-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    root = tmp / "cells"
    manifest = cell.materialize(spec_path, root)
    entry_bytes = CELL_PATH.read_bytes()
    check("materialized exactly 42 cells", manifest["cell_count"] == 42 == len(manifest["cells"]))
    check("campaign manifest is written", (root / "campaign-manifest.json").is_file())
    good = 0
    for entry in manifest["cells"]:
        cell_dir = root / entry["cell_id"]
        config_path = cell_dir / "effective_config.json"
        entrypoint_path = cell_dir / "run_cell.entrypoint.py"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        same_source = entrypoint_path.read_bytes() == entry_bytes
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
        )
        good += int(same_source and bindable)
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


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print("ALL RESEARCH CELL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
