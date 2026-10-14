#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import gzip
import hashlib
import inspect
import json
import logging
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Dict, Text, Any

from ...contrib.eva.alpha import calc_ic
from ...workflow.record_temp import RecordTemp
from ...workflow.record_temp import SignalRecord
from ...data import dataset as qlib_dataset
from ...log import get_module_logger

logger = get_module_logger("workflow", logging.INFO)


class MultiSegRecord(RecordTemp):
    """
    This is the multiple segments signal record class that generates the signal prediction.
    This class inherits the ``RecordTemp`` class.
    """

    def __init__(self, model, dataset, recorder=None):
        super().__init__(recorder=recorder)
        if not isinstance(dataset, qlib_dataset.DatasetH):
            raise ValueError("The type of dataset is not DatasetH instead of {:}".format(type(dataset)))
        self.model = model
        self.dataset = dataset

    def generate(self, segments: Dict[Text, Any], save: bool = False):
        for key, segment in segments.items():
            predics = self.model.predict(self.dataset, segment)
            if isinstance(predics, pd.Series):
                predics = predics.to_frame("score")
            labels = self.dataset.prepare(
                segments=segment, col_set="label", data_key=qlib_dataset.handler.DataHandlerLP.DK_R
            )
            # Compute the IC and Rank IC
            ic, ric = calc_ic(predics.iloc[:, 0], labels.iloc[:, 0])
            results = {"all-IC": ic, "mean-IC": ic.mean(), "all-Rank-IC": ric, "mean-Rank-IC": ric.mean()}
            logger.info("--- Results for {:} ({:}) ---".format(key, segment))
            ic_x100, ric_x100 = ic * 100, ric * 100
            logger.info("IC: {:.4f}%".format(ic_x100.mean()))
            logger.info("ICIR: {:.4f}%".format(ic_x100.mean() / ic_x100.std()))
            logger.info("Rank IC: {:.4f}%".format(ric_x100.mean()))
            logger.info("Rank ICIR: {:.4f}%".format(ric_x100.mean() / ric_x100.std()))

            if save:
                save_name = "results-{:}.pkl".format(key)
                self.save(**{save_name: results})
                logger.info(
                    "The record '{:}' has been saved as the artifact of the Experiment {:}".format(
                        save_name, self.recorder.experiment_id
                    )
                )


class SignalMseRecord(RecordTemp):
    """
    This is the Signal MSE Record class that computes the mean squared error (MSE).
    This class inherits the ``SignalMseRecord`` class.
    """

    artifact_path = "sig_analysis"
    depend_cls = SignalRecord

    def __init__(self, recorder, **kwargs):
        super().__init__(recorder=recorder, **kwargs)

    def generate(self):
        self.check()

        pred = self.load("pred.pkl")
        label = self.load("label.pkl")
        masks = ~np.isnan(label.values)
        mse = mean_squared_error(pred.values[masks], label[masks])
        metrics = {"MSE": mse, "RMSE": np.sqrt(mse)}
        objects = {"mse.pkl": mse, "rmse.pkl": np.sqrt(mse)}
        self.recorder.log_metrics(**metrics)
        self.save(**objects)
        logger.info("The evaluation results in SignalMseRecord is {:}".format(metrics))

    def list(self):
        return ["mse.pkl", "rmse.pkl"]


class S01509DCARecord(RecordTemp):
    """Config-driven Donchian signal/standalone-DCA diagnostic, not a full strategy backtest.

    Instantiate through ``init_instance_by_config`` with an existing Recorder.
    The JSON file is consumed before execution; its relative input files are
    copied unchanged into the same run for offline replay. No qrun/model shim
    or generic Account is involved. This preserves the existing bounded DCA
    path: an observed Donchian exit bounds each executor replay. It does NOT
    prove causal controller execution, base fills, account equity, isolated
    margin, funding or slippage, and cannot give a formal Node04/05 ACK.
    """

    artifact_path = "s01509_diagnostic"

    def __init__(self, config_path, recorder):
        super().__init__(recorder)
        self.config_path = Path(config_path)

    def generate(self):
        # Keep exact-int checks (bool must fail); import here to avoid changing dependency order.
        # pylint: disable=import-outside-toplevel,unidiomatic-typecheck
        from ...backtest import backtest_hummingbot_dca

        config_bytes = self.config_path.read_bytes()
        cfg = json.loads(config_bytes)
        required = {
            "family_id",
            "input_kind",
            "symbol",
            "timeframe",
            "source",
            "data",
            "window",
            "entry_length",
            "exit_length",
            "capital",
            "dca",
            "engine_image_digest",
            "launch_command",
        }
        if set(cfg) != required or cfg["family_id"] != "S01509":
            raise ValueError("Only the complete S01509 diagnostic Config is supported")
        if cfg["input_kind"] not in ("raw", "synthetic") or (cfg["symbol"], cfg["timeframe"]) != ("BTCUSDT", "1h"):
            raise ValueError("Declare raw/synthetic BTCUSDT 1h inputs explicitly")
        lengths = [cfg["entry_length"], cfg["exit_length"]]
        if any(type(n) is not int or n <= 0 for n in lengths):
            raise ValueError("Donchian lengths must be positive integers")
        window = cfg["window"]
        if set(window) != {"start_open_ms", "end_open_ms"}:
            raise ValueError("An explicit inclusive open-time window is required")
        lo, hi = window["start_open_ms"], window["end_open_ms"]
        hour = 3600000
        if any(type(t) is not int or t % hour for t in (lo, hi)) or lo > hi:
            raise ValueError("Window must be ordered integer UTC hour boundaries")
        capital, dca = cfg["capital"], cfg["dca"]
        if set(capital) != {"initial_margin_quote", "leverage", "margin_mode", "margin_fractions"}:
            raise ValueError("Complete house-derived capital allocation is required")
        if capital["margin_mode"] != "isolated" or type(capital["leverage"]) is not int or capital["leverage"] != 3:
            raise ValueError("Only the declared house-derived isolated 3x allocation is supported")
        fractions = [Decimal(str(v)) for v in capital["margin_fractions"]]
        margin = Decimal(str(capital["initial_margin_quote"]))
        if len(fractions) != 3 or any(not v.is_finite() or v <= 0 for v in [margin, *fractions]):
            raise ValueError("Declare finite positive BO/SO1/SO2 margin allocations")
        if sum(fractions) > 1:
            raise ValueError("Margin allocation exceeds the declared initial margin")
        if set(dca) != {"side", "mode", "safety_offsets", "take_profit", "stop_loss", "trailing_stop", "trade_cost"}:
            raise ValueError("Complete native DCA parameters are required")
        if (dca["side"], dca["mode"]) != ("BUY", "MAKER"):
            raise ValueError("This S01509 diagnostic only supports BUY/MAKER")
        if any(dca[k] is not None for k in ("take_profit", "stop_loss", "trailing_stop")):
            raise ValueError("This diagnostic is bounded by Donchian/window exits, not extra exit rules")
        offsets = [Decimal(str(v)) for v in dca["safety_offsets"]]
        if len(offsets) != 3 or any(not v.is_finite() or not 0 <= v < 1 for v in offsets):
            raise ValueError("Three finite ordered DCA offsets are required")
        if offsets[0] != 0 or offsets != sorted(set(offsets)):
            raise ValueError("BO offset must be zero and safety offsets strictly increasing")
        if not np.isfinite(dca["trade_cost"]) or dca["trade_cost"] < 0:
            raise ValueError("Explicit finite non-negative per-side trade cost is required")

        # Config paths are portable basenames, never arbitrary file/URL readers.
        if set(cfg["source"]) != {"file", "sha256", "uri", "ref"} or any(
            set(item) != {"file", "sha256"} for item in cfg["data"]
        ):
            raise ValueError("Exact source provenance and data SHA256 entries are required")
        inputs = [cfg["source"], *cfg["data"]]
        names = [item["file"] for item in inputs]
        if not cfg["data"] or len(set(names)) != len(names) or self.config_path.name in names:
            raise ValueError("Distinct source/data input files are required")
        for item in inputs:
            name = item["file"]
            if not name or name in (".", "..") or Path(name).name != name:
                raise ValueError("Input files must be relative basenames")
            path = self.config_path.parent / name
            if path.resolve().parent != self.config_path.parent.resolve():
                raise ValueError("Input symlinks must not escape the Config directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Input SHA256 mismatch: {name}")
        warmup = max(lengths)
        rows = {}
        for item in cfg["data"]:
            with gzip.open(self.config_path.parent / item["file"], "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    t = row["open_time_ms"]
                    if type(t) is not int:
                        raise ValueError("Raw open_time_ms must be integer milliseconds")
                    if lo - warmup * hour <= t <= hi + hour:
                        if t in rows:
                            raise ValueError("Duplicate raw candle timestamp")
                        prices = [Decimal(str(row[k])) for k in ("open", "high", "low", "close")]
                        if any(not v.is_finite() or v <= 0 for v in prices):
                            raise ValueError("Raw OHLC must be finite and positive")
                        if prices[1] < max(prices) or prices[2] > min(prices):
                            raise ValueError("Raw OHLC bounds are inconsistent")
                        rows[t] = row
        if sorted(rows) != list(range(lo - warmup * hour, hi + 2 * hour, hour)):
            raise ValueError("Missing warmup, window candle or force-flat bar")
        rows = [rows[t] for t in sorted(rows)]
        signals, episodes, ledgers = [], [], []
        entry_index = None
        # ponytail: bounded offline O(N*(entry_length+exit_length)) signal replay;
        # retain native quote-allocation semantics, not a new fill/account engine.
        for i in range(warmup, len(rows)):
            row = rows[i]
            t, close = row["open_time_ms"], Decimal(str(row["close"]))
            entry = close > max(Decimal(str(r["high"])) for r in rows[i - lengths[0] : i])
            exit_signal = close < min(Decimal(str(r["low"])) for r in rows[i - lengths[1] : i])
            if lo <= t <= hi and entry:
                signals.append({"signal": "entry", "open_time_ms": t, "price": str(close)})
            if entry_index is None and lo <= t <= hi and entry:
                entry_index = i
            elif entry_index is not None and (exit_signal or t > hi):
                if exit_signal:
                    signals.append({"signal": "exit", "open_time_ms": t, "price": str(close)})
                bars = rows[entry_index : i + 1]
                frame = pd.DataFrame(
                    [
                        {
                            "timestamp": float(r["open_time_ms"] // 1000 + 3600),
                            **{k: float(r[k]) for k in ("open", "high", "low", "close", "volume")},
                        }
                        for r in bars
                    ]
                ).set_index("timestamp", drop=False)
                entry_price = Decimal(str(bars[0]["close"]))
                # ponytail: same float/Decimal boundary conversion as the existing replay.
                levels = [Decimal.from_float(float(entry_price * (1 - offset))) for offset in offsets]
                amounts = [margin * capital["leverage"] * fraction for fraction in fractions]
                parameters = {
                    "prices": levels,
                    "amounts_quote": amounts,
                    **{k: dca[k] for k in dca if k != "safety_offsets"},
                    "time_limit": (len(bars) - 1) * 3600,
                }
                ledger, close_type = backtest_hummingbot_dca(frame, **parameters)
                episodes.append(
                    {
                        "entry_open_ms": bars[0]["open_time_ms"],
                        "exit_open_ms": t,
                        "exit_reason": "donchian_exit" if exit_signal else "window_end_force_flat",
                        "close_type": close_type,
                        "parameters": parameters,
                    }
                )
                ledgers.append(ledger)
                entry_index = None

        self.recorder.set_tags(scope="s01509_bounded_signal_dca_diagnostic", input_kind=cfg["input_kind"])
        self.recorder.log_params(
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            family_id=cfg["family_id"],
            entry_length=lengths[0],
            exit_length=lengths[1],
            trade_cost=dca["trade_cost"],
        )
        self.recorder.log_metrics(input_bars=len(rows), episode_count=len(episodes), ledger_rows=sum(map(len, ledgers)))
        self.save(**{"signals.pkl": signals, "episodes.pkl": episodes, "ledgers.pkl": ledgers})
        self.recorder.save_objects(local_path=self.config_path, artifact_path=self.get_path("inputs"))
        for name in names:
            self.recorder.save_objects(local_path=self.config_path.parent / name, artifact_path=self.get_path("inputs"))
        sources = {
            "record": inspect.getsourcefile(type(self)),
            "dca": inspect.getsourcefile(backtest_hummingbot_dca),
            "recorder": inspect.getsourcefile(type(self.recorder)),
        }
        if any(path is None for path in sources.values()):
            raise ValueError("Native source files must be available for replay fingerprints")
        sources = {name: Path(path) for name, path in sources.items() if path is not None}
        for name, path in sources.items():
            self.recorder.save_objects(local_path=path, artifact_path=self.get_path(f"engine/{name}"))
        self.save(
            **{
                "metadata.pkl": {
                    "config": cfg,
                    "input_hashes": {item["file"]: item["sha256"] for item in inputs},
                    "versions": {name: version(name) for name in ("pyqlib", "mlflow", "pandas", "numpy")},
                    "source_hashes": {
                        name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in sources.items()
                    },
                    "execution_path": "qlib.backtest.backtest_hummingbot_dca",
                    "exit_bound": "observed_donchian_exit_or_first_bar_after_window",
                    "scope": "house_derived_bounded_diagnostic_not_full_strategy",
                    "missing": [
                        "causal_controller_execution",
                        "base_fill_journal",
                        "account_equity",
                        "isolated_margin_accounting",
                        "funding",
                        "slippage",
                    ],
                }
            }
        )
        return signals, episodes, ledgers
