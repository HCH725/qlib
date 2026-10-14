"""Offline persistence contract example, not a deployed Runner or fill exporter."""

import gzip
import hashlib
import inspect
import json
import os
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from qlib.backtest import backtest_hummingbot_dca
from qlib.workflow.recorder import MLflowRecorder


REQUIRED = ("ledger.pkl", "metadata.json", "candles.json", "parameters.json", "native_dca.py")


def _verify_download(root, expected_manifest):
    # Fixed artifact demo: the expected manifest is held by the caller,
    # separately from the downloaded files. Production must additionally bind
    # this expectation to its independently verified submit/run context.
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["run_id"] == expected_manifest["run_id"]
    assert set(manifest["artifacts"]) == set(REQUIRED)
    assert set(expected_manifest["artifacts"]) == set(REQUIRED)
    for name in REQUIRED:
        payload = (root / name).read_bytes()
        entry = manifest["artifacts"][name]
        assert len(payload) == entry["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert entry == expected_manifest["artifacts"][name]


@pytest.fixture
def saved_dca(tmp_path, monkeypatch):
    """Run real DCA logic on explicitly synthetic bars and use a real file store."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = (tmp_path / "mlruns").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("dca-artifact-contract")
    run = client.create_run(experiment_id, tags={"scope": "synthetic_native_dca_input_replay_test"})
    recorder = MLflowRecorder(experiment_id, uri, mlflow_run=run)
    timestamps = [1000.0, 2000.0, 3000.0]
    candles = pd.DataFrame(
        {"timestamp": timestamps, "close": [100.0, 90.0, 110.0], "low": [100.0, 90.0, 110.0]},
        index=pd.Index(timestamps, name="timestamp"),
    )
    parameters = {
        "prices": ["100", "90"],
        "amounts_quote": ["100", "100"],
        "take_profit": "0.05",
        "stop_loss": "0.10",
        "trade_cost": 0.0002,
        "side": "BUY",
        "mode": "MAKER",
    }
    ledger, close_type = backtest_hummingbot_dca(candles, **parameters)
    assert close_type == "TAKE_PROFIT"
    recorder.save_objects(artifact_path="hummingbot_dca", **{"ledger.pkl": ledger})
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "candles.json").write_text(json.dumps(candles.to_dict(orient="tight"), allow_nan=False))
    (upload / "parameters.json").write_text(json.dumps(parameters, sort_keys=True))
    source_file = inspect.getsourcefile(backtest_hummingbot_dca)
    assert source_file is not None
    source = Path(source_file)
    (upload / "native_dca.py").write_bytes(source.read_bytes())
    for name in ("candles.json", "parameters.json", "native_dca.py"):
        recorder.save_objects(local_path=upload / name, artifact_path="hummingbot_dca")
    metadata = {
        "fixture": "synthetic",
        "close_type": close_type,
        "parameters": parameters,
        "versions": {name: version(name) for name in ("pyqlib", "mlflow", "pandas", "numpy")},
        "execution_path": "qlib.backtest.backtest_hummingbot_dca",
        "scope": "executor_only_not_full_strategy",
    }
    (upload / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
    recorder.save_objects(local_path=upload / "metadata.json", artifact_path="hummingbot_dca")
    manifest = {"run_id": run.info.run_id, "artifacts": {}}
    for name in REQUIRED:
        payload = Path(client.download_artifacts(run.info.run_id, f"hummingbot_dca/{name}")).read_bytes()
        manifest["artifacts"][name] = {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    (upload / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    recorder.save_objects(local_path=upload / "manifest.json", artifact_path="hummingbot_dca")
    download = tmp_path / "download"
    download.mkdir()
    root = Path(client.download_artifacts(run.info.run_id, "hummingbot_dca", dst_path=str(download)))
    try:
        yield recorder, ledger, metadata, root, manifest
    finally:
        client.set_terminated(run.info.run_id)


def test_raw_dca_ledger_is_saved_read_back_and_hash_verified(saved_dca):
    recorder, ledger, metadata, root, manifest = saved_dca
    _verify_download(root, manifest)
    pd.testing.assert_frame_equal(recorder.load_object("hummingbot_dca/ledger.pkl"), ledger)
    assert json.loads((root / "metadata.json").read_text()) == metadata
    assert "filled_amount_quote_0" in ledger and "filled_amount_quote_1" in ledger
    # The returned frame is NOT a base-quantity fill journal or account equity.
    assert "quantity_base" not in ledger and "equity" not in ledger


def test_saved_inputs_reproduce_native_dca_result(saved_dca):
    _, ledger, metadata, root, manifest = saved_dca
    _verify_download(root, manifest)
    candles = pd.DataFrame.from_dict(json.loads((root / "candles.json").read_text()), orient="tight")
    parameters = json.loads((root / "parameters.json").read_text())
    source_file = inspect.getsourcefile(backtest_hummingbot_dca)
    assert source_file is not None
    source = Path(source_file)
    assert source.read_bytes() == (root / "native_dca.py").read_bytes()
    assert {name: version(name) for name in metadata["versions"]} == metadata["versions"]
    replay, close_type = backtest_hummingbot_dca(candles, **parameters)
    pd.testing.assert_frame_equal(replay, ledger, check_exact=True)
    assert close_type == metadata["close_type"]
    assert metadata["scope"] == "executor_only_not_full_strategy"


def test_replay_consumes_saved_config_not_hardcoded_parameters(saved_dca):
    _, _, metadata, root, manifest = saved_dca
    _verify_download(root, manifest)
    candles = pd.DataFrame.from_dict(json.loads((root / "candles.json").read_text()), orient="tight")
    parameters = json.loads((root / "parameters.json").read_text())
    parameters["take_profit"] = "0.50"  # Synthetic control, not a strategy retune.
    _, close_type = backtest_hummingbot_dca(candles, **parameters)
    assert close_type == "TIME_LIMIT" != metadata["close_type"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "truncate",
        "wrong_hash",
        "missing_manifest_entry",
        "wrong_run",
        "missing_input",
        "changed_config",
        "changed_source",
    ],
)
def test_missing_or_corrupt_artifacts_cannot_pass_acceptance(saved_dca, fault):
    _, _, _, root, manifest = saved_dca
    if fault == "missing":
        (root / "ledger.pkl").unlink()
    elif fault == "truncate":
        payload = (root / "ledger.pkl").read_bytes()
        (root / "ledger.pkl").write_bytes(payload[:-1])
    elif fault == "missing_input":
        (root / "candles.json").unlink()
    elif fault == "changed_config":
        (root / "parameters.json").write_text("{}")
    elif fault == "changed_source":
        (root / "native_dca.py").write_text("# not the saved executor\n")
    else:
        altered = json.loads((root / "manifest.json").read_text())
        if fault == "wrong_hash":
            altered["artifacts"]["ledger.pkl"]["sha256"] = "0" * 64
        elif fault == "missing_manifest_entry":
            del altered["artifacts"]["ledger.pkl"]
        else:
            altered["run_id"] = "different-run"
        (root / "manifest.json").write_text(json.dumps(altered))
    with pytest.raises((AssertionError, FileNotFoundError)):
        _verify_download(root, manifest)


def _s01509_config(root, real=False):
    """Test intake only: raw mode copies existing read-only bytes, never fetches markets."""
    if real:
        data_root = Path(os.environ["S01509_RAW_ROOT"])
        source = Path(os.environ["S01509_SOURCE_PINE"]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == "887460b8e4bdbaed70b6f7f1a6d3563ba4bd7d1e739d9b279ee5d3349161e258"
        names = [
            f"BTCUSDT-1h-{y}-{m:02}.jsonl.gz"
            for y, months in ((2025, range(4, 13)), (2026, range(1, 5)))
            for m in months
        ]
        for name in names:
            (root / name).write_bytes((data_root / name).read_bytes())
        lo, hi = 1745078400000, 1776614400000
    else:
        source = b"// synthetic signal fixture; not Forven source or a trading result\n"
        names = ["synthetic.jsonl.gz"]
        rows = [
            {"open_time_ms": i * 3600000, **{k: str(close) for k in ("open", "high", "low", "close")}, "volume": "1"}
            for i, close in enumerate([100] * 20 + [110, 100, 90, 100])
        ]
        with gzip.open(root / names[0], "wt") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in rows)
        lo, hi = 20 * 3600000, 22 * 3600000
    (root / "source.pine").write_bytes(source)
    cfg = {
        "family_id": "S01509",
        "input_kind": "raw" if real else "synthetic",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "source": {
            "file": "source.pine",
            "sha256": hashlib.sha256(source).hexdigest(),
            "uri": "https://github.com/judder659/Forven" if real else "synthetic-fixture",
            "ref": "98266984fa85391e58760eb6a1b5e45707504ecd" if real else "synthetic",
        },
        "data": [{"file": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()} for name in names],
        "window": {"start_open_ms": lo, "end_open_ms": hi},
        "entry_length": 20,
        "exit_length": 10,
        "capital": {
            "initial_margin_quote": "1000",
            "leverage": 3,
            "margin_mode": "isolated",
            "margin_fractions": ["0.06", "0.06", "0.06"],
        },
        "dca": {
            "side": "BUY",
            "mode": "MAKER",
            "safety_offsets": ["0", "0.01", "0.02"],
            "take_profit": None,
            "stop_loss": None,
            "trailing_stop": None,
            "trade_cost": 0.0002,
        },
        "engine_image_digest": os.environ.get("S01509_IMAGE_DIGEST", "synthetic-test-no-image-claim"),
        "launch_command": "python -m pytest tests/backtest/test_hummingbot_dca_artifacts.py",
    }
    (root / "config.json").write_text(json.dumps(cfg, sort_keys=True))
    return cfg


def _native_s01509(config_path, uri, experiment_id):
    from qlib.utils import init_instance_by_config

    client = MlflowClient(tracking_uri=uri)
    run = client.create_run(experiment_id)
    recorder = MLflowRecorder(experiment_id, uri, mlflow_run=run)
    try:
        record = init_instance_by_config(
            {
                "class": "S01509DCARecord",
                "module_path": "qlib.contrib.workflow.record_temp",
                "kwargs": {"config_path": str(config_path)},
            },
            recorder=recorder,
        )
        result = record.generate()
    except Exception:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise
    client.set_terminated(run.info.run_id, status="FINISHED")
    return recorder, result, client.get_run(run.info.run_id)


def _s01509_readback_and_replay(root, cfg):
    uri = (root / "mlruns").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("s01509-config-diagnostic")
    recorder, result, run = _native_s01509(root / "config.json", uri, experiment_id)
    prefix = "s01509_diagnostic"
    download = root / "download"
    download.mkdir()
    recovered = Path(client.download_artifacts(run.info.run_id, prefix, dst_path=str(download)))
    assert run.info.status == "FINISHED"
    assert run.data.tags["scope"] == "s01509_bounded_signal_dca_diagnostic"
    assert run.data.tags["input_kind"] == cfg["input_kind"]
    assert run.data.params["config_sha256"] == hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    assert (recovered / "inputs/config.json").read_bytes() == (root / "config.json").read_bytes()
    for item in [cfg["source"], *cfg["data"]]:
        assert hashlib.sha256((recovered / "inputs" / item["file"]).read_bytes()).hexdigest() == item["sha256"]
    metadata = recorder.load_object(f"{prefix}/metadata.pkl")
    assert metadata["config"] == cfg
    assert "account_equity" in metadata["missing"] and "base_fill_journal" in metadata["missing"]
    for name, digest in metadata["source_hashes"].items():
        files = list((recovered / "engine" / name).iterdir())
        assert len(files) == 1 and hashlib.sha256(files[0].read_bytes()).hexdigest() == digest
    for name, value in zip(("signals", "episodes"), result[:2]):
        assert recorder.load_object(f"{prefix}/{name}.pkl") == value
    ledgers = recorder.load_object(f"{prefix}/ledgers.pkl")
    assert len(ledgers) == len(result[2])
    for saved, actual in zip(ledgers, result[2]):
        pd.testing.assert_frame_equal(saved, actual, check_exact=True)
        assert "quantity_base" not in saved and "equity" not in saved
    # Remove the ORIGINAL inputs before replay: no dependence on the raw mount or source path.
    for name in ["config.json", cfg["source"]["file"], *[item["file"] for item in cfg["data"]]]:
        (root / name).unlink()
    _, replay, replay_run = _native_s01509(recovered / "inputs/config.json", uri, experiment_id)
    assert replay[:2] == result[:2] and len(replay[2]) == len(ledgers)
    for expected, actual in zip(ledgers, replay[2]):
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    hashes = {
        str(p.relative_to(recovered)): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "size_bytes": p.stat().st_size,
        }
        for p in recovered.rglob("*")
        if p.is_file()
    }
    (root / "readback.json").write_text(
        json.dumps(
            {
                "run_id": run.info.run_id,
                "replay_run_id": replay_run.info.run_id,
                "tracking_uri": uri,
                "artifact_uri": run.info.artifact_uri,
                "input_kind": cfg["input_kind"],
                "scope": metadata["scope"],
                "params": run.data.params,
                "metrics": run.data.metrics,
                "signal_count": len(result[0]),
                "artifacts": hashes,
                "missing": metadata["missing"],
                "offline_replay_exact": True,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return result, recovered


def test_s01509_config_signals_native_dca_save_and_offline_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    cfg = _s01509_config(tmp_path)
    result, _ = _s01509_readback_and_replay(tmp_path, cfg)
    assert [s["signal"] for s in result[0]] == ["entry", "exit"]
    assert len(result[1]) == 1 and result[1][0]["close_type"] == "TIME_LIMIT"
    assert result[1][0]["entry_open_ms"] == 20 * 3600000
    assert result[1][0]["exit_open_ms"] == 22 * 3600000
    assert list(map(str, result[1][0]["parameters"]["amounts_quote"])) == ["180.00"] * 3


def test_s01509_config_changes_are_consumed_before_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    cfg = _s01509_config(tmp_path)
    uri = (tmp_path / "mlruns").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("s01509-config-control")
    _, baseline, _ = _native_s01509(tmp_path / "config.json", uri, experiment_id)
    cfg["dca"]["trade_cost"] = 0.01  # Synthetic fee control, not a strategy retune.
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    _, altered, run = _native_s01509(tmp_path / "config.json", uri, experiment_id)
    assert run.data.params["trade_cost"] == "0.01"
    assert baseline[:2][0] == altered[:2][0]
    assert not baseline[2][0]["cum_fees_quote"].equals(altered[2][0]["cum_fees_quote"])


@pytest.mark.parametrize(
    "fault",
    [
        "input_hash",
        "source_hash",
        "missing_input",
        "path_escape",
        "unsupported_side",
        "unknown_config",
        "gap",
        "duplicate",
    ],
)
def test_s01509_bad_inputs_fail_before_saving_outputs(tmp_path, monkeypatch, fault):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    cfg = _s01509_config(tmp_path)
    if fault == "input_hash":
        cfg["data"][0]["sha256"] = "0" * 64
    elif fault == "source_hash":
        cfg["source"]["sha256"] = "0" * 64
    elif fault == "missing_input":
        (tmp_path / cfg["data"][0]["file"]).unlink()
    elif fault == "path_escape":
        cfg["source"]["file"] = "../source.pine"
    elif fault == "unsupported_side":
        cfg["dca"]["side"] = "SELL"
    elif fault in ("gap", "duplicate"):
        path = tmp_path / cfg["data"][0]["file"]
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        if fault == "gap":
            del rows[19]  # Missing warmup is not silently filled.
        else:
            rows.append(rows[20])
        with gzip.open(path, "wt") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in rows)
        cfg["data"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        cfg["ignored_setting"] = True
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    uri = (tmp_path / "mlruns").as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("s01509-invalid-config")
    with pytest.raises((ValueError, FileNotFoundError)):
        _native_s01509(tmp_path / "config.json", uri, experiment_id)
    runs = client.search_runs([experiment_id])
    assert len(runs) == 1 and runs[0].info.status == "FAILED"
    assert client.list_artifacts(runs[0].info.run_id) == []


@pytest.mark.parametrize("case", ["zero", "force_flat"])
def test_s01509_zero_signals_and_window_force_flat(tmp_path, monkeypatch, case):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    cfg = _s01509_config(tmp_path)
    closes = [100] * 24 if case == "zero" else [100] * 20 + [110] * 4
    path = tmp_path / cfg["data"][0]["file"]
    with gzip.open(path, "wt") as stream:
        for i, close in enumerate(closes):
            stream.write(
                json.dumps(
                    {
                        "open_time_ms": i * 3600000,
                        **{k: str(close) for k in ("open", "high", "low", "close")},
                        "volume": "1",
                    }
                )
                + "\n"
            )
    cfg["data"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    result, _ = _s01509_readback_and_replay(tmp_path, cfg)
    if case == "zero":
        assert result == ([], [], [])  # Authentic synthetic no-entry control; no fabricated trades/equity.
    else:
        assert len(result[1]) == 1 and result[1][0]["exit_reason"] == "window_end_force_flat"
        assert result[1][0]["exit_open_ms"] == 23 * 3600000
        assert result[1][0]["close_type"] == "TIME_LIMIT"


@pytest.mark.skipif(not os.environ.get("S01509_RAW_ROOT"), reason="Existing raw Klines required; no network download")
def test_s01509_existing_raw_config_diagnostic_and_offline_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    cfg = _s01509_config(tmp_path, real=True)
    result, _ = _s01509_readback_and_replay(tmp_path, cfg)
    assert len(result[1]) == 136 and sum(map(len, result[2])) == 3359
    assert len(result[0]) == 550  # Historical bounded diagnostic, NOT a full-strategy baseline.
    print((tmp_path / "readback.json").read_text())
