"""Offline persistence contract example, not a deployed Runner or fill exporter."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from qlib.backtest import backtest_hummingbot_dca
from qlib.workflow.recorder import MLflowRecorder


REQUIRED = ("ledger.pkl", "metadata.json")


def _verify_download(root, expected_manifest):
    # Fixed two-artifact demo: the expected manifest is held by the caller,
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
    run = client.create_run(experiment_id)
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
    metadata = {"fixture": "synthetic", "close_type": close_type, "parameters": parameters}
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


@pytest.mark.parametrize("fault", ["missing", "truncate", "wrong_hash", "missing_manifest_entry", "wrong_run"])
def test_missing_or_corrupt_artifacts_cannot_pass_acceptance(saved_dca, fault):
    _, _, _, root, manifest = saved_dca
    if fault == "missing":
        (root / "ledger.pkl").unlink()
    elif fault == "truncate":
        payload = (root / "ledger.pkl").read_bytes()
        (root / "ledger.pkl").write_bytes(payload[:-1])
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
