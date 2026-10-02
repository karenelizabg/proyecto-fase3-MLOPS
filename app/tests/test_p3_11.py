"""P3-11 (#18): validador de corridas, selección por validación y candado.

Las pruebas puras no tocan MLflow; solo `test_read_runs_roundtrip` usa un MLflow
temporal (SQLite en `tmp_path`), igual patrón que `test_p3_08.py`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pytest

from selection.contracts import (
    Candidate,
    ExperimentsValidity,
    RunSummary,
    Selection,
    SelectionLocked,
)
from selection.lock import (
    load_selection,
    lock_reason,
    require_closed,
)
from selection.lock import (
    test_ids_sha256 as compute_test_ids_sha256,
)
from selection.mlflow_reader import read_runs
from selection.select import rank_runs, select_candidate
from selection.validate import validate_runs

MANIFEST_SHA = "a" * 64
CHECKPOINT_SHA = "b" * 64
CLASSES = {"0": "cat", "1": "dog"}


def a_run(
    run_id: str = "run-1",
    grid_row: str = "r01",
    *,
    status: str = "FINISHED",
    stopped_epoch: float = 5.0,
    git_dirty: str = "false",
    manifest_sha256: str = MANIFEST_SHA,
    classes: dict | None = None,
    params: dict | None = None,
    extra_metrics: dict | None = None,
) -> RunSummary:
    base_params = {
        "optimizer": "adam",
        "batch_size": "32",
        "max_epochs": "15",
        "learning_rate": "0.001",
        "image_size": "128",
        "hidden_layers": "0",
        "dropout": "0.0",
    }
    base_params.update(params or {})
    metrics = {
        "best_val_accuracy": 0.9,
        "best_val_macro_f1": 0.9,
        "best_val_loss": 0.2,
        "best_epoch": 4.0,
        "stopped_epoch": stopped_epoch,
    }
    metrics.update(extra_metrics or {})
    return RunSummary(
        run_id=run_id,
        status=status,
        params=base_params,
        tags={
            "grid_row": grid_row,
            "run_kind": "campaign",
            "manifest_sha256": manifest_sha256,
            "classes": json.dumps(classes or CLASSES),
            "git_dirty": git_dirty,
            "checkpoint_sha256": CHECKPOINT_SHA,
        },
        metrics=metrics,
    )


# --- validación -----------------------------------------------------------------


def test_validate_accepts_a_valid_run():
    result = validate_runs([a_run()], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert result.valid == ["run-1"]
    assert result.invalid == []


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(a_run(status="FAILED"), id="no-finish"),
        pytest.param(a_run(manifest_sha256="c" * 64), id="otro-manifiesto"),
        pytest.param(a_run(classes={"0": "dog", "1": "cat"}), id="otras-clases"),
        pytest.param(a_run(stopped_epoch=1.0), id="una-epoca"),
        pytest.param(a_run(git_dirty="true"), id="dirty"),
    ],
)
def test_validate_rejects_each_reason(run):
    result = validate_runs([run], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert not result.passed
    assert result.valid == []
    assert result.invalid[0].reasons


def test_validate_rejects_duplicated_grid_params():
    runs = [a_run("run-1", "r01"), a_run("run-2", "r02")]  # mismos params de rejilla
    result = validate_runs(runs, manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert sorted(result.valid) == []
    assert all("duplicad" in reason for entry in result.invalid for reason in entry.reasons)


def test_validate_requires_minimum_valid_runs():
    runs = [a_run("run-1", "r01")]
    result = validate_runs(runs, manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=10)
    assert not result.passed
    assert result.valid == ["run-1"]


def test_validate_requires_two_values_per_param():
    runs = [a_run("run-1", "r01", params={"optimizer": "adam"})]
    result = validate_runs(runs, manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    assert not result.passed
    assert result.per_param_values["optimizer"] == 1


def test_validate_passes_full_grid():
    rows = {
        "r01": {
            "optimizer": "adam",
            "batch_size": "32",
            "max_epochs": "15",
            "learning_rate": "0.001",
            "image_size": "128",
            "hidden_layers": "0",
            "dropout": "0.0",
        },
        "r02": {
            "optimizer": "sgd",
            "batch_size": "16",
            "max_epochs": "30",
            "learning_rate": "0.01",
            "image_size": "160",
            "hidden_layers": "1",
            "dropout": "0.3",
        },
    }
    runs = [a_run(f"run-{row}", row, params=params) for row, params in rows.items()]
    result = validate_runs(runs, manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=2)
    assert result.passed
    assert all(count == 2 for count in result.per_param_values.values())


# --- selección ------------------------------------------------------------------


def test_select_rejects_any_test_metric():
    run = a_run(extra_metrics={"test_accuracy": 0.99})
    with pytest.raises(ValueError, match="test_"):
        rank_runs([run])


def test_select_orders_by_accuracy_then_tiebreakers():
    low = a_run("low", "r01", extra_metrics={"best_val_accuracy": 0.8})
    mid = a_run(
        "mid",
        "r02",
        extra_metrics={"best_val_accuracy": 0.9, "best_val_macro_f1": 0.5, "best_val_loss": 0.1},
    )
    high = a_run(
        "high",
        "r03",
        extra_metrics={"best_val_accuracy": 0.9, "best_val_macro_f1": 0.9, "best_val_loss": 0.3},
    )
    ranked = rank_runs([low, mid, high])
    assert [run.run_id for run in ranked] == ["high", "mid", "low"]
    assert select_candidate([low, high, mid]).run_id == "high"


# --- candado --------------------------------------------------------------------


def write_manifest(path: Path, test_ids: list[str]) -> None:
    lines = ["crop_id,split,label,source_image_id,duplicate_group_id"]
    for index, crop_id in enumerate(test_ids):
        lines.append(f"{crop_id},test,0,{index},g{index:06d}")
    lines.append("000001_000001,train,0,99,g000099")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def a_selection(test_ids_sha256: str, release: str = "v0.1.1") -> Selection:
    return Selection(
        run_id="run-1",
        release=release,
        manifest_id="v0.1.1-test",
        manifest_sha256=MANIFEST_SHA,
        checkpoint_sha256=CHECKPOINT_SHA,
        test_ids_sha256=test_ids_sha256,
        selected_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        candidate=Candidate(
            grid_row="r02",
            best_val_accuracy=0.96,
            best_val_macro_f1=0.96,
            best_val_loss=0.12,
        ),
    )


def test_test_ids_sha256_is_deterministic(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["b", "a"])
    first = compute_test_ids_sha256(manifest)
    assert first == compute_test_ids_sha256(manifest)
    assert len(first) == 64


def test_require_closed_refuses_without_selection(tmp_path):
    selection_path = tmp_path / "selection.json"
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a"])
    with pytest.raises(SelectionLocked):
        require_closed(split="test", selection_path=selection_path, manifest_csv=manifest)


def test_require_closed_refuses_on_hash_mismatch(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a"])
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(a_selection("0" * 64).model_dump_json(), encoding="utf-8")
    with pytest.raises(SelectionLocked, match="hash del test"):
        require_closed(split="test", selection_path=selection_path, manifest_csv=manifest)


def test_require_closed_accepts_a_matching_selection(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a", "b"])
    selection = a_selection(compute_test_ids_sha256(manifest))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(selection.model_dump_json(), encoding="utf-8")
    assert (
        require_closed(split="test", selection_path=selection_path, manifest_csv=manifest)
        == selection
    )


def test_require_closed_exempts_validation_split(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a"])
    assert (
        require_closed(
            split="validation", selection_path=tmp_path / "nope.json", manifest_csv=manifest
        )
        is None
    )


def test_lock_reason_reports_not_configured_and_open(tmp_path):
    manifest = tmp_path / "derived" / "manifests" / "v0.1.1" / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    write_manifest(manifest, ["a"])
    selection_path = tmp_path / "reports" / "selection.json"
    assert lock_reason(selection_path=selection_path, derived_dir=tmp_path / "derived")

    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        a_selection(compute_test_ids_sha256(manifest)).model_dump_json(), encoding="utf-8"
    )
    assert lock_reason(selection_path=selection_path, derived_dir=tmp_path / "derived") is None


# --- lectura de MLflow ----------------------------------------------------------


def test_read_runs_roundtrip(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("clasificador-perro-gato")
        with mlflow.start_run():
            mlflow.log_params({"optimizer": "adam"})
            mlflow.set_tags(
                {
                    "run_kind": "campaign",
                    "grid_row": "r01",
                    "manifest_sha256": MANIFEST_SHA,
                    "classes": json.dumps(CLASSES),
                    "git_dirty": "false",
                }
            )
            mlflow.log_metric("best_val_accuracy", 0.9)
            mlflow.log_metric("stopped_epoch", 5)

        runs = read_runs(uri)
        assert len(runs) == 1
        assert runs[0].grid_row == "r01"
        assert runs[0].status == "FINISHED"
        assert runs[0].metrics["best_val_accuracy"] == pytest.approx(0.9)
        assert runs[0].classes == CLASSES
    finally:
        mlflow.set_tracking_uri(previous)


def test_read_runs_filters_by_run_kind(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("clasificador-perro-gato")
        with mlflow.start_run():
            mlflow.set_tags({"run_kind": "smoke"})
        with mlflow.start_run():
            mlflow.set_tags({"run_kind": "campaign", "grid_row": "r01"})
        assert len(read_runs(uri, run_kind="campaign")) == 1
        assert len(read_runs(uri, run_kind="smoke")) == 1
    finally:
        mlflow.set_tracking_uri(previous)


def test_experiments_validity_contract_roundtrip():
    result = validate_runs([a_run()], manifest_sha256=MANIFEST_SHA, classes=CLASSES, min_required=1)
    again = ExperimentsValidity.model_validate_json(result.model_dump_json())
    assert again == result


def test_load_selection_validates_contract(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, ["a"])
    path = tmp_path / "selection.json"
    path.write_text(
        a_selection(compute_test_ids_sha256(manifest)).model_dump_json(), encoding="utf-8"
    )
    loaded = load_selection(path)
    assert loaded.selection_metric == "best_val_accuracy"
    assert loaded.candidate.grid_row == "r02"
