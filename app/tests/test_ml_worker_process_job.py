"""`ml_worker.__main__.process_one_job` (P3-09).

`spawn` se inyecta -- igual que `list_jobs`/`create_job` en `ml_api.server`
-- así esta prueba nunca lanza un subproceso ni toca una MariaDB real: solo
prueba que, dada una secuencia de mensajes de la cola, `process_one_job`
llama a la función correcta de `ml_worker.repository` con los argumentos
correctos, en el orden correcto. Las funciones de `ml_worker.repository`
se parchan directo (son las que sí tocan la base).
"""

from datetime import datetime
from queue import Empty

import pytest

from ml_api.contracts import TrainingJob
from ml_worker.__main__ import process_one_job


def a_job(**overrides) -> TrainingJob:
    defaults = {
        "id": "job-1",
        "status": "running",
        "progress": 0.0,
        "config": {"optimizer": "adam"},
        "dataset_release": "v0.1.1",
        "manifest_id": "manifest-abc",
        "mlflow_run_id": None,
        "error": None,
        "logs": [],
        "heartbeat_at": datetime(2026, 9, 30, 12, 0, 0),
    }
    return TrainingJob(**{**defaults, **overrides})


class FakeQueue:
    def __init__(self, messages):
        self._messages = list(messages)

    def get(self, timeout):
        if not self._messages:
            raise Empty
        return self._messages.pop(0)


class FakeProcess:
    def __init__(self, *, dies_silently=False, exitcode=1):
        self.joined = False
        self.exitcode = exitcode
        self._dies_silently = dies_silently

    def is_alive(self):
        return not self._dies_silently

    def join(self):
        self.joined = True


@pytest.fixture
def recorder(monkeypatch):
    calls = []
    for name in ("update_progress", "mark_completed", "mark_failed"):

        def make(name):
            def fn(engine, job_id, **kwargs):
                calls.append((name, job_id, kwargs))

            return fn

        monkeypatch.setattr(f"ml_worker.__main__.{name}", make(name))
    return calls


def test_applies_each_progress_message_and_then_completed(recorder, tmp_path):
    job = a_job()
    messages = [
        {"type": "progress", "progress": 0.5, "log_line": "época 1/2"},
        {"type": "progress", "progress": 1.0, "log_line": "época 2/2"},
        {"type": "completed", "mlflow_run_id": "run-42"},
    ]

    def spawn(_job, _manifest_path, _crops_root):
        return FakeProcess(), FakeQueue(messages)

    process_one_job(engine=object(), job=job, derived_dir=tmp_path, spawn=spawn)

    assert recorder == [
        ("update_progress", "job-1", {"progress": 0.5, "log_line": "época 1/2"}),
        ("update_progress", "job-1", {"progress": 1.0, "log_line": "época 2/2"}),
        ("mark_completed", "job-1", {"mlflow_run_id": "run-42"}),
    ]


def test_applies_failed_message_and_stops(recorder, tmp_path):
    job = a_job()
    messages = [
        {"type": "progress", "progress": 0.3, "log_line": "época 1/3"},
        {"type": "failed", "error": "CUDA out of memory"},
    ]

    def spawn(_job, _manifest_path, _crops_root):
        return FakeProcess(), FakeQueue(messages)

    process_one_job(engine=object(), job=job, derived_dir=tmp_path, spawn=spawn)

    assert recorder == [
        ("update_progress", "job-1", {"progress": 0.3, "log_line": "época 1/3"}),
        ("mark_failed", "job-1", {"error": "CUDA out of memory"}),
    ]


def test_marks_failed_when_the_subprocess_dies_without_a_final_message(recorder, tmp_path):
    job = a_job()

    def spawn(_job, _manifest_path, _crops_root):
        return FakeProcess(dies_silently=True, exitcode=137), FakeQueue([])

    process_one_job(engine=object(), job=job, derived_dir=tmp_path, spawn=spawn)

    assert len(recorder) == 1
    name, job_id, kwargs = recorder[0]
    assert (name, job_id) == ("mark_failed", "job-1")
    assert "137" in kwargs["error"]


def test_spawn_receives_the_dataset_paths_built_from_derived_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_worker.__main__.update_progress", lambda *a, **k: None)
    monkeypatch.setattr("ml_worker.__main__.mark_completed", lambda *a, **k: None)
    job = a_job(dataset_release="v0.1.1")
    received = {}

    def spawn(_job, manifest_path, crops_root):
        received["manifest_path"] = manifest_path
        received["crops_root"] = crops_root
        return FakeProcess(), FakeQueue([{"type": "completed", "mlflow_run_id": "run-1"}])

    process_one_job(engine=object(), job=job, derived_dir=tmp_path, spawn=spawn)

    assert received["manifest_path"] == tmp_path / "manifests" / "v0.1.1" / "manifest.csv"
    assert received["crops_root"] == tmp_path / "crops"
