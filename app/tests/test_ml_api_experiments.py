"""`GET /experiments` (P3-12/P3-15): la UI descubre el experimento por nombre
en vez de fijar el `experiment_id` a mano en `Experiments.tsx`.

Se inyecta un `MlflowClient` falso (mismo criterio que `list_jobs`): no se toca
un MLflow real.
"""

from starlette.testclient import TestClient

from ml_api import experiments
from ml_api.server import create_app
from tests._mcp_fixtures import mcp_settings


class _FakeExperiment:
    def __init__(self, experiment_id: str, name: str) -> None:
        self.experiment_id = experiment_id
        self.name = name
        self.lifecycle_stage = "active"


class _FakeClient:
    def search_experiments(self):
        return [_FakeExperiment("0", "Default"), _FakeExperiment("1", "clasificador-perro-gato")]


def _client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setattr(experiments, "get_client", lambda: _FakeClient())
    settings = mcp_settings(
        monkeypatch, tmp_path / "dataset", tmp_path / "reports", tmp_path / "derived"
    )
    return TestClient(create_app(settings))


def test_list_experiments_returns_id_and_name(monkeypatch, tmp_path):
    response = _client(monkeypatch, tmp_path).get("/experiments")

    assert response.status_code == 200
    assert response.json() == [
        {"experiment_id": "0", "name": "Default", "lifecycle_stage": "active"},
        {"experiment_id": "1", "name": "clasificador-perro-gato", "lifecycle_stage": "active"},
    ]


def test_list_experiments_is_502_when_mlflow_fails(monkeypatch, tmp_path):
    class _BrokenClient:
        def search_experiments(self):
            raise RuntimeError("MLflow caído")

    monkeypatch.setattr(experiments, "get_client", lambda: _BrokenClient())
    settings = mcp_settings(
        monkeypatch, tmp_path / "dataset", tmp_path / "reports", tmp_path / "derived"
    )
    response = TestClient(create_app(settings)).get("/experiments")

    assert response.status_code == 502
