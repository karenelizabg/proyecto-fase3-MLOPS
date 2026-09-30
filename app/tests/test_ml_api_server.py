"""`ml_api.server` (P3-03): /health, /training/jobs (real) y los 4 endpoints pendientes.

`list_jobs` se inyecta -- igual que `create_message` en `copilot/server.py` --
así esta prueba nunca toca una MariaDB real; eso lo cubre
`test_ml_api_repository.py` con datos tomados de una corrida real.
"""

from datetime import datetime

from starlette.testclient import TestClient

from ml_api.contracts import TrainingJob
from ml_api.server import create_app
from tests._mcp_fixtures import mcp_settings


def client_for(monkeypatch, tmp_path, *, jobs: list[TrainingJob] | None = None) -> TestClient:
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    return TestClient(create_app(settings, list_jobs=lambda: jobs or []))


def a_job(**overrides) -> TrainingJob:
    defaults = {
        "id": "r01",
        "status": "running",
        "progress": 0.4,
        "config": {"optimizer": "adam"},
        "dataset_release": "v0.1.1",
        "manifest_id": "manifest-abc",
        "mlflow_run_id": None,
        "error": None,
        "logs": ["epoch 1 done"],
        "heartbeat_at": datetime(2026, 9, 29, 21, 23, 26),
    }
    return TrainingJob(**{**defaults, **overrides})


def test_health_says_ok(monkeypatch, tmp_path):
    response = client_for(monkeypatch, tmp_path).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_training_jobs_returns_the_injected_jobs(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, jobs=[a_job(), a_job(id="r02", status="queued")])
    response = client.get("/training/jobs")

    assert response.status_code == 200
    body = response.json()
    assert [job["id"] for job in body["jobs"]] == ["r01", "r02"]
    assert body["jobs"][0]["config"] == {"optimizer": "adam"}


def test_training_jobs_is_an_empty_list_before_any_run(monkeypatch, tmp_path):
    response = client_for(monkeypatch, tmp_path).get("/training/jobs")

    assert response.status_code == 200
    assert response.json() == {"jobs": []}


def test_the_four_pending_endpoints_name_their_own_ticket(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path)
    expected_tickets = {
        "/experiments": "P3-12",
        "/evaluation": "P3-13",
        "/models": "P3-14",
        "/inference": "P3-16",
    }

    for path, ticket in expected_tickets.items():
        response = client.get(path)
        assert response.status_code == 200, path
        body = response.json()
        assert body["status"] == "pending"
        assert body["ticket"] == ticket
