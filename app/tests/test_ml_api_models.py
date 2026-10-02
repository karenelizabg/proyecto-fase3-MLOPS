"""`/models` (P3-15): catálogo de P3-14 + estado S3 en vivo + versión activa.

El catálogo real (`reports/models/registry.json`) lo produce P3-14, que todavía
no existe. Estas pruebas usan un `registry.json` de datos de prueba en `tmp_path`
y un `head-object` falso inyectado -- sin tocar MinIO/S3 (mismo criterio que
`list_jobs`/`create_job` en `test_ml_api_server.py`).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from starlette.testclient import TestClient

from ml_api.server import create_app
from storage.object_store import object_status, presigned_get_url
from tests._mcp_fixtures import mcp_settings

LAST_MODIFIED = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def a_registry() -> dict:
    return {
        "schema_version": "1.0",
        "model_name": "clasificador-perro-gato",
        "versions": [
            {
                "version": "0.1.0",
                "dataset_version": "v0.1.0",
                "run_id": "smoke-run",
                "release": "v0.1.0",
                "manifest_id": "v0.1.0-aaaa",
                "manifest_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "package_sha256": "c" * 64,
                "s3_bucket": "mlops-p3-models-222629887955",
                "s3_key": "models/clasificador-perro-gato/0.1.0/",
                "s3_version_id": "v-old",
                "published_at": "2026-09-29T10:00:00Z",
                "selected": False,
                "card": "Ensayo, no seleccionada.",
            },
            {
                "version": "1.0.0",
                "dataset_version": "v0.1.1",
                "run_id": "7e7b4a4b35464cfebb6b41714a3ad931",
                "release": "v0.1.1",
                "manifest_id": "v0.1.1-53fc84fdaa07",
                "manifest_sha256": "d" * 64,
                "checkpoint_sha256": "e" * 64,
                "package_sha256": "f" * 64,
                "s3_bucket": "mlops-p3-models-222629887955",
                "s3_key": "models/clasificador-perro-gato/1.0.0/",
                "s3_version_id": "v-final",
                "published_at": "2026-09-30T18:00:00Z",
                "selected": True,
                "card": "Modelo final, candidato r02.",
            },
        ],
    }


def write_registry(reports_dir: Path, registry: dict) -> None:
    models_dir = reports_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "registry.json").write_text(json.dumps(registry), encoding="utf-8")


def client_for(monkeypatch, tmp_path, *, existing: set[str] | None = None) -> TestClient:
    reports_dir = tmp_path / "reports"
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir, tmp_path / "derived")
    existing = existing if existing is not None else {"0.1.0", "1.0.0"}

    def status_of(entry):
        if entry.version not in existing:
            return None
        return {
            "version_id": entry.s3_version_id,
            "size_bytes": 1024,
            "last_modified": LAST_MODIFIED,
        }

    return TestClient(
        create_app(
            settings,
            model_status_of=status_of,
            model_download_url_of=lambda entry: f"https://signed.example/{entry.s3_key}",
        )
    )


def test_models_is_pending_until_p3_14_publishes_the_registry(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["ticket"] == "P3-14"


def test_models_lists_versions_with_live_s3_status_and_dataset_version(monkeypatch, tmp_path):
    write_registry(tmp_path / "reports", a_registry())
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/models")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_name"] == "clasificador-perro-gato"
    assert body["active_version"] is None
    first = body["versions"][0]
    assert first["version"] == "0.1.0"
    assert first["dataset_version"] == "v0.1.0"
    assert first["s3_status"]["exists"] is True
    assert first["s3_status"]["version_id"] == "v-old"


def test_models_marks_a_missing_object_as_not_existing(monkeypatch, tmp_path):
    write_registry(tmp_path / "reports", a_registry())
    client = client_for(monkeypatch, tmp_path, existing={"1.0.0"})

    body = client.get("/models").json()

    statuses = {entry["version"]: entry["s3_status"]["exists"] for entry in body["versions"]}
    assert statuses == {"0.1.0": False, "1.0.0": True}


def test_model_detail_carries_provenance_and_presigned_url(monkeypatch, tmp_path):
    write_registry(tmp_path / "reports", a_registry())
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/models/1.0.0")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "1.0.0"
    assert body["dataset_version"] == "v0.1.1"
    assert body["run_id"] == "7e7b4a4b35464cfebb6b41714a3ad931"
    assert body["card"] == "Modelo final, candidato r02."
    assert body["download_url"] == ("https://signed.example/models/clasificador-perro-gato/1.0.0/")


def test_model_detail_is_404_for_an_unknown_version(monkeypatch, tmp_path):
    write_registry(tmp_path / "reports", a_registry())
    client = client_for(monkeypatch, tmp_path)

    response = client.get("/models/9.9.9")

    assert response.status_code == 404


def test_setting_active_version_persists_and_rejects_a_missing_object(monkeypatch, tmp_path):
    write_registry(tmp_path / "reports", a_registry())
    client = client_for(monkeypatch, tmp_path, existing={"1.0.0"})
    active_path = tmp_path / "reports" / "models" / "active_version.json"

    rejected = client.post("/models/active", json={"version": "0.1.0"})
    assert rejected.status_code == 400
    assert "no existe en S3" in rejected.json()["error"]
    assert not active_path.exists()

    accepted = client.post("/models/active", json={"version": "1.0.0"})
    assert accepted.status_code == 200
    assert accepted.json()["active"] is True
    assert json.loads(active_path.read_text(encoding="utf-8")) == {"active_version": "1.0.0"}

    assert client.get("/models").json()["active_version"] == "1.0.0"


def test_setting_active_version_rejects_an_unknown_version_and_bad_body(monkeypatch, tmp_path):
    write_registry(tmp_path / "reports", a_registry())
    client = client_for(monkeypatch, tmp_path)

    assert client.post("/models/active", json={"version": "9.9.9"}).status_code == 404
    assert client.post("/models/active", json={}).status_code == 400


# --- object_store (P3-15) ----------------------------------------------------


class _FakeStat:
    version_id = "v-1"
    size = 2048
    last_modified = LAST_MODIFIED


class _FakeMinio:
    def stat_object(self, bucket, key):
        return _FakeStat()

    def presigned_get_object(self, bucket, key, expires=None):
        return f"https://signed/{bucket}/{key}?expires={expires.total_seconds()}"


def test_object_status_reads_head_object():
    status = object_status(_FakeMinio(), bucket="b", key="k")

    assert status == {"version_id": "v-1", "size_bytes": 2048, "last_modified": LAST_MODIFIED}


def test_object_status_returns_none_for_a_missing_key():
    from minio.error import S3Error

    class _Missing(_FakeMinio):
        def stat_object(self, bucket, key):
            error = S3Error.__new__(S3Error)
            error.code = "NoSuchKey"
            raise error

    assert object_status(_Missing(), bucket="b", key="k") is None


def test_presigned_get_url_signs_the_object():
    url = presigned_get_url(_FakeMinio(), bucket="b", key="k", expires_seconds=60)

    assert url == "https://signed/b/k?expires=60.0"
