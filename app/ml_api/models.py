"""Lectura y estado de los modelos publicados (P3-15).

El catálogo (`reports/models/registry.json`) lo produce P3-14; aquí solo se lee
y se le agrega el estado **en vivo** de S3 (`head-object`), la URL prefirmada de
descarga y la elección de la versión activa para inferencia.

La dependencia de S3 entra como `status_of`/`download_url_of`, no como una
llamada directa: así esta capa es pura y las pruebas no tocan una red (mismo
criterio que `create_app` inyectando `list_jobs`).
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

from ml_api.contracts import (
    ModelDetail,
    ModelEntry,
    ModelList,
    ModelRegistry,
    ModelS3Status,
    ModelSummary,
)


class ModelVersionNotFound(LookupError):
    """La versión no está en `registry.json`."""


class ModelObjectMissing(RuntimeError):
    """Se pidió marcar/publicar una versión cuyo objeto no existe en S3."""


StatusOf = Callable[[ModelEntry], dict | None]
DownloadUrlOf = Callable[[ModelEntry], str | None]


def load_registry(path: Path) -> ModelRegistry | None:
    """`None` mientras P3-14 no haya publicado el catálogo."""
    if not path.is_file():
        return None
    return ModelRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def read_active_version(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    version = payload.get("active_version")
    return version if isinstance(version, str) and version else None


def write_active_version(path: Path, version: str) -> None:
    """Escritura atómica: nunca deja el archivo a medias entre dos requests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"active_version": version}), encoding="utf-8")
    os.replace(tmp, path)


def _status(raw: dict | None) -> ModelS3Status:
    if raw is None:
        return ModelS3Status(exists=False)
    return ModelS3Status(
        exists=True,
        version_id=raw.get("version_id"),
        size_bytes=raw.get("size_bytes"),
        last_modified=raw.get("last_modified"),
    )


def summarize(
    entry: ModelEntry, *, active_version: str | None, status: ModelS3Status
) -> ModelSummary:
    return ModelSummary(
        version=entry.version,
        dataset_version=entry.dataset_version,
        run_id=entry.run_id,
        release=entry.release,
        manifest_id=entry.manifest_id,
        published_at=entry.published_at,
        selected=entry.selected,
        active=entry.version == active_version,
        s3_status=status,
    )


def build_list(
    registry: ModelRegistry, *, active_version: str | None, status_of: StatusOf
) -> ModelList:
    versions = [
        summarize(entry, active_version=active_version, status=_status(status_of(entry)))
        for entry in registry.versions
    ]
    return ModelList(
        model_name=registry.model_name, active_version=active_version, versions=versions
    )


def find_entry(registry: ModelRegistry, version: str) -> ModelEntry:
    for entry in registry.versions:
        if entry.version == version:
            return entry
    raise ModelVersionNotFound(version)


def build_detail(
    registry: ModelRegistry,
    version: str,
    *,
    active_version: str | None,
    status_of: StatusOf,
    download_url_of: DownloadUrlOf,
) -> ModelDetail:
    entry = find_entry(registry, version)
    raw_status = status_of(entry)
    return ModelDetail(
        model_name=registry.model_name,
        version=entry.version,
        dataset_version=entry.dataset_version,
        run_id=entry.run_id,
        release=entry.release,
        manifest_id=entry.manifest_id,
        manifest_sha256=entry.manifest_sha256,
        checkpoint_sha256=entry.checkpoint_sha256,
        package_sha256=entry.package_sha256,
        s3_bucket=entry.s3_bucket,
        s3_key=entry.s3_key,
        s3_version_id=entry.s3_version_id,
        published_at=entry.published_at,
        selected=entry.selected,
        active=entry.version == active_version,
        card=entry.card,
        download_url=download_url_of(entry) if raw_status is not None else None,
        s3_status=_status(raw_status),
    )


def activate_version(registry: ModelRegistry, version: str, *, status_of: StatusOf) -> ModelEntry:
    """Marca `version` como activa **solo si su objeto existe en S3**.

    El rechazo explícito de un objeto inexistente es parte del #23: no se puede
    dejar el sistema apuntando a un paquete que no está.
    """
    entry = find_entry(registry, version)
    if status_of(entry) is None:
        raise ModelObjectMissing(version)
    return entry
