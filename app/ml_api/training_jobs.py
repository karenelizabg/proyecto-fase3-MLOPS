"""Valida una solicitud de `training_job` nueva, antes de crear ninguna fila (P3-09).

No confía en que quien construyó el manifiesto (P3-06, `dvc_manifest_stage.py`)
ya garantizó que el release pasó la compuerta y que el manifiesto no tiene
fuga -- los vuelve a comprobar aquí, sobre los archivos reales, porque esta
es la última puerta antes de lanzar un entrenamiento real. Mismo criterio que
`presentation/release.cut_release` y `dvc_manifest_stage.quality_reference`,
que también revisan `status == "failed"` cada una por su cuenta en vez de
asumir que la otra ya lo hizo.
"""

import json
from pathlib import Path

from pydantic import ValidationError

from manifest.check_leakage import find_leakage, read_manifest
from training.config import TrainingConfig


class TrainingJobRejected(Exception):
    """La solicitud se rechaza antes de crear ninguna fila en `training_jobs`."""


def validate_new_training_job(
    *,
    dataset_release: str,
    config: dict,
    reports_dir: Path,
    derived_dir: Path,
) -> tuple[TrainingConfig, str]:
    """Valida `config` y el release; devuelve (config validado, manifest_id).

    Levanta `TrainingJobRejected` si `config` no cumple `TrainingConfig`, si no
    hay un manifiesto para `dataset_release`, si su compuerta de calidad está
    `failed`, o si el manifiesto tiene fuga entre particiones.
    """
    try:
        validated_config = TrainingConfig.model_validate(config)
    except ValidationError as error:
        raise TrainingJobRejected(f"config inválido: {error}") from error

    meta_path = reports_dir / "manifests" / dataset_release / "manifest_meta.json"
    if not meta_path.exists():
        raise TrainingJobRejected(f"no existe un manifiesto para el release {dataset_release!r}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    status = meta["quality_reference"]["status"]
    if status == "failed":
        raise TrainingJobRejected(
            f"el release {dataset_release!r} está failed en la compuerta de calidad"
        )

    manifest_csv = derived_dir / "manifests" / dataset_release / "manifest.csv"
    if not manifest_csv.exists():
        raise TrainingJobRejected(f"no existe el manifiesto {manifest_csv}")
    leakage = find_leakage(read_manifest(manifest_csv))
    if any(leakage.values()):
        raise TrainingJobRejected(f"el manifiesto de {dataset_release!r} tiene fuga: {leakage}")

    return validated_config, meta["manifest_id"]
