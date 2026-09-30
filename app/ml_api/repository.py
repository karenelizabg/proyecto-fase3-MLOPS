"""Lectura de `training_jobs` (P3-03): SQL crudo, no un ORM duplicado.

La tabla la migra y posee el backend (Drizzle, `schema.ts`) -- `ml-api` solo
lee. Nombres de columna entre backticks a propósito: `dataset_release` se
renombró para evitar la palabra reservada `release`, pero cualquier otra
columna futura corre el mismo riesgo, y un SQL crudo no se defiende solo.
"""

import json
from collections.abc import Sequence

from sqlalchemy import Engine, text

from ml_api.contracts import TrainingJob

_SELECT_COLUMNS = """
    `id`, `status`, `progress`, `config`, `dataset_release`, `manifest_id`,
    `mlflow_run_id`, `error`, `logs`, `heartbeat_at`
"""


def _row_to_job(row: Sequence) -> TrainingJob:
    (
        id_,
        status,
        progress,
        config,
        dataset_release,
        manifest_id,
        mlflow_run_id,
        error,
        logs,
        heartbeat_at,
    ) = row
    return TrainingJob(
        id=id_,
        status=status,
        progress=progress,
        # El driver (PyMySQL) entrega las columnas JSON como texto crudo, no
        # como dict/list ya parseado.
        config=json.loads(config),
        dataset_release=dataset_release,
        manifest_id=manifest_id,
        mlflow_run_id=mlflow_run_id,
        error=error,
        logs=json.loads(logs),
        heartbeat_at=heartbeat_at,
    )


def list_training_jobs(engine: Engine) -> list[TrainingJob]:
    query = text(f"SELECT {_SELECT_COLUMNS} FROM `training_jobs` ORDER BY `updated_at` DESC")
    with engine.connect() as connection:
        rows = connection.execute(query).fetchall()
    return [_row_to_job(row) for row in rows]
