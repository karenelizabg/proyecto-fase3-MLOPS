"""`trainer-worker` (P3-09): toma trabajos reales de `training_jobs`.

Bucle: reclama trabajos con heartbeat vencido como `failed` (el worker
anterior murió a medio entrenamiento), reclama el siguiente `queued` con
bloqueo atómico, lo corre en un subproceso aparte (`multiprocessing.Process`
-- si el entrenamiento se cuelga o revienta, este proceso sigue vivo y
puede reaccionar, no se cae con él), y va aplicando su progreso a la base
mientras corre.
"""

import logging
import multiprocessing
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty
from typing import Any

import torch
from sqlalchemy import Engine, text

from ml_api.contracts import TrainingJob
from ml_worker.repository import (
    claim_next_queued_job,
    mark_completed,
    mark_failed,
    reclaim_stale_jobs,
    update_progress,
)
from ml_worker.run_training import dataset_paths, manifest_meta_path, run_training_subprocess
from storage.db import get_engine
from storage.settings import Settings

logger = logging.getLogger("trainer-worker")

POLL_INTERVAL_SECONDS = 5.0
QUEUE_POLL_TIMEOUT_SECONDS = 2.0
# Cuánto puede callar el heartbeat de un trabajo `running` antes de asumir
# que el worker que lo tenía murió sin avisar. Una época real puede tardar
# más que esto con un dataset grande -- se documenta como simplificación
# conocida del worker único de hoy, no un límite pensado para varios workers.
HEARTBEAT_STALE_SECONDS = 600

SpawnFn = Callable[[TrainingJob, Path, Path, Path], tuple[Any, Any]]


def _spawn_training_process(
    job: TrainingJob, manifest_path: Path, crops_root: Path, meta_path: Path
) -> tuple[multiprocessing.Process, multiprocessing.Queue]:
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=run_training_subprocess,
        # `model_dump()` corriente, no `mode="json"`: `TrainingJob` es `strict=True`,
        # así que `heartbeat_at` tiene que seguir siendo un `datetime` real para que
        # `TrainingJob(**job_payload)` lo acepte del otro lado -- pickle (lo que usa
        # `multiprocessing` para pasar los argumentos) ya sabe serializar `datetime`.
        args=(job.model_dump(), str(manifest_path), str(crops_root), str(meta_path), queue),
    )
    process.start()
    return process, queue


def process_one_job(
    engine: Engine,
    job: TrainingJob,
    *,
    derived_dir: Path,
    reports_dir: Path,
    spawn: SpawnFn = _spawn_training_process,
) -> None:
    """Corre `job` hasta terminar (completed/failed) y deja la base al día."""
    manifest_path, crops_root = dataset_paths(derived_dir, job.dataset_release)
    meta_path = manifest_meta_path(reports_dir, job.dataset_release)
    process, queue = spawn(job, manifest_path, crops_root, meta_path)

    finished = False
    try:
        while True:
            try:
                message = queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except Empty:
                if not process.is_alive():
                    break
                continue

            if message["type"] == "progress":
                update_progress(
                    engine, job.id, progress=message["progress"], log_line=message["log_line"]
                )
                logger.info("%s: %s", job.id, message["log_line"])
            elif message["type"] == "completed":
                mark_completed(engine, job.id, mlflow_run_id=message["mlflow_run_id"])
                logger.info("%s: completado (mlflow_run_id=%s).", job.id, message["mlflow_run_id"])
                finished = True
                break
            else:  # "failed"
                mark_failed(engine, job.id, error=message["error"])
                logger.warning("%s: falló (%s).", job.id, message["error"])
                finished = True
                break
    finally:
        process.join()

    if not finished:
        # El subproceso terminó (se cayó, lo mataron) sin mandar un mensaje
        # final -- el padre no se queda esperando para siempre, lo cierra él.
        error = f"el proceso de entrenamiento terminó inesperadamente (código {process.exitcode})"
        mark_failed(engine, job.id, error=error)
        logger.error("%s: %s", job.id, error)


def _wait_for_dependencies(retries: int = 10, delay_seconds: float = 3.0) -> None:
    engine = get_engine()
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info("Conectado a MariaDB.")
            return
        except Exception as error:  # reintento deliberado en el arranque
            logger.warning("Intento %s/%s: MariaDB aún no lista (%s).", attempt, retries, error)
            time.sleep(delay_seconds)
    raise RuntimeError("No se pudo conectar a MariaDB tras varios intentos.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    logger.info(
        "trainer-worker: arrancando (MLflow en %s, torch %s, cuda=%s).",
        settings.mlflow_tracking_uri,
        torch.__version__,
        torch.cuda.is_available(),
    )
    _wait_for_dependencies()
    engine = get_engine()

    logger.info("trainer-worker listo, esperando training_jobs en 'queued'.")
    while True:
        reclaimed = reclaim_stale_jobs(engine, stale_after_seconds=HEARTBEAT_STALE_SECONDS)
        if reclaimed:
            logger.warning("%s trabajo(s) con heartbeat vencido marcados como failed.", reclaimed)

        job = claim_next_queued_job(engine)
        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        logger.info("Reclamado %s (release=%s).", job.id, job.dataset_release)
        process_one_job(
            engine, job, derived_dir=settings.derived_dir, reports_dir=settings.reports_dir
        )


if __name__ == "__main__":
    main()
