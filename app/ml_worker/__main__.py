"""`trainer-worker` (P3-03): placeholder de infraestructura.

El bucle real -- tomar un `TrainingJob` en `queued`, entrenar, escribir
progreso/heartbeat, registrar en MLflow -- lo construye P3-09. Este ticket
(P3-03) solo prueba que el servicio existe, arranca con el grupo `ml`
(torch/torchvision/mlflow importables), y se conecta a MariaDB/MinIO -- para
que P3-09 tenga algo real sobre lo que construir, no un contenedor que no
levanta.

Se queda vivo (no termina en 0) para que `docker compose up` lo muestre
"Up", igual que `presentation/main.py`.
"""

import logging
import time

import torch
from sqlalchemy import text

from storage.db import get_engine
from storage.settings import Settings

logger = logging.getLogger("trainer-worker")


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
        "trainer-worker: arrancando (placeholder P3-03, MLflow en %s).",
        settings.mlflow_tracking_uri,
    )
    _wait_for_dependencies()

    logger.info(
        "torch %s importado correctamente (CPU, cuda=%s).",
        torch.__version__,
        torch.cuda.is_available(),
    )
    logger.info("Esperando el bucle de entrenamiento real (P3-09).")

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
