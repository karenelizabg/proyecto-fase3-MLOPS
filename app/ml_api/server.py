"""`ml-api` (P3-03): endpoints HTTP para las 5 pantallas de "Modelo".

Solo `/training/jobs` tiene datos reales hoy (lee `training_jobs`, que este
mismo ticket migra). Los otros cuatro responden `PendingEndpoint`: existen
como contrato -- para que el frontend los pueda consumir ya -- pero sin
datos reales todavía, porque eso lo construyen P3-12 (Experiments), P3-13
(Evaluation), P3-14 (Models) y P3-15/P3-16 (Inference).

Mismo patrón que `copilot/server.py`: `create_app(settings)` para pruebas,
`main()` para producción.
"""

import logging
from collections.abc import Callable

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ml_api.contracts import PendingEndpoint, TrainingJob, TrainingJobList
from ml_api.repository import list_training_jobs
from storage.db import get_engine
from storage.settings import Settings

logger = logging.getLogger("ml-api")

ListJobs = Callable[[], list[TrainingJob]]

_PENDING = {
    "experiments": PendingEndpoint(
        ticket="P3-12", message="Experiments se conecta a MLflow en vivo en P3-12."
    ),
    "evaluation": PendingEndpoint(
        ticket="P3-13", message="Evaluation se llena con la evaluación final en P3-13."
    ),
    "models": PendingEndpoint(
        ticket="P3-14", message="Models se llena al publicar el paquete del modelo en P3-14."
    ),
    "inference": PendingEndpoint(
        ticket="P3-16", message="Inference se conecta a la inferencia real en P3-16."
    ),
}


def create_app(settings: Settings | None = None, *, list_jobs: ListJobs | None = None) -> Starlette:
    settings = settings if settings is not None else Settings()
    list_jobs = list_jobs or (lambda: list_training_jobs(get_engine()))

    def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    def training_jobs(_: Request) -> JSONResponse:
        jobs = list_jobs()
        return JSONResponse(TrainingJobList(jobs=jobs).model_dump(mode="json"))

    def pending(name: str):
        async def handler(_: Request) -> JSONResponse:
            return JSONResponse(_PENDING[name].model_dump(mode="json"))

        return handler

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/training/jobs", training_jobs, methods=["GET"]),
            Route("/experiments", pending("experiments"), methods=["GET"]),
            Route("/evaluation", pending("evaluation"), methods=["GET"]),
            Route("/models", pending("models"), methods=["GET"]),
            Route("/inference", pending("inference"), methods=["GET"]),
        ]
    )


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=settings.ml_api_host, port=settings.ml_api_port)


if __name__ == "__main__":
    main()
