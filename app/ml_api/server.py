"""`ml-api` (P3-03): endpoints HTTP para las 5 pantallas de "Modelo".

`/training/jobs` (P3-09) y `/evaluation` + `/crops` (P3-15) tienen datos reales.
Experiments (P3-12) lee MLflow; Models (P3-14) e Inference (P3-16) siguen
respondiendo `PendingEndpoint`: existen como contrato -- para que el frontend
los pueda consumir ya -- pero sin datos reales todavía.

Mismo patrón que `copilot/server.py`: `create_app(settings)` para pruebas,
`main()` para producción.
"""

import logging
import re
from collections.abc import Callable

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from ml_api.contracts import (
    EvaluationLocked,
    ModelEntry,
    PendingEndpoint,
    TrainingJob,
    TrainingJobList,
)
from ml_api.evaluation import build_evaluation_report
from ml_api.experiments import get_metric_history, list_runs, update_run_tag
from ml_api.models import (
    ModelObjectMissing,
    ModelVersionNotFound,
    activate_version,
    build_detail,
    build_list,
    load_registry,
    read_active_version,
    write_active_version,
)
from ml_api.repository import (
    TrainingJobNotCancellable,
    create_training_job,
    list_training_jobs,
    request_training_job_cancellation,
)
from ml_api.training_jobs import TrainingJobRejected, validate_new_training_job
from selection.lock import lock_reason
from storage.db import get_engine
from storage.object_store import get_minio_client, object_status, presigned_get_url
from storage.settings import Settings

logger = logging.getLogger("ml-api")

ListJobs = Callable[[], list[TrainingJob]]
CreateJob = Callable[[str, str, dict, str, str | None], TrainingJob]
CancelJob = Callable[[str], TrainingJob | None]
ModelStatusOf = Callable[[ModelEntry], dict | None]
ModelDownloadUrlOf = Callable[[ModelEntry], str | None]


def _default_model_status_of(entry: ModelEntry) -> dict | None:
    """`head-object` en vivo (P3-15). Inyectable para no depender de S3 en las
    pruebas; en producción apunta al bucket del `registry.json`."""
    return object_status(get_minio_client(), bucket=entry.s3_bucket, key=entry.s3_key)


def _default_model_download_url_of(entry: ModelEntry) -> str | None:
    return presigned_get_url(get_minio_client(), bucket=entry.s3_bucket, key=entry.s3_key)


_PENDING = {
    "experiments": PendingEndpoint(
        ticket="P3-12", message="Experiments se conecta a MLflow en vivo en P3-12."
    ),
    "evaluation": PendingEndpoint(
        ticket="P3-13",
        message="La selección está cerrada, pero la evaluación final todavía no dejó "
        "reports/evaluation (P3-13).",
    ),
    "models": PendingEndpoint(
        ticket="P3-14",
        message="El catálogo de modelos (reports/models/registry.json) todavía no existe; "
        "lo publica P3-14.",
    ),
    "inference": PendingEndpoint(
        ticket="P3-16", message="Inference se conecta a la inferencia real en P3-16."
    ),
}

# Los recortes viven en `data/derived/crops/images/<crop_id>.jpg` (P3-04). El
# formato lo fija `crops.build.make_crop_id`; se revalida aquí porque este
# endpoint sirve un archivo a partir de un valor de la URL (path traversal).
CROP_ID_PATTERN = re.compile(r"^\d{6}_\d{6}$")


def create_app(
    settings: Settings | None = None,
    *,
    list_jobs: ListJobs | None = None,
    create_job: CreateJob | None = None,
    cancel_job: CancelJob | None = None,
    model_status_of: ModelStatusOf | None = None,
    model_download_url_of: ModelDownloadUrlOf | None = None,
) -> Starlette:
    settings = settings if settings is not None else Settings()
    list_jobs = list_jobs or (lambda: list_training_jobs(get_engine()))
    create_job = create_job or (
        lambda dataset_release, manifest_id, config, run_kind, grid_row: create_training_job(
            get_engine(),
            dataset_release=dataset_release,
            manifest_id=manifest_id,
            config=config,
            run_kind=run_kind,
            grid_row=grid_row,
        )
    )
    cancel_job = cancel_job or (
        lambda job_id: request_training_job_cancellation(get_engine(), job_id)
    )
    model_status_of = model_status_of or _default_model_status_of
    model_download_url_of = model_download_url_of or _default_model_download_url_of

    registry_path = settings.reports_dir / "models" / "registry.json"
    active_version_path = settings.reports_dir / "models" / "active_version.json"

    def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    def training_jobs(_: Request) -> JSONResponse:
        jobs = list_jobs()
        return JSONResponse(TrainingJobList(jobs=jobs).model_dump(mode="json"))

    async def create_training_job_route(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "el cuerpo debe ser JSON"}, status_code=400)
        if not isinstance(payload, dict) or not isinstance(payload.get("dataset_release"), str):
            return JSONResponse({"error": "falta dataset_release (str)"}, status_code=400)
        if not isinstance(payload.get("config"), dict):
            return JSONResponse({"error": "falta config (obj)"}, status_code=400)
        if not isinstance(payload.get("run_kind"), str):
            return JSONResponse({"error": "falta run_kind (str)"}, status_code=400)
        grid_row = payload.get("grid_row")
        if grid_row is not None and not isinstance(grid_row, str):
            return JSONResponse({"error": "grid_row debe ser str o null"}, status_code=400)

        try:
            validated_config, manifest_id = validate_new_training_job(
                dataset_release=payload["dataset_release"],
                config=payload["config"],
                run_kind=payload["run_kind"],
                grid_row=grid_row,
                reports_dir=settings.reports_dir,
                derived_dir=settings.derived_dir,
            )
        except TrainingJobRejected as error:
            return JSONResponse({"error": str(error)}, status_code=400)

        job = create_job(
            payload["dataset_release"],
            manifest_id,
            validated_config.model_dump(),
            payload["run_kind"],
            grid_row,
        )
        return JSONResponse(job.model_dump(mode="json"), status_code=201)

    async def cancel_training_job_route(request: Request) -> JSONResponse:
        job_id = request.path_params["job_id"]
        try:
            job = cancel_job(job_id)
        except TrainingJobNotCancellable as error:
            return JSONResponse(
                {"error": f"no se puede cancelar: ya está {error}"}, status_code=409
            )
        if job is None:
            return JSONResponse({"error": "no existe ese training_job"}, status_code=404)
        return JSONResponse(job.model_dump(mode="json"))

    def evaluation(_: Request) -> JSONResponse:
        reason = lock_reason(
            selection_path=settings.reports_dir / "selection.json",
            derived_dir=settings.derived_dir,
        )
        if reason:
            return JSONResponse(EvaluationLocked(message=reason).model_dump(mode="json"))
        report = build_evaluation_report(
            selection_path=settings.reports_dir / "selection.json",
            evaluation_dir=settings.reports_dir / "evaluation",
        )
        if report is None:
            return JSONResponse(_PENDING["evaluation"].model_dump(mode="json"))
        return JSONResponse(report.model_dump(mode="json"))

    def crop(request: Request) -> FileResponse | JSONResponse:
        """Sirve un recorte real para la galería de Evaluation (P3-15)."""
        crop_id = request.path_params["crop_id"]
        if not CROP_ID_PATTERN.match(crop_id):
            return JSONResponse({"error": "crop_id inválido"}, status_code=400)
        path = settings.derived_dir / "crops" / "images" / f"{crop_id}.jpg"
        if not path.is_file():
            return JSONResponse({"error": "no existe ese recorte"}, status_code=404)
        return FileResponse(path, media_type="image/jpeg")

    def models(_: Request) -> JSONResponse:
        registry = load_registry(registry_path)
        if registry is None:
            return JSONResponse(_PENDING["models"].model_dump(mode="json"))
        payload = build_list(
            registry,
            active_version=read_active_version(active_version_path),
            status_of=model_status_of,
        )
        return JSONResponse(payload.model_dump(mode="json"))

    def model_detail(request: Request) -> JSONResponse:
        registry = load_registry(registry_path)
        if registry is None:
            return JSONResponse(
                {"error": "Models todavía no está publicado (P3-14)"}, status_code=404
            )
        version = request.path_params["version"]
        try:
            detail = build_detail(
                registry,
                version,
                active_version=read_active_version(active_version_path),
                status_of=model_status_of,
                download_url_of=model_download_url_of,
            )
        except ModelVersionNotFound:
            return JSONResponse({"error": f"no existe la versión {version}"}, status_code=404)
        return JSONResponse(detail.model_dump(mode="json"))

    async def set_active_model(request: Request) -> JSONResponse:
        registry = load_registry(registry_path)
        if registry is None:
            return JSONResponse(
                {"error": "Models todavía no está publicado (P3-14)"}, status_code=404
            )
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "el cuerpo debe ser JSON"}, status_code=400)
        version = payload.get("version") if isinstance(payload, dict) else None
        if not isinstance(version, str) or not version:
            return JSONResponse({"error": "falta version (str)"}, status_code=400)
        try:
            activate_version(registry, version, status_of=model_status_of)
        except ModelVersionNotFound:
            return JSONResponse({"error": f"no existe la versión {version}"}, status_code=404)
        except ModelObjectMissing:
            return JSONResponse(
                {
                    "error": f"el objeto de la versión {version} no existe en S3; "
                    "no se marca como activa"
                },
                status_code=400,
            )
        write_active_version(active_version_path, version)
        detail = build_detail(
            registry,
            version,
            active_version=version,
            status_of=model_status_of,
            download_url_of=model_download_url_of,
        )
        return JSONResponse(detail.model_dump(mode="json"))

    def pending(name: str):
        async def handler(_: Request) -> JSONResponse:
            return JSONResponse(_PENDING[name].model_dump(mode="json"))

        return handler

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/training/jobs", training_jobs, methods=["GET"]),
            Route("/training/jobs", create_training_job_route, methods=["POST"]),
            Route("/training/jobs/{job_id}/cancel", cancel_training_job_route, methods=["POST"]),
            Route("/experiments/{experiment_id}/runs", list_runs, methods=["GET"]),
            Route(
                "/experiments/runs/{run_id}/metrics/{metric_key}",
                get_metric_history,
                methods=["GET"],
            ),
            Route("/experiments/runs/{run_id}/tags", update_run_tag, methods=["POST"]),
            Route("/evaluation", evaluation, methods=["GET"]),
            Route("/crops/{crop_id}", crop, methods=["GET"]),
            Route("/models", models, methods=["GET"]),
            Route("/models/active", set_active_model, methods=["POST"]),
            Route("/models/{version}", model_detail, methods=["GET"]),
            Route("/inference", pending("inference"), methods=["GET"]),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            )
        ],
    )


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=settings.ml_api_host, port=settings.ml_api_port)


if __name__ == "__main__":
    main()
