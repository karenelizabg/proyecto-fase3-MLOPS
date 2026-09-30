# Variante de Dockerfile con el grupo de dependencias `ml` (torch, torchvision,
# mlflow, scikit-learn, pillow fijados -- P3-03). La usan `trainer-worker` y
# `ml-api`, que sí necesitan entrenar/cargar el modelo; `app` y `copilot`
# siguen construyéndose con el Dockerfile base, sin torch (el Copilot no debe
# cargarlo: es un proceso aparte, más pesado, y sin ninguna razón para
# entrenar ni cargar un checkpoint).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app

# A diferencia del Dockerfile base: SÍ necesita un home real. trainer-worker
# monta ~/.aws de solo lectura ahí (docker-compose.yml) para el perfil SSO
# del equipo, y boto3 resuelve las credenciales vía $HOME/.aws/credentials.
RUN useradd --system --create-home --home-dir /home/appuser appuser
ENV HOME=/home/appuser

COPY pyproject.toml uv.lock ./
ENV UV_HTTP_TIMEOUT=180
ENV UV_CONCURRENT_DOWNLOADS=2
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --no-build --group ml

COPY ingestion/ ./ingestion/
COPY analyzers/ ./analyzers/
COPY policies/ ./policies/
COPY splits/ ./splits/
COPY storage/ ./storage/
COPY presentation/ ./presentation/
COPY projections/ ./projections/
COPY crops/ ./crops/
# P3-06/P3-07: ml-api los necesita para reverificar el manifiesto antes de
# crear un training_job (P3-09); trainer-worker los necesita para entrenar.
COPY manifest/ ./manifest/
COPY training/ ./training/
COPY mlflow_ops/ ./mlflow_ops/
COPY ml_worker/ ./ml_worker/
COPY ml_api/ ./ml_api/

ENV PATH="/app/.venv/bin:$PATH"
USER appuser

CMD ["python", "-m", "ml_worker"]
