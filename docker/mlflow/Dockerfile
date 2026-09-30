# La imagen oficial de MLflow no trae driver de MariaDB (pymysql) ni cliente
# S3 (boto3) -- los necesita para el backend store (MariaDB) y el artifact
# store (MinIO) de P3-03. Se instalan aquí, en el build, no con un `pip
# install` en el entrypoint del contenedor: así no depende de que PyPI esté
# alcanzable cada vez que el servicio arranca o se reinicia.
FROM ghcr.io/mlflow/mlflow:v3.16.1

RUN pip install --no-cache-dir pymysql boto3
