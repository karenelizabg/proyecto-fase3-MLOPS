from datetime import timedelta

from minio import Minio
from minio.error import S3Error

from storage.settings import Settings


def get_minio_client() -> Minio:
    settings = Settings()
    return Minio(
        f"{settings.minio_endpoint}:{settings.minio_port}",
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_use_ssl,
    )


def get_bucket_name() -> str:
    return Settings().minio_bucket


def object_status(client: Minio, *, bucket: str, key: str) -> dict | None:
    """Estado en vivo de un objeto (P3-15, `head-object`).

    `None` si el objeto no existe. Los binarios de modelos pueden vivir en el
    bucket de MinIO o en el de AWS; el cliente lo decide quien llama (este
    módulo nunca lee credenciales por su cuenta, solo recibe el cliente).
    """
    try:
        stat = client.stat_object(bucket, key)
    except S3Error as error:
        if error.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket", "NotFound"}:
            return None
        raise
    return {
        "version_id": stat.version_id,
        "size_bytes": stat.size,
        "last_modified": stat.last_modified,
    }


def presigned_get_url(client: Minio, *, bucket: str, key: str, expires_seconds: int = 3600) -> str:
    """URL prefirmada de descarga (P3-15). El bucket es privado: sin firma no
    se puede bajar el paquete."""
    return client.presigned_get_object(bucket, key, expires=timedelta(seconds=expires_seconds))
