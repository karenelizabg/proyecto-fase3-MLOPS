import hashlib
import json
import tarfile
from pathlib import Path

import boto3


def calculate_sha256(filepath: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def verify_and_generate_evidence():
    registry_path = Path("models/registry.json")
    if not registry_path.exists():
        print("No se encontró models/registry.json")
        return

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    session = boto3.Session(profile_name="mlops-p3")
    s3 = session.client("s3")

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    evidence_path = docs_dir / "evidencia-recarga.md"

    evidence_md = "# Evidencia de Recarga Limpia y Verificación\n\n"
    evidence_md += (
        "Este documento contiene la validación de los artefactos descargados desde S3, "
        "garantizando su integridad mediante SHA-256 y VersionId.\n\n"
    )

    for version, data in registry.items():
        print(f"Verificando versión {version}...")
        s3_path = data["s3_path"]
        expected_sha = data["sha256"]
        version_id = data.get("VersionId")

        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts[0]
        key = "/".join(path_parts[1:])

        local_tar = Path(f"dist/downloaded_v{version}.tar.gz")
        local_tar.parent.mkdir(exist_ok=True)

        print(f"  Descargando desde S3 (VersionId: {version_id})...")
        try:
            extra_args = (
                {"VersionId": version_id} if version_id and version_id.lower() != "null" else {}
            )
            s3.download_file(Bucket=bucket, Key=key, Filename=str(local_tar), ExtraArgs=extra_args)
        except Exception as e:
            print(f"  ❌ Error al descargar la versión {version}: {e}")
            continue

        local_sha = calculate_sha256(local_tar)
        match = local_sha == expected_sha

        print(f"  SHA-256 local:  {local_sha}")
        print(f"  {'HASH COINCIDE' if match else '❌ ERROR DE HASH'}")

        extract_dir = Path(f"eval_v{version}")
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f" Archivos extraídos en: {extract_dir}/")

        evidence_md += f"## Versión {version}\n"
        evidence_md += f"- **Ruta S3:** `{s3_path}`\n"
        evidence_md += f"- **VersionId:** `{version_id}`\n"
        evidence_md += f"- **SHA-256 Esperado (Registry):** `{expected_sha}`\n"
        evidence_md += f"- **SHA-256 Calculado (Local):** `{local_sha}`\n"
        evidence_md += (
            f"- **Resultado de Integridad:** {'COINCIDE' if match else '❌ NO COINCIDE'}\n\n"
        )
        evidence_md += "### Prueba de Inferencia (Clean Reload)\n"
        evidence_md += (
            "> **Instrucción para Emilio:** El paquete se descomprimió correctamente. "
            "Carga el `architecture.json` y los pesos, ejecuta la inferencia sobre las 3 imágenes "
            "de prueba y documenta aquí la comparación contra `predictions.csv`.\n\n"
        )

        local_tar.unlink()

    with open(evidence_path, "w", encoding="utf-8") as f:
        f.write(evidence_md)

    print(f"\nEvidencia base generada en: {evidence_path}")


if __name__ == "__main__":
    verify_and_generate_evidence()
