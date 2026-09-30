"""Fixtures compartidos por las pruebas de `POST /training/jobs` (P3-09).

Escriben lo mismo que `dvc_manifest_stage.write_manifest()` deja en disco
(`reports/manifests/<release>/manifest_meta.json` +
`data/derived/manifests/<release>/manifest.csv`), no una forma inventada.
"""

import csv
import json
from pathlib import Path

VALID_TRAINING_CONFIG = {
    "optimizer": "adam",
    "batch_size": 16,
    "max_epochs": 15,
    "learning_rate": 0.001,
    "image_size": 128,
    "hidden_layers": 0,
    "dropout": 0.0,
    "seed_split": 42,
    "seed_train": 43,
    "seed_aug": 44,
    "seed_model": 45,
    "patience": 5,
    "min_delta": 0.01,
}


def write_manifest(
    *,
    reports_dir: Path,
    derived_dir: Path,
    release: str = "v0.1.1",
    quality_status: str = "passed",
    leaking: bool = False,
) -> str:
    """Escribe manifest_meta.json + manifest.csv; devuelve el manifest_id."""
    manifest_id = f"{release}-testmeta"
    meta_dir = reports_dir / "manifests" / release
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "manifest_meta.json").write_text(
        json.dumps(
            {
                "manifest_id": manifest_id,
                "quality_reference": {"status": quality_status, "dataset_version": release},
            }
        ),
        encoding="utf-8",
    )

    manifest_dir = derived_dir / "manifests" / release
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "crop_id": "000001_000001",
            "split": "train",
            "label": 0,
            "source_image_id": 1,
            "duplicate_group_id": "g000001",
        },
        {
            "crop_id": "000002_000001",
            "split": "val",
            "label": 1,
            "source_image_id": 2,
            "duplicate_group_id": "g000002",
        },
        {
            # Sin fuga: mismo crop_id que la fila de arriba solo si `leaking`,
            # para forzar la intersección entre splits que detecta `find_leakage`.
            "crop_id": "000002_000001" if leaking else "000003_000001",
            "split": "train",
            "label": 1,
            "source_image_id": 3,
            "duplicate_group_id": "g000003",
        },
    ]
    with (manifest_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return manifest_id
