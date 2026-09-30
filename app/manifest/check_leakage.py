"""P3-06: verifica que ningún recorte, imagen de origen ni grupo cruce particiones.

    uv run python -m manifest.check_leakage --manifest ../data/derived/manifests/v0.1.1/manifest.csv

Imprime un JSON con las intersecciones y termina con código 1 si hay fuga.
"""

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

LEAKAGE_KEYS = ("crop_id", "source_image_id", "duplicate_group_id")


def find_leakage(rows: Sequence[Mapping]) -> dict[str, list[dict]]:
    """Por cada llave, los valores que aparecen en más de una partición."""
    leakage = {}
    for key in LEAKAGE_KEYS:
        owners: dict[str, set[str]] = {}
        for row in rows:
            owners.setdefault(str(row[key]), set()).add(row["split"])
        leakage[key] = [
            {"value": value, "splits": sorted(splits)}
            for value, splits in sorted(owners.items())
            if len(splits) > 1
        ]
    return leakage


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Ruta a manifest.csv.")
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    leakage = find_leakage(rows)
    leaked = any(leakage.values())
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "rows": len(rows),
                "leakage": leaked,
                "intersections": {key: len(values) for key, values in leakage.items()},
                "details": leakage,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if leaked else 0


if __name__ == "__main__":
    sys.exit(main())
