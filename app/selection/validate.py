"""Validación de corridas (chequeo 1 de #18); puro, recibe `RunSummary` ya leídas.

Una corrida cuenta como válida solo si:
1. está `FINISHED`;
2. usa el mismo `manifest_sha256` y las mismas `classes`;
3. tiene ≥2 épocas (`stopped_epoch`);
4. tiene `git_dirty == "false"`;
5. no duplica los parámetros de rejilla de otra corrida;
6. trae las semillas, `patience` y `min_delta` congelados (secciones 4 y 6);
7. si es `campaign`, trae un `grid_row` `r01` a `r12`.

A nivel agregado, exige ≥`min_required` válidas y cada uno de los 7 parámetros
de la rejilla con ≥2 valores distintos.
"""

import re
from collections import defaultdict
from datetime import datetime, timezone

from selection.contracts import (
    GRID_PARAMS,
    ExperimentsValidity,
    RunSummary,
    RunValidity,
)
from training.grid import FROZEN_MIN_DELTA, FROZEN_PATIENCE, FROZEN_SEEDS

CAMPAIGN = "campaign"
GRID_ROW_PATTERN = re.compile(r"^r(0[1-9]|1[0-2])$")


def _grid_signature(run: RunSummary) -> tuple[str, ...]:
    return tuple(str(run.params.get(param)) for param in GRID_PARAMS)


def _frozen_reasons(params: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    for field, expected in FROZEN_SEEDS.items():
        if params.get(field) != str(expected):
            reasons.append(f"{field}={params.get(field)!r} (congelada en {expected})")
    if params.get("patience") != str(FROZEN_PATIENCE):
        reasons.append(f"patience={params.get('patience')!r} (congelada en {FROZEN_PATIENCE})")
    if params.get("min_delta") != str(FROZEN_MIN_DELTA):
        reasons.append(f"min_delta={params.get('min_delta')!r} (congelada en {FROZEN_MIN_DELTA})")
    return reasons


def _grid_row_reasons(run: RunSummary) -> list[str]:
    if run.tags.get("run_kind") != CAMPAIGN:
        return []
    if not run.grid_row:
        return ["run_kind=campaign sin grid_row"]
    if not GRID_ROW_PATTERN.fullmatch(run.grid_row):
        return [f"grid_row inválido para campaign: {run.grid_row!r}"]
    return []


def _check_run(
    run: RunSummary,
    *,
    manifest_sha256: str,
    classes: dict[str, str],
    duplicated_ids: set[str],
) -> RunValidity:
    """Reglas por corrida; el resultado se reutiliza para 'valid'/'invalid'."""
    reasons: list[str] = []
    if run.status != "FINISHED":
        reasons.append(f"status={run.status!r} (se espera 'FINISHED')")
    if run.tags.get("manifest_sha256") != manifest_sha256:
        reasons.append("manifest_sha256 distinto al del manifiesto objetivo")
    if run.classes != classes:
        reasons.append("classes distintas a las del manifiesto objetivo")
    if run.metrics.get("stopped_epoch", 0.0) < 2:
        reasons.append("menos de 2 épocas entrenadas")
    if run.tags.get("git_dirty") != "false":
        reasons.append(f"git_dirty={run.tags.get('git_dirty')!r} (se espera 'false')")
    if run.run_id in duplicated_ids:
        reasons.append("parámetros de rejilla duplicados con otra corrida")
    reasons.extend(_grid_row_reasons(run))
    reasons.extend(_frozen_reasons(run.params))
    return RunValidity(
        run_id=run.run_id,
        grid_row=run.grid_row,
        valid=not reasons,
        reasons=reasons,
    )


def _duplicated_ids(runs: list[RunSummary]) -> set[str]:
    by_signature: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for run in runs:
        by_signature[_grid_signature(run)].append(run.run_id)
    return {run_id for run_ids in by_signature.values() if len(run_ids) > 1 for run_id in run_ids}


def validate_runs(
    runs: list[RunSummary],
    *,
    manifest_sha256: str,
    classes: dict[str, str],
    min_required: int = 10,
) -> ExperimentsValidity:
    duplicated = _duplicated_ids(runs)
    checked = [
        _check_run(run, manifest_sha256=manifest_sha256, classes=classes, duplicated_ids=duplicated)
        for run in runs
    ]
    by_id = {run.run_id: run for run in runs}
    valid_runs = [by_id[entry.run_id] for entry in checked if entry.valid]

    per_param_values = {
        param: len({str(run.params.get(param)) for run in valid_runs}) for param in GRID_PARAMS
    }
    passed = len(valid_runs) >= min_required and all(
        count >= 2 for count in per_param_values.values()
    )

    return ExperimentsValidity(
        manifest_sha256=manifest_sha256,
        classes=classes,
        min_required=min_required,
        valid=[entry.run_id for entry in checked if entry.valid],
        invalid=[entry for entry in checked if not entry.valid],
        per_param_values=per_param_values,
        passed=passed,
        produced_at=datetime.now(timezone.utc),
    )
