"""Validación de corridas (chequeo 1 de #18); puro, recibe `RunSummary` ya leídas.

Una corrida cuenta como válida solo si:
1. está `FINISHED`;
2. usa el mismo `manifest_sha256` y las mismas `classes`;
3. tiene ≥2 épocas (`stopped_epoch`);
4. tiene `git_dirty == "false"`;
5. no duplica los parámetros de rejilla de otra corrida.

A nivel agregado, exige ≥`min_required` válidas y cada uno de los 7 parámetros
de la rejilla con ≥2 valores distintos.
"""

from collections import defaultdict
from datetime import datetime, timezone

from selection.contracts import (
    GRID_PARAMS,
    ExperimentsValidity,
    RunSummary,
    RunValidity,
)


def _grid_signature(run: RunSummary) -> tuple[str, ...]:
    return tuple(str(run.params.get(param)) for param in GRID_PARAMS)


def _run_reasons(
    run: RunSummary,
    *,
    manifest_sha256: str,
    classes: dict[str, str],
    duplicated_ids: set[str],
) -> list[str]:
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
    return reasons


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

    validity: list[RunValidity] = []
    valid_runs: list[RunSummary] = []
    for run in runs:
        reasons = _run_reasons(
            run, manifest_sha256=manifest_sha256, classes=classes, duplicated_ids=duplicated
        )
        validity.append(
            RunValidity(
                run_id=run.run_id,
                grid_row=run.grid_row,
                valid=not reasons,
                reasons=reasons,
            )
        )
        if not reasons:
            valid_runs.append(run)

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
        valid=[run.run_id for run in valid_runs],
        invalid=[entry for entry in validity if not entry.valid],
        per_param_values=per_param_values,
        passed=passed,
        produced_at=datetime.now(timezone.utc),
    )
