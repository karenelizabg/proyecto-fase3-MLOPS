"""Lee un run de MLflow desde el respaldo SQL del snapshot (mlflow-store/mlflow.sql).

No usa la librería de MLflow ni mlflow.db. Para revisar qué trae un run:

    uv run python mlflow_dump.py <run_id>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def snapshot_dir() -> Path:
    """Carpeta del snapshot de MLflow: MLFLOW_SNAPSHOT o, por defecto, mlflow-store/."""
    return Path(os.environ.get("MLFLOW_SNAPSHOT") or ROOT / "mlflow-store")


# Orden de columnas de MLflow cuando el INSERT no las nombra.
SCHEMAS = {
    "params": ["key", "value", "run_uuid"],
    "tags": ["key", "value", "run_uuid"],
    "latest_metrics": ["key", "value", "timestamp", "step", "is_nan", "run_uuid"],
    "metrics": ["key", "value", "timestamp", "run_uuid", "step", "is_nan"],
}
_INSERT = re.compile(r"INSERT\s+INTO\s+[`\"]?(\w+)[`\"]?\s*(?:\(([^)]*)\))?", re.IGNORECASE)
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}


def _scalar(token: str):
    if token.upper() == "NULL":
        return None
    for cast in (int, float):
        try:
            return cast(token)
        except ValueError:
            pass
    return token


def _tuples(text: str):
    """Extrae las tuplas (...) de un fragmento de INSERT, respetando comillas y escapes."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "(":
            i += 1
            continue
        i += 1
        values = []
        while i < n and text[i] != ")":
            c = text[i]
            if c == "'":
                buf, i = [], i + 1
                while i < n:
                    if text[i] == "\\" and i + 1 < n:
                        buf.append(_ESCAPES.get(text[i + 1], text[i + 1]))
                        i += 2
                    elif text[i] == "'" and i + 1 < n and text[i + 1] == "'":
                        buf.append("'")
                        i += 2
                    elif text[i] == "'":
                        i += 1
                        break
                    else:
                        buf.append(text[i])
                        i += 1
                values.append("".join(buf))
            elif c in ", \t\r\n":
                i += 1
            else:
                j = i
                while j < n and text[j] not in ",)":
                    j += 1
                values.append(_scalar(text[i:j].strip()))
                i = j
        i += 1
        yield values


def _rows(dump: Path):
    table, columns = None, None
    for line in dump.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _INSERT.search(line)
        if match:
            table = match.group(1).lower()
            if match.group(2):
                columns = [c.strip(' `"') for c in match.group(2).split(",")]
            else:
                columns = SCHEMAS.get(table)
            rest = line[match.end() :]
        elif table and line.lstrip().startswith(("(", "VALUES", "values")):
            rest = line
        else:
            table = None
            continue
        if not columns:
            continue
        for values in _tuples(rest):
            if len(values) == len(columns):
                yield table, dict(zip(columns, values, strict=True))


def load_run(run_id: str, dump: Path | None = None) -> dict:
    """Devuelve params, tags y métricas (último valor) de un run."""
    dump = dump or snapshot_dir() / "mlflow.sql"
    if not dump.exists():
        sys.exit(f"No existe el respaldo de MLflow: {dump} (falta dvc pull?)")
    params: dict = {}
    tags: dict = {}
    latest: dict = {}
    history: dict = defaultdict(list)
    for table, row in _rows(dump):
        if row.get("run_uuid") != run_id:
            continue
        if table == "params":
            params[row["key"]] = row["value"]
        elif table == "tags":
            tags[row["key"]] = row["value"]
        elif table == "latest_metrics":
            latest[row["key"]] = row["value"]
        elif table == "metrics":
            history[row["key"]].append(
                (row.get("step") or 0, row.get("timestamp") or 0, row["value"])
            )
    metrics = {key: max(values)[2] for key, values in history.items()}
    metrics.update(latest)
    if not params or not tags:
        sys.exit(f"El run {run_id} no tiene params/tags en {dump}.")
    return {"run_id": run_id, "params": params, "tags": tags, "metrics": metrics}


def artifacts_dir(run_id: str) -> Path:
    """Carpeta de artefactos del run dentro del snapshot de DVC."""
    root = snapshot_dir() / "artifacts"
    found = [p for p in root.glob(f"*/{run_id}/artifacts") if p.is_dir()]
    if len(found) != 1:
        sys.exit(f"Se esperaba 1 carpeta de artefactos para {run_id} en {root}; hay {len(found)}.")
    return found[0]


def typed_params(params: dict) -> dict:
    """MLflow guarda los params como texto: '32' -> 32, '0.0' -> 0.0, 'adam' -> 'adam'."""
    return {
        key: _scalar(value) if isinstance(value, str) else value for key, value in params.items()
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python mlflow_dump.py <run_id>")
    print(json.dumps(load_run(sys.argv[1]), indent=2, ensure_ascii=False))
