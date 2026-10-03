"""Genera la tarjeta de modelo (P3-14) solo a partir de datos.

Fuentes: params, tags y métricas del run (mlflow-store/mlflow.sql), reports/selection.json,
reports/manifests/<release>/counts.json y, si el run es el seleccionado, sus métricas de
test (artifacts/evaluation/ del run). Si falta un dato, falla en vez de dejar un campo vacío.

    uv run python card.py --run-id b828e032156e423097e90d797b643faa --version 0.1.0
    uv run python card.py --run-id 7e7b4a4b35464cfebb6b41714a3ad931 --version 1.0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlflow_dump import ROOT, artifacts_dir, load_run

REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Falta {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def need(mapping: dict, key: str, source: str):
    value = mapping.get(key)
    if value is None or value == "":
        sys.exit(f"Falta '{key}' en {source}: la tarjeta no puede quedar con campos vacíos.")
    return value


def f4(value) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def build_card(run_id: str, version: str, reports: Path = REPORTS) -> str:
    run = load_run(run_id)
    params, tags, metrics = run["params"], run["tags"], run["metrics"]
    src = f"el run {run_id}"

    selection = read_json(reports / "selection.json")
    selected_id = need(selection, "run_id", "selection.json")
    selected = selected_id == run_id
    if int(version.split(".")[0]) >= 1 and not selected:
        sys.exit(
            f"La versión {version} debe ser el modelo seleccionado ({selected_id}), no {run_id}."
        )

    release = need(tags, "release", src)
    manifest_id = need(tags, "manifest_id", src)
    ckpt_sha = need(tags, "checkpoint_sha256", src)
    run_kind = need(tags, "run_kind", src)
    classes = json.loads(need(tags, "classes", src))
    class_names = [classes[k] for k in sorted(classes, key=int)]

    if selected:
        for key, value in (("checkpoint_sha256", ckpt_sha), ("manifest_id", manifest_id)):
            if selection.get(key) != value:
                sys.exit(f"selection.json y el run no coinciden en '{key}'.")
        candidate = need(selection, "candidate", "selection.json")
        status = (
            f"**Seleccionada.** Métrica de selección "
            f"`{need(selection, 'selection_metric', 'selection.json')}`, "
            f"fila `{need(candidate, 'grid_row', 'selection.json')}`, "
            f"seleccionada el {need(selection, 'selected_at', 'selection.json')}."
        )
    else:
        label = "Ensayo, no seleccionada" if run_kind == "smoke" else "No seleccionada"
        status = (
            f"**{label}** (run_kind `{run_kind}`). "
            f"El modelo seleccionado es el run `{selected_id}`."
        )

    counts = read_json(reports / "manifests" / release / "counts.json")
    if counts.get("release") != release:
        sys.exit(f"counts.json no es del release {release}.")
    totals = need(counts, "totals", "counts.json")
    splits = need(counts, "splits", "counts.json")

    val_keys = (
        "best_val_accuracy",
        "best_val_macro_f1",
        "best_val_loss",
        "best_epoch",
        "stopped_epoch",
    )
    val = {key: need(metrics, key, src) for key in val_keys}

    out: list[str] = [f"# Tarjeta de modelo - versión {version}", "", f"> {status}", ""]

    out += [
        "## Identificación",
        f"- Versión del paquete: `{version}`",
        f"- Run de MLflow: `{run_id}` "
        f"(`{need(tags, 'mlflow.runName', src)}`, run_kind `{run_kind}`)",
        "",
        "## Propósito",
        f"Clasificar recortes de imagen en {len(class_names)} clases ({', '.join(class_names)}), "
        f"entrenado con el manifiesto `{manifest_id}` del release `{release}`.",
        "",
        "## Clases",
        "| Índice | Clase |",
        "|---|---|",
        *[f"| {k} | {classes[k]} |" for k in sorted(classes, key=int)],
        "",
        "## Datos y release",
        f"- Release del dataset: `{release}`; manifiesto `{manifest_id}` "
        f"(SHA-256 `{need(tags, 'manifest_sha256', src)}`)",
        f"- {totals['crops']} recortes de {totals['originals']} imágenes originales; por clase: "
        + ", ".join(f"{name} {info['crops']}" for name, info in totals["classes"].items()),
        "",
        "## Split",
        "| Split | Recortes | Originales | Recortes por clase |",
        "|---|---|---|---|",
    ]
    for name in ("train", "val", "test"):
        s = need(splits, name, "counts.json")
        per_class = " / ".join(f"{c}: {v['crops']}" for c, v in s["classes"].items())
        out.append(f"| {name} | {s['crops']} | {s['originals']} | {per_class} |")

    out += [
        "",
        "## Métricas de validación (MLflow)",
        "| Métrica | Valor |",
        "|---|---|",
        *[f"| {key} | {f4(value)} |" for key, value in val.items()],
        "",
        "## Métricas de test",
    ]

    limitations = [
        f"Solo reconoce {len(class_names)} clases ({', '.join(class_names)}); "
        f"cualquier otra imagen "
        "se asignará a una de ellas.",
        f"Se entrenó y evaluó sobre recortes del manifiesto `{manifest_id}`, "
        f"no sobre imágenes completas; "
        "la entrada debe prepararse igual (ver `preprocess.json`).",
    ]

    if selected:
        art = artifacts_dir(run_id) / "evaluation"
        test = read_json(art / "metrics.json")
        analysis = read_json(art / "analysis.json")
        if list(test["classes"]) != class_names:
            sys.exit("Las clases de metrics.json no coinciden con las del run.")
        out += [
            f"Split `test`, {test['total']} recortes. Accuracy **{f4(test['accuracy'])}**, "
            f"macro F1 **{f4(test['macro_f1'])}**.",
            "",
            "| Clase | Precision | Recall | F1 | Soporte |",
            "|---|---|---|---|---|",
            *[
                f"| {c} | {f4(m['precision'])} | {f4(m['recall'])} | "
                f"{f4(m['f1'])} | {m['support']} |"
                for c, m in test["per_class"].items()
            ],
            "",
            "Matriz de confusión (filas: clase real, columnas: predicha):",
            "",
            "| | " + " | ".join(class_names) + " |",
            "|---|" + "---|" * len(class_names),
            *[
                f"| {class_names[i]} | " + " | ".join(str(v) for v in row) + " |"
                for i, row in enumerate(test["confusion_matrix"])
            ],
        ]
        worst = analysis["most_confused_class"]
        limitations += [
            f"El test tiene {test['total']} recortes "
            f"({splits['test']['originals']} imágenes originales): "
            f"cada error cambia la accuracy {100 / test['total']:.1f} puntos.",
            f"Clase con menor recall: `{worst}` ({f4(analysis['recall_per_class'][worst])}). "
            f"Accuracy de predecir siempre la clase mayoritaria: "
            f"{f4(analysis['baseline_majority_accuracy'])}.",
            *[
                f"Error en test: recorte `{e['crop_id']}` "
                f"({e['true_class']} -> {e['predicted_class']}, "
                f"probabilidad {e['probability']:.3f})."
                for e in analysis["errors"]
            ],
        ]
        if analysis.get("accuracy_hides_low_recall"):
            limitations.append("La accuracy oculta un recall bajo en al menos una clase.")
    else:
        out.append(
            f"No evaluada en test: el split de test es exclusivo del modelo seleccionado "
            f"(run `{selected_id}`)."
        )
        limitations += [
            f"Sin evaluación en test: solo hay métricas de validación "
            f"({splits['val']['crops']} recortes).",
            f"Se detuvo por early stopping en la época {val['stopped_epoch']}; "
            f"los pesos son de la época {val['best_epoch']}.",
        ]

    limitations += [
        f"Entrenado en `{need(tags, 'device', src)}` sobre `{need(tags, 'platform', src)}` "
        f"con Python "
        f"{need(tags, 'python_version', src)}, torch {need(tags, 'torch_version', src)} "
        f"y torchvision "
        f"{need(tags, 'torchvision_version', src)}; "
        f"en otro entorno puede haber diferencias numéricas mínimas.",
        "`build_model` descarga los pesos ImageNet de torchvision al construir el modelo: "
        "la carga necesita red o la caché de torch.",
    ]

    out += [
        "",
        "## Hiperparámetros",
        "| Parámetro | Valor |",
        "|---|---|",
        *[f"| {key} | {value} |" for key, value in sorted(params.items())],
        "",
        "## Limitaciones",
        *[f"- {item}" for item in limitations],
        "",
        "## Origen de los pesos",
        f"- Archivo: `artifacts/checkpoint/best.pt` del run `{run_id}` "
        f"(mejor época {val['best_epoch']} según validación)",
        f"- SHA-256 del checkpoint: `{ckpt_sha}`",
        f"- Código: commit `{need(tags, 'git_commit', src)}` "
        f"(git_dirty `{need(tags, 'git_dirty', src)}`)",
        "",
        "## Modo de carga",
        "Desde la carpeta extraída del paquete, con `app/` del repo en el `PYTHONPATH`:",
        "",
        "```python",
        "import json",
        "import torch",
        "from training.config import TrainingConfig",
        "from training.model import build_model",
        "from training.preprocess import get_preprocessing_transforms",
        "",
        'config = TrainingConfig.model_validate(json.load(open("config.json")))',
        "model = build_model(config)",
        'model.load_state_dict(torch.load("artifacts/checkpoint/best.pt", weights_only=True))',
        "model.eval()",
        "",
        "transform = get_preprocessing_transforms("
        '"test", config.image_size)  # descrita en preprocess.json',
        'class_map = json.load(open("class_map.json"))',
        "# probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)",
        "# clase = class_map[str(int(probs.argmax()))]",
        "```",
        "",
    ]

    card = "\n".join(out)
    if "None" in card:
        sys.exit("La tarjeta generada contiene 'None': revisa las fuentes de datos.")
    return card


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = p.parse_args(argv)

    output = args.output or ROOT / "build" / f"model_card_{args.version}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_card(args.run_id, args.version, args.reports_dir), encoding="utf-8")
    print(f"Tarjeta v{args.version} escrita en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
