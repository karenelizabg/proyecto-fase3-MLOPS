# Reproducibilidad y smoke test (P3-10)

**Objetivo:** comprobar, antes de la campaña (issue #17), que una corrida corta
repetida con la misma configuración y semillas da las mismas métricas, y que
`checkpoint/best.pt` cargado en un proceso nuevo reproduce las predicciones.
La corrida del smoke test es la versión `0.1.0` del modelo ("ensayo, no
seleccionada", sección 10 de `decisiones-proyecto3.md`).

> **Estado:** la parte automatizada (sección 1) está implementada y corre en el
> CI. La corrida real (sección 2) **está pendiente**: los campos marcados como
> _pendiente_ se llenan con los `run_id` y tiempos reales de MLflow. Este
> documento no lleva datos inventados.

## 1. Prueba automatizada (modo SMOKE)

`app/tests/test_smoke_reproducibility.py` entrena con
`ml_worker.run_training.run_training`, la misma función que usa el worker real
(`set_reproducibility`, `create_dataloader` con `batch_size=config.batch_size` y
`seed_aug=config.seed_aug`, y `train_with_mlflow` con
`{"run_kind": "smoke", "grid_row": ""}`). Usa recortes JPEG pequeños escritos en
`tmp_path` con un `manifest.csv` real y un MLflow temporal (SQLite y artefactos en
`tmp_path`); no escribe en `app/` ni en `mlruns/`. `GIT_COMMIT` y
`GIT_DIRTY` se fijan con `monkeypatch` solo dentro de la prueba.

Comprueba:

1. Dos corridas con la misma config y semillas dan las mismas `train_loss`,
   `train_accuracy`, `val_loss`, `val_accuracy` y `val_macro_f1` por época
   (tolerancia 1e-6) y el mismo `best_epoch` y `stopped_epoch`.
2. Los parámetros y tags registrados corresponden a lo que se usó
   (`batch_size`, `run_kind`, `grid_row`, `manifest_sha256` del manifiesto
   entrenado).
3. En un proceso nuevo (`multiprocessing.get_context("spawn")`) se descarga
   `checkpoint/best.pt`, su SHA-256 coincide con el tag `checkpoint_sha256`, se
   carga y se predice sobre los recortes de `val`; las probabilidades coinciden
   con las del proceso original dentro de 1e-6.

## 2. Corrida real (manifiesto v0.1.1)

Dos corridas `run_kind=smoke` lanzadas desde Training con el manifiesto v0.1.1,
sus recortes reales, la misma config y las mismas semillas.

| Dato | Corrida A (candidata a 0.1.0) | Corrida B |
|---|---|---|
| `run_id` | _pendiente_ | _pendiente_ |
| Manifiesto (`manifest_id`) | _pendiente_ | _pendiente_ |
| `manifest_sha256` | _pendiente_ | _pendiente_ |
| `git_commit` / `git_dirty` | _pendiente_ | _pendiente_ |
| `best_epoch` / `stopped_epoch` | _pendiente_ | _pendiente_ |
| `best_val_loss` | _pendiente_ | _pendiente_ |
| `best_val_accuracy` | _pendiente_ | _pendiente_ |
| `best_val_macro_f1` | _pendiente_ | _pendiente_ |
| `duration_s` | _pendiente_ | _pendiente_ |

Etiquetas: la corrida A es la candidata a `0.1.0` (**ensayo, no seleccionada**).
La B solo confirma la reproducibilidad.

**Tiempo por corrida (sección 7):** _pendiente_ (CPU, Docker en Mac M3, 4 hilos).
Si pasa de 15 min, la campaña se lanza con imagen {112, 128}.

## 3. Checkpoint en un proceso nuevo (corrida real)

Se verifica con `python -m training.smoke verify` (ver el comando en el PR):
compara las dos corridas y, para la A, descarga `best.pt`, recalcula su SHA-256
y predice sobre los recortes reales de `val` en un proceso `spawn`.

| Comprobación | Resultado |
|---|---|
| Métricas por época y `best_epoch` A = B (1e-6) | _pendiente_ |
| `checkpoint_sha256` (tag) = SHA-256 recalculado | _pendiente_ |
| Recortes de `val` predichos y máx. \|Δ probabilidad\| | _pendiente_ |
