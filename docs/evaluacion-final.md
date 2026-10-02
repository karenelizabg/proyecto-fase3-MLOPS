# Evaluación final (P3-13 / #20)

Evaluación **única** del candidato `r02` sobre el split `test`, hecha después de
sellar la selección (`reports/selection.json`, ver `docs/seleccion-candidato.md`).
Se corrió una sola vez; `app/final.py` se niega a repetirla.

- **Candidato:** `r02` · `run_id` `7e7b4a4b35464cfebb6b41714a3ad931`
- **`checkpoint_sha256`:** `f759cde23fc314b63c4a4e3c20127ef1eb84121e579b04e7c97911ada20ab7d9` (verificado antes de predecir)
- **Manifiesto:** `v0.1.1` · split `test`, **73 recortes**
- **Corrida de evaluación en MLflow:** `0bb3379f5a444c00b2627e8135d3eb34` (`evaluation_of` = r02)
- **Comando:** `app/.venv/bin/python app/final.py --split test --checkpoint <best.pt local>`

## Métricas

| Métrica | Valor |
|---|---|
| `accuracy` (diagonal ÷ total, sin redondear) | **0.9863013698630136** (72 / 73) |
| `macro_f1` | 0.9862081995087851 |
| Baseline de clase mayoritaria | 0.5342 |
| Accuracy en validación (r02) | 0.9618 |

| Clase | Precisión | Recall | F1 | Support |
|---|---|---|---|---|
| cat | 0.9750 | 1.0000 | 0.9873 | 39 |
| dog | 1.0000 | 0.9706 | 0.9851 | 34 |

Matriz de confusión (filas = clase real, columnas = predicha):

| | pred. cat | pred. dog |
|---|---|---|
| **real cat** | 39 | 0 |
| **real dog** | 1 | 33 |

`recompute.py` (scikit-learn, desde `predictions.csv`) coincide con `metrics.json`.

## Análisis de errores

- **Clase más confundida:** `dog`, con 1 error (un perro clasificado como gato).
- **Único error:** recorte `000275_000320` (imagen fuente `275`): real `dog`,
  predicho `cat` con probabilidad 0.577, es decir, una decisión poco segura
  (`prob_dog` = 0.423).
- **¿La accuracy oculta un recall bajo?** No: ninguna clase baja de 0.70
  (el menor es `dog`, 0.9706).
- **Aciertos:** los 6 primeros tienen probabilidad entre 0.95 y 1.00
  (`analysis.json`, campo `successes`).

## Cómo leer el resultado

- Supera con holgura el baseline (0.534) y el objetivo de 0.85.
- El `test` solo tiene 73 recortes: un error mueve la accuracy 1.4 puntos. Un
  intervalo de Wilson al 95 % para 72/73 va de ≈0.926 a ≈0.998, así que el
  0.986 no es una estimación precisa. La diferencia con validación (0.962,
  131 recortes) **no es significativa**.
- Con un solo error no hay un patrón de fallo que analizar; solo se puede decir
  que fue un caso de baja confianza.
- El test se usó una vez y no se ajustó nada después de verlo.

## Archivos

`reports/evaluation/`: `predictions.csv` (`crop_id`, `source_image_id`, clase real
y predicha, probabilidades), `metrics.json` y `analysis.json`. Los tres se
recalculan con `app/recompute.py`.

## Nota de reproducción

El proxy de artefactos de MLflow falló al bajar el checkpoint (~45 MB) por HTTP,
así que `best.pt` se reconstruyó desde el almacenamiento de MinIO y se pasó con
`--checkpoint`. `final.py` verifica igualmente su SHA-256 contra `selection.json`.
