# Decisiones del Proyecto 3

> **Congeladas antes de cualquier corrida de entrenamiento.** La fecha del commit
> que mergea este archivo es la evidencia de que las clases, la métrica de
> selección y la rejilla se eligieron antes de mirar el test (rúbrica 1.2, 2.1,
> 2.4, 3.1, 3.3 y 5.1). Cambiar algo después exige un commit nuevo que lo
> justifique y, si ya hubo corridas, repetirlas.

## 1. Release de origen

| Campo | Valor |
|---|---|
| Release | `v0.1.1` |
| Imágenes (`data/raw/images.dvc`) | `md5 951150dd4fb053f4665089fcb37a1c87.dir` · 600 archivos |
| Anotaciones (`data/raw/annotations.dvc`) | `md5 c7cb86ae7ece94ef7b853620e464a4d7.dir` · 10 archivos |
| Remote DVC | `s3://mlops-p2-dvc-cache-222629887955` |
| Baseline de P2 | commit `9cfea4c`, tag `p2-final-baseline` |

La política de compuerta y los conteos útiles por clase los documenta P3-02
(#5) en la sección "Política de compuerta" de este archivo.

## 2. Clases

- **Clases:** `cat` y `dog` (las dos categorías COCO del release).
- **Exclusiones:** ninguna otra categoría. Se excluyen las cajas inválidas
  (degeneradas o fuera de la imagen) y las imágenes faltantes; cada una queda en
  `exclusions.csv` con su motivo (P3-04).
- **Unidad de clasificación:** un recorte por caja válida, nunca la imagen
  completa.
- **Conteo en bruto** (antes de filtrar cajas inválidas): `dog` 300 imágenes
  originales y `cat` 301. `dog` está exactamente en el mínimo de 300; P3-02
  confirma el conteo después del filtro.

## 3. CNN

- **Arquitectura:** ResNet18 de `torchvision`.
- **Pesos:** `ResNet18_Weights.IMAGENET1K_V1` (preentrenados en ImageNet-1k,
  descargados por `torchvision`).
- **Capas entrenables:** `layer4` y la cabeza. `conv1`, `bn1`, `layer1`–`layer3`
  quedan congeladas (`requires_grad = False`).
- **Cabeza:** `capas_ocultas` capas lineales de 256 unidades con ReLU y
  `dropout`, seguidas de `Linear(→ 2)`.
- **Salida:** logits `[batch, 2]`; `class_map.json` = `{0: "cat", 1: "dog"}`.

## 4. Early stopping

| Parámetro | Valor |
|---|---|
| Métrica vigilada | `val_loss` (se minimiza) |
| `patience` | 3 épocas |
| `min_delta` | 0.001 |
| Tope | `epocas_maximas` de cada corrida |
| Al terminar | **siempre** se restauran los pesos de la mejor época |

## 5. Métrica de selección

- **Principal:** `val_accuracy` (mayor es mejor).
- **Desempate:** `val_macro_f1`.
- **Prohibido:** cualquier métrica `test_*`. `select.py` falla si la recibe.

## 6. Semillas

| Uso | Semilla |
|---|---|
| Split del manifiesto | 42 |
| DataLoader (`generator`) | 43 |
| Aumentación | 44 |
| Inicialización de la cabeza | 45 |

## 7. Rejilla de experimentos

Cada uno de los 7 parámetros toma al menos 2 valores y ninguna fila se repite.
Las corridas son sobre el mismo manifiesto (P3-06).

| Corrida | Optimizador | Batch | Épocas máx. | LR | Imagen | Capas ocultas | Dropout |
|---|---|---|---|---|---|---|---|
| r01 | adam | 32 | 10 | 1e-3 | 128 | 0 | 0.0 |
| r02 | adam | 32 | 10 | 3e-4 | 128 | 1 | 0.3 |
| r03 | adam | 16 | 10 | 1e-3 | 160 | 1 | 0.3 |
| r04 | adam | 16 | 15 | 3e-4 | 160 | 0 | 0.3 |
| r05 | sgd | 32 | 15 | 1e-2 | 128 | 1 | 0.0 |
| r06 | sgd | 16 | 10 | 1e-2 | 160 | 0 | 0.3 |
| r07 | sgd | 32 | 15 | 3e-3 | 160 | 1 | 0.3 |
| r08 | adam | 32 | 15 | 1e-4 | 160 | 1 | 0.0 |
| r09 | sgd | 16 | 15 | 1e-2 | 128 | 1 | 0.3 |
| r10 | adam | 16 | 10 | 1e-3 | 128 | 1 | 0.5 |

Valores por parámetro: optimizador {adam, sgd} · batch {16, 32} · épocas
{10, 15} · LR {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} · imagen {128, 160} · capas
ocultas {0, 1} · dropout {0.0, 0.3, 0.5}. SGD usa `momentum = 0.9`.

**Corridas extra r11–r12:** solo si la mejor `val_accuracy` de r01–r10 es menor
que 0.90, y siempre antes de cerrar la selección.

**Presupuesto de tiempo:** si el smoke test (P3-10) mide más de 15 minutos por
corrida, antes de lanzar la campaña se reduce el tamaño de imagen a {112, 128}
en un commit que cite esta sección. No se permite ningún otro cambio.

## 8. MLflow

- Servidor en `docker-compose.yml` (servicio `mlflow`) con almacén en
  `./mlflow-store`.
- El almacén se versiona con DVC (`scripts/snapshot_mlflow.sh`) después de la
  campaña y después de la selección.

## 9. S3 para modelos

- Bucket en la cuenta `222629887955`, con versionado activo.
- Prefijo: `models/clasificador-perro-gato/<semver>/`.
- Nunca se sobrescribe una versión publicada.

## 10. Versiones del modelo

| Versión | Contenido |
|---|---|
| `0.1.0` | Ensayo: la corrida del smoke test, marcada "ensayo, no seleccionada" |
| `1.0.0` | Final: el candidato elegido en P3-11 con las métricas de P3-13 |

## 11. Custodio del test

**Uriel.** Solo él ejecuta `final.py --split test`, una única vez, después de
que Karen declare **MODEL SELECTION CLOSED**.
