# Contratos de API — Frente "Modelo" (P3-03)

Congela la forma de los datos entre `ml-api` (Python/Starlette), el frontend
(React) y `trainer-worker` (Python), para que Uriel, Emilio y Juan Pablo
avancen en paralelo sin esperarse los unos a los otros. Ver
`docs/decisiones-proyecto3.md` para las decisiones de modelado/entrenamiento
que estos contratos exponen, no redefinen.

Convención: **snake_case** en todos los campos JSON, igual que el resto de
los reportes/contratos del repo (`presentation/contracts.py`,
`copilot/contracts.py`, `reports/*.json`) — este proyecto no mezcla
camelCase y snake_case entre payloads.

Fuente de verdad de cada contrato:

- Python (`ml-api`, `trainer-worker`): `app/ml_api/contracts.py` (Pydantic,
  `extra="forbid"`, `strict=True`).
- TypeScript (`frontend`): `frontend/src/model/contracts.ts` (Zod), espejo
  campo a campo del Pydantic — un campo que se agregue de un lado y no del
  otro es un bug, no una libertad de implementación.

## Rutas

Todas bajo `/ml-api/` (nginx, `frontend/docker/nginx.conf`), que reenvía al
servicio `ml-api` (`docker-compose.yml`). El backend Node (`/api/`) no las
expone: es un servicio Python aparte, no una ruta más del backend existente.

| Ruta | Método | Contrato de respuesta | Estado en P3-03 |
|---|---|---|---|
| `/ml-api/health` | GET | `{"status": "ok"}` | Real |
| `/ml-api/training/jobs` | GET | `TrainingJobList` | **Real** (lee `training_jobs`) |
| `/ml-api/experiments` | GET | `PendingEndpoint` | Pendiente — P3-12 |
| `/ml-api/evaluation` | GET | `PendingEndpoint` | Pendiente — P3-13 |
| `/ml-api/models` | GET | `PendingEndpoint` | Pendiente — P3-14 |
| `/ml-api/inference` | GET | `PendingEndpoint` | Pendiente — P3-16 |

Ninguna ruta escribe todavía: `trainer-worker` (P3-09) es quien va a
insertar/actualizar filas de `training_jobs` directo en MariaDB, no a través
de `ml-api` — no hay round-trip HTTP en la escritura del progreso de una
corrida, para no depender de que `ml-api` esté arriba mientras se entrena.

## `TrainingJob`

Refleja uno a uno la tabla `training_jobs`
(`backend/src/data/db/schema.ts`) — misma fuente de verdad en dos lenguajes,
no dos modelos independientes que puedan desincronizarse.

| Campo | Tipo | Fuente | Notas |
|---|---|---|---|
| `id` | `string` | Asignado por quien encola la corrida (ej. `"r01"`, ver la rejilla en `docs/decisiones-proyecto3.md`) | Primary key, no autoincremental |
| `status` | `"queued" \| "running" \| "completed" \| "failed" \| "cancelled"` | `trainer-worker`, al cambiar de fase | — |
| `progress` | `float` (0-1) | `trainer-worker`, fracción de `epocas_maximas` completadas | — |
| `config` | `object` | `trainer-worker`, al encolar — los hiperparámetros de la corrida (optimizador, batch, lr, ...) | Forma exacta la define P3-07 |
| `dataset_release` | `string` | `trainer-worker`, al encolar (ej. `"v0.1.1"`) | Columna `dataset_release`, no `release` — palabra reservada en SQL, confirmado con MariaDB real |
| `manifest_id` | `string` | `trainer-worker`, al encolar — el manifiesto congelado (P3-06) usado | — |
| `mlflow_run_id` | `string \| null` | `trainer-worker`, al crear el run en MLflow | `null` hasta que el run existe |
| `error` | `string \| null` | `trainer-worker`, solo si `status = "failed"` | — |
| `logs` | `string[]` | `trainer-worker`, líneas recientes (no el log completo — eso vive en MLflow) | — |
| `heartbeat_at` | `string \| null` (ISO 8601) | `trainer-worker`, cada vez que sigue vivo | `null` antes de arrancar; sin heartbeat reciente + `status="running"` = corrida colgada |

## `PendingEndpoint`

Respuesta de un área que todavía no tiene datos reales — para que el
frontend reciba un `200` con forma conocida en vez de un `404` sin explicar
por qué, mientras esa pieza no existe.

| Campo | Tipo | Notas |
|---|---|---|
| `status` | `"pending"` | Constante |
| `ticket` | `string` | Qué ticket construye el contrato real (ej. `"P3-12"`) |
| `message` | `string` | Explicación corta para mostrar en la UI |
