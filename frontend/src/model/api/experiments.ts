import { z } from "zod";
import { mlApiRequest } from "./client";

/**
 * Experimento y corridas de MLflow (P3-12/P3-15). La UI resuelve el
 * experimento por nombre (`clasificador-perro-gato`) con `GET /experiments`
 * en vez de fijar el `experiment_id` a mano (ver docs/api-contratos.md).
 */

export const mlflowExperimentSchema = z
  .object({
    experiment_id: z.string().min(1),
    name: z.string().min(1),
    lifecycle_stage: z.string().min(1),
  })
  .passthrough();

export const mlflowRunSchema = z
  .object({
    info: z
      .object({
        run_id: z.string().min(1),
        experiment_id: z.string().min(1),
      })
      .passthrough(),
    data: z
      .object({
        metrics: z.record(z.string(), z.number()),
        tags: z.record(z.string(), z.string()),
      })
      .passthrough(),
  })
  .passthrough();

export const metricHistorySchema = z.array(
  z.object({
    step: z.number(),
    value: z.number(),
    timestamp: z.number(),
  })
);

/** `reports/experiments_validity.json` (P3-11); solo necesitamos `valid`. */
export const experimentsValiditySchema = z
  .object({
    valid: z.array(z.string()),
  })
  .passthrough();

export type MlflowExperiment = z.infer<typeof mlflowExperimentSchema>;
export type MlflowRun = z.infer<typeof mlflowRunSchema>;

export async function listExperiments(signal?: AbortSignal) {
  return mlApiRequest("/experiments", z.array(mlflowExperimentSchema), { signal });
}

export async function listRuns(experimentId: string, signal?: AbortSignal) {
  return mlApiRequest(
    `/experiments/${encodeURIComponent(experimentId)}/runs`,
    z.array(mlflowRunSchema),
    {
      signal,
    }
  );
}

export async function getMetricHistory(runId: string, metricKey: string, signal?: AbortSignal) {
  return mlApiRequest(
    `/experiments/runs/${encodeURIComponent(runId)}/metrics/${encodeURIComponent(metricKey)}`,
    metricHistorySchema,
    { signal }
  );
}
