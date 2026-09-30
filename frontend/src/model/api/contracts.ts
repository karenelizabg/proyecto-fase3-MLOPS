import { z } from "zod";

/**
 * Espejo campo a campo de `app/ml_api/contracts.py` (Pydantic) -- ver
 * docs/api-contratos.md. snake_case a propósito: este repo no mezcla
 * camelCase y snake_case entre payloads (mismo criterio que
 * lib/api/settings.ts).
 */

export const trainingJobStatusSchema = z.enum([
  "queued",
  "running",
  "completed",
  "failed",
  "cancelled",
]);

export const trainingJobSchema = z
  .object({
    id: z.string().min(1),
    status: trainingJobStatusSchema,
    progress: z.number().finite().min(0).max(1),
    config: z.record(z.string(), z.unknown()),
    dataset_release: z.string().min(1),
    manifest_id: z.string().min(1),
    mlflow_run_id: z.string().min(1).nullable(),
    error: z.string().min(1).nullable(),
    logs: z.array(z.string()),
    heartbeat_at: z.string().min(1).nullable(),
  })
  .strict();

export const trainingJobListSchema = z
  .object({
    jobs: z.array(trainingJobSchema),
  })
  .strict();

export const pendingEndpointSchema = z
  .object({
    status: z.literal("pending"),
    ticket: z.string().min(1),
    message: z.string().min(1),
  })
  .strict();

export type TrainingJob = z.infer<typeof trainingJobSchema>;
export type TrainingJobList = z.infer<typeof trainingJobListSchema>;
export type PendingEndpoint = z.infer<typeof pendingEndpointSchema>;
