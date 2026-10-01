import { mlApiRequest } from "./client";
import type { CreateTrainingJobRequest } from "./contracts";
import { trainingJobListSchema, trainingJobSchema } from "./contracts";

export async function getTrainingJobs(signal?: AbortSignal) {
  return mlApiRequest("/training/jobs", trainingJobListSchema, { signal });
}

export async function createTrainingJob(request: CreateTrainingJobRequest) {
  return mlApiRequest("/training/jobs", trainingJobSchema, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}
