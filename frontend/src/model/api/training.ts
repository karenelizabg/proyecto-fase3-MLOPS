import { mlApiRequest } from "./client";
import { trainingJobListSchema } from "./contracts";

export async function getTrainingJobs(signal?: AbortSignal) {
  return mlApiRequest("/training/jobs", trainingJobListSchema, { signal });
}
