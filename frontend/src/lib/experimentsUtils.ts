export type MLflowRun = {
  info: { run_id: string; experiment_id: string };
  data: {
    metrics: Record<string, number>;
    tags: Record<string, string>;
  };
};

const GRID_ROW = /^r(0[1-9]|1[0-2])$/;

/** Una corrida de la campaña (r01–r12), no el smoke test ni la evaluación. */
export function isCampaignRun(run: MLflowRun): boolean {
  const row = run.data.tags?.grid_row ?? "";
  return run.data.tags?.run_kind === "campaign" || GRID_ROW.test(row);
}

export function filterCampaign(runs: MLflowRun[]): MLflowRun[] {
  return runs.filter(isCampaignRun);
}

/** Se queda con las corridas válidas según `reports/experiments_validity.json` (P3-11). */
export function filterValid(runs: MLflowRun[], validRunIds: ReadonlySet<string>): MLflowRun[] {
  return runs.filter((run) => validRunIds.has(run.info.run_id));
}

export function sortByMetric(
  runs: MLflowRun[],
  metric: string,
  order: "asc" | "desc"
): MLflowRun[] {
  return [...runs].sort((a, b) => {
    const valA = a.data.metrics?.[metric] ?? 0;
    const valB = b.data.metrics?.[metric] ?? 0;

    return order === "asc" ? valA - valB : valB - valA;
  });
}

export function getMLflowLink(experimentId: string, runId: string): string {
  const base =
    (import.meta.env.VITE_MLFLOW_URL as string | undefined)?.trim() ||
    `${window.location.protocol}//${window.location.hostname}:5050`;
  return `${base}/#/experiments/${experimentId}/runs/${runId}`;
}

export const EXPERIMENT_NAME = "clasificador-perro-gato";
