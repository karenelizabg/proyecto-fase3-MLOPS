export type MLflowRun = {
  info: { run_id: string; experiment_id: string };
  data: {
    metrics: Record<string, number>;
    tags: Record<string, string>;
  };
};

export const filterValidRuns = (runs: MLflowRun[]): MLflowRun[] => {
  return runs.filter((run) => run.data.tags?.estado === "valida");
};

export const sortByMetric = (
  runs: MLflowRun[],
  metric: string,
  order: "asc" | "desc"
): MLflowRun[] => {
  return [...runs].sort((a, b) => {
    const valA = a.data.metrics?.[metric] ?? 0;
    const valB = b.data.metrics?.[metric] ?? 0;

    return order === "asc" ? valA - valB : valB - valA;
  });
};

export const getMLflowLink = (experimentId: string, runId: string): string => {
  return `http://localhost:5000/#/experiments/${experimentId}/runs/${runId}`;
};
