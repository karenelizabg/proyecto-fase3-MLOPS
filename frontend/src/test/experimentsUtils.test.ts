import { describe, expect, it } from "vitest";

import {
  filterCampaign,
  filterValid,
  getMLflowLink,
  isCampaignRun,
  type MLflowRun,
  sortByMetric,
} from "../lib/experimentsUtils";

describe("Lógica de la Tabla de Experimentos (P3-12/P3-15)", () => {
  const campaignRun = (gridRow: string, runId: string): MLflowRun => ({
    info: { run_id: runId, experiment_id: "1" },
    data: { metrics: { best_val_loss: 0.5 }, tags: { run_kind: "campaign", grid_row: gridRow } },
  });

  const smoke: MLflowRun = {
    info: { run_id: "smoke", experiment_id: "1" },
    data: { metrics: { best_val_loss: 0.1 }, tags: { run_kind: "smoke", grid_row: "" } },
  };

  const mockRuns: MLflowRun[] = [
    campaignRun("r01", "abc"),
    smoke,
    campaignRun("r02", "def"),
    { info: { run_id: "ghi", experiment_id: "1" }, data: { metrics: {}, tags: {} } },
  ];

  it("reconoce una corrida de campaña por run_kind o por grid_row", () => {
    expect(isCampaignRun(campaignRun("r10", "x"))).toBe(true);
    expect(isCampaignRun(smoke)).toBe(false);
    expect(
      isCampaignRun({
        info: { run_id: "y", experiment_id: "1" },
        data: { metrics: {}, tags: { grid_row: "r12" } },
      })
    ).toBe(true);
  });

  it("filtra solo las corridas de la campaña (r01–r12), sin el smoke", () => {
    const campaign = filterCampaign(mockRuns);
    expect(campaign.map((run) => run.info.run_id)).toEqual(["abc", "def"]);
  });

  it("filtra por los run_id válidos de reports/experiments_validity.json", () => {
    const valid = filterValid(mockRuns, new Set(["abc", "ghi"]));
    expect(valid.map((run) => run.info.run_id)).toEqual(["abc", "ghi"]);
  });

  it("ordena las corridas por best_val_loss", () => {
    const runs: MLflowRun[] = [
      {
        info: { run_id: "slow", experiment_id: "1" },
        data: { metrics: { best_val_loss: 0.8 }, tags: {} },
      },
      {
        info: { run_id: "fast", experiment_id: "1" },
        data: { metrics: { best_val_loss: 0.2 }, tags: {} },
      },
    ];

    expect(sortByMetric(runs, "best_val_loss", "asc")[0]?.info.run_id).toBe("fast");
    expect(sortByMetric(runs, "best_val_loss", "desc")[0]?.info.run_id).toBe("slow");
  });

  it("construye el enlace profundo exacto a la UI de MLflow", () => {
    expect(getMLflowLink("1", "abc")).toBe("http://localhost:5050/#/experiments/1/runs/abc");
  });
});
