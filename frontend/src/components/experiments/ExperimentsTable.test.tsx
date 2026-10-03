import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ExperimentsTable } from "./ExperimentsTable";

/**
 * P3-15 (Ticket C): por defecto la tabla muestra solo las corridas **válidas**
 * (`reports/experiments_validity.json`, P3-11), no las 12 de r01–r12.
 */

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function a_run(index: number) {
  const runId = `run-${String(index).padStart(2, "0")}`.padEnd(32, "0");
  return {
    info: { run_id: runId, experiment_id: "1" },
    data: {
      metrics: { best_val_loss: 0.1 * index, best_val_accuracy: 0.9 },
      tags: { run_kind: "campaign", grid_row: `r${String(index).padStart(2, "0")}` },
    },
  };
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ExperimentsTable (P3-15)", () => {
  it("por defecto muestra solo las corridas válidas de experiments_validity.json", async () => {
    const runs = Array.from({ length: 12 }, (_, index) => a_run(index + 1));
    const valid = runs.slice(0, 10).map((run) => run.info.run_id);
    const invalidRunId = runs[10]?.info.run_id ?? "";

    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("experiments_validity.json")) {
          return Promise.resolve(jsonResponse({ valid, min_required: 10, passed: true }));
        }
        if (url.includes("/experiments/1/runs")) return Promise.resolve(jsonResponse(runs));
        return Promise.resolve(jsonResponse(null, 404));
      })
    );

    render(<ExperimentsTable experimentId="1" />);

    expect(await screen.findByText("10 corridas")).toBeInTheDocument();
    expect(screen.getByText(runs[0]?.info.run_id.slice(0, 8) ?? "")).toBeInTheDocument();
    expect(screen.queryByText(invalidRunId.slice(0, 8))).not.toBeInTheDocument();
  });
});
