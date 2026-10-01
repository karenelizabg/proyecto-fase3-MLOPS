import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { TrainingPage } from "../src/model/pages/Training";
import failedQuality from "./fixtures/quality.json";

/**
 * SPEC-P3-09-TRAINING: el formulario de lanzamiento (checklist punto 3 del
 * ticket #15). Las dos pruebas que el propio ticket exige explícitamente:
 * "con un valor inválido no se llama a la API" y "con v0.1.0 el botón queda
 * bloqueado" (porque v0.1.0 está failed en la compuerta).
 */

const passedQuality = { ...failedQuality, dataset_version: "v0.1.1", status: "passed" as const };

const versions = {
  schema_version: "1.0",
  releases: [
    {
      dataset_version: "v0.1.1",
      quality_file: "releases/v0.1.1/quality.json",
      splits_file: "releases/v0.1.1/splits.json",
    },
    {
      dataset_version: "v0.1.0",
      quality_file: "releases/v0.1.0/quality.json",
      splits_file: "releases/v0.1.0/splits.json",
    },
  ],
};

const manifestMeta = (release: string) => ({
  manifest_id: `${release}-abc123`,
  manifest_sha256: "deadbeef",
  release: { name: release },
  seed: 42,
  classes: { "0": "cat", "1": "dog" },
  rows: 668,
});

const manifestCounts = {
  release: "v0.1.1",
  totals: {
    crops: 668,
    originals: 600,
    classes: { cat: { crops: 343, originals: 301 }, dog: { crops: 325, originals: 300 } },
  },
  splits: {
    train: {
      target_fraction: 0.7,
      crops: 464,
      originals: 420,
      crops_fraction: 0.69,
      deviation_pp: -0.5,
      classes: { cat: { crops: 237, originals: 210 }, dog: { crops: 227, originals: 210 } },
    },
  },
};

function response(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), { status });
}

function serve() {
  const fetcher = vi.fn((url: string, init?: RequestInit) => {
    if (url === "/reports/versions.json") return Promise.resolve(response(versions));
    if (url === "/reports/releases/v0.1.1/quality.json")
      return Promise.resolve(response(passedQuality));
    if (url === "/reports/releases/v0.1.0/quality.json")
      return Promise.resolve(response(failedQuality));
    if (url === "/reports/manifests/v0.1.1/manifest_meta.json")
      return Promise.resolve(response(manifestMeta("v0.1.1")));
    if (url === "/reports/manifests/v0.1.0/manifest_meta.json")
      return Promise.resolve(response(manifestMeta("v0.1.0")));
    if (url.startsWith("/reports/manifests/")) return Promise.resolve(response(manifestCounts));
    if (url === "/ml-api/training/jobs" && init?.method === "POST") {
      return Promise.resolve(
        response({
          id: "new-job",
          status: "queued",
          progress: 0,
          config: JSON.parse(String(init.body)).config,
          dataset_release: "v0.1.1",
          manifest_id: "v0.1.1-abc123",
          run_kind: "smoke",
          grid_row: null,
          mlflow_run_id: null,
          error: null,
          logs: [],
          heartbeat_at: null,
        })
      );
    }
    if (url === "/ml-api/training/jobs") return Promise.resolve(response({ jobs: [] }));
    return Promise.resolve(response(null, 404));
  });
  vi.stubGlobal("fetch", fetcher);
  return fetcher;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function selectRelease(version: string) {
  const select = await screen.findByLabelText("Release");
  fireEvent.change(select, { target: { value: version } });
}

it("con un valor inválido no se llama a la API", async () => {
  const fetcher = serve();
  render(<TrainingPage />);
  await selectRelease("v0.1.1");

  const learningRate = await screen.findByLabelText("Learning rate");
  fireEvent.change(learningRate, { target: { value: "1" } }); // fuera de [1e-4, 1e-2]

  const postCallsBefore = fetcher.mock.calls.filter(([, init]) => init?.method === "POST").length;
  fireEvent.click(screen.getByRole("button", { name: "Lanzar entrenamiento" }));

  expect(await screen.findByText(/Revisa los parámetros/)).toBeInTheDocument();
  const postCallsAfter = fetcher.mock.calls.filter(([, init]) => init?.method === "POST").length;
  expect(postCallsAfter).toBe(postCallsBefore);
});

it("con v0.1.0 el botón queda bloqueado", async () => {
  serve();
  render(<TrainingPage />);
  await selectRelease("v0.1.0");

  const button = await screen.findByRole("button", { name: "Lanzar entrenamiento" });
  expect((await screen.findAllByText("failed")).length).toBeGreaterThan(0);
  expect(button).toBeDisabled();
});

it("con un valor válido sí llama a la API y refresca la lista", async () => {
  const fetcher = serve();
  render(<TrainingPage />);
  await selectRelease("v0.1.1");
  await screen.findByText("passed");

  fireEvent.click(screen.getByRole("button", { name: "Lanzar entrenamiento" }));

  await waitFor(() => {
    expect(fetcher.mock.calls.filter(([, init]) => init?.method === "POST")).toHaveLength(1);
  });
  expect(await screen.findByText("Sin corridas todavía.")).toBeInTheDocument();
});
