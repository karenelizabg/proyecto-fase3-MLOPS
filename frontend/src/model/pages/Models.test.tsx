import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ModelsPage } from "./Models";

/**
 * P3-15: la página lista las versiones del `registry.json` de P3-14 con su
 * estado S3, separa "versión del dataset" de "versión del modelo", muestra la
 * tarjeta + URL prefirmada y refleja el rechazo del backend al marcar una
 * versión activa sin objeto en S3. `fetch` va mockeado; no se toca ml-api real.
 */

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const PENDING = {
  status: "pending",
  ticket: "P3-14",
  message: "El catálogo de modelos todavía no existe; lo publica P3-14.",
};

function summary(overrides: Record<string, unknown> = {}) {
  return {
    version: "0.1.0",
    dataset_version: "v0.1.0",
    run_id: "smoke-run-1234",
    release: "v0.1.0",
    manifest_id: "v0.1.0-aaaa",
    published_at: "2026-09-29T10:00:00Z",
    selected: false,
    active: false,
    s3_status: { exists: true, version_id: "v-1", size_bytes: 1024, last_modified: null },
    ...overrides,
  };
}

function detail(overrides: Record<string, unknown> = {}) {
  return {
    status: "ready",
    model_name: "clasificador-perro-gato",
    version: "1.0.0",
    dataset_version: "v0.1.1",
    run_id: "7e7b4a4b35464cfebb6b41714a3ad931",
    release: "v0.1.1",
    manifest_id: "v0.1.1-53fc84fdaa07",
    manifest_sha256: "d".repeat(64),
    checkpoint_sha256: "e".repeat(64),
    package_sha256: "f".repeat(64),
    s3_bucket: "mlops-p3-models",
    s3_key: "models/clasificador-perro-gato/1.0.0/",
    s3_version_id: "v-final",
    published_at: "2026-09-30T18:00:00Z",
    selected: true,
    active: false,
    card: "Modelo final, candidato r02.",
    download_url: "https://signed.example/models/clasificador-perro-gato/1.0.0/",
    s3_status: { exists: true, version_id: "v-final", size_bytes: 2048, last_modified: null },
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ModelsPage (P3-15)", () => {
  it("muestra que no hay versiones publicadas mientras falta el registry de P3-14", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(PENDING)));

    render(<ModelsPage />);

    expect(await screen.findByText("Sin versiones publicadas")).toBeInTheDocument();
    expect(screen.getByText(PENDING.message)).toBeInTheDocument();
  });

  it("separa la versión del dataset de la del modelo y marca la activa", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          status: "ready",
          model_name: "clasificador-perro-gato",
          active_version: "1.0.0",
          versions: [
            summary(),
            summary({
              version: "1.0.0",
              dataset_version: "v0.1.1",
              release: "v0.1.1",
              selected: true,
              active: true,
              s3_status: {
                exists: true,
                version_id: "v-final",
                size_bytes: 2048,
                last_modified: null,
              },
            }),
          ],
        })
      )
    );

    render(<ModelsPage />);

    expect(await screen.findByText(/versión activa: 1.0.0/)).toBeInTheDocument();
    expect(screen.getByText("dataset v0.1.0 · release v0.1.0")).toBeInTheDocument();
    expect(screen.getByText("dataset v0.1.1 · release v0.1.1")).toBeInTheDocument();
    expect(screen.getAllByText("activa").length).toBeGreaterThan(0);
    expect(screen.getByText("candidato")).toBeInTheDocument();
  });

  it("refleja el rechazo del backend al marcar una versión activa sin objeto en S3", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/models/active")) {
        expect(init?.method).toBe("POST");
        return Promise.resolve(
          jsonResponse(
            { error: "el objeto de la versión 1.0.0 no existe en S3; no se marca como activa" },
            400
          )
        );
      }
      if (url.endsWith("/models/1.0.0")) {
        return Promise.resolve(jsonResponse(detail()));
      }
      return Promise.resolve(
        jsonResponse({
          status: "ready",
          model_name: "clasificador-perro-gato",
          active_version: null,
          versions: [summary({ version: "1.0.0", dataset_version: "v0.1.1", selected: true })],
        })
      );
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<ModelsPage />);

    fireEvent.click(await screen.findByText("1.0.0"));
    const activate = await screen.findByRole("button", { name: "Marcar como activa" });
    fireEvent.click(activate);

    expect(await screen.findByText(/no existe en S3; no se marca como activa/)).toBeInTheDocument();
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/ml-api/models/active",
        expect.objectContaining({ method: "POST" })
      )
    );
  });
});
