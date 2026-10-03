import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ExperimentsPage } from "./Experiments";

/**
 * P3-15 (Ticket C): la página resuelve el experimento por nombre. Si no existe
 * `clasificador-perro-gato` (o ml-api no responde) debe mostrar un error, nunca
 * corridas de otro experimento.
 */

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ExperimentsPage (P3-15)", () => {
  it("muestra un error si no existe el experimento clasificador-perro-gato", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse([{ experiment_id: "0", name: "Default", lifecycle_stage: "active" }])
      )
    );

    render(<ExperimentsPage />);

    expect(
      await screen.findByText(
        /No se pudo cargar el experimento «clasificador-perro-gato»: ml-api no responde o el experimento no existe\./
      )
    ).toBeInTheDocument();
  });

  it("monta la tabla del experimento cuando existe", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/experiments")) {
          return Promise.resolve(
            jsonResponse([
              { experiment_id: "1", name: "clasificador-perro-gato", lifecycle_stage: "active" },
            ])
          );
        }
        if (url.includes("/experiments/1/runs")) return Promise.resolve(jsonResponse([]));
        if (url.includes("experiments_validity.json")) {
          return Promise.resolve(jsonResponse({ valid: [] }));
        }
        return Promise.resolve(jsonResponse(null, 404));
      })
    );

    render(<ExperimentsPage />);

    expect(await screen.findByText("0 corridas")).toBeInTheDocument();
  });
});
