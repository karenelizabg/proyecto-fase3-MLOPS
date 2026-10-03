import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EvaluationPage } from "./Evaluation";

/**
 * P3-19: visor navegable de ejemplos (aciertos primero, luego errores) con
 * "Anterior" y "Siguiente". Se mockea `fetch` (nunca se toca ml-api real).
 * Los datos son fixtures: el componente no tiene ninguna cifra escrita a mano.
 */

const VIEWER_NAME = "Explorador de ejemplos";

const SUCCESSES = [
  {
    crop_id: "000008_000001",
    source_image_id: "8",
    true_class: "cat",
    predicted_class: "cat",
    probability: 0.9534333348274231,
  },
  {
    crop_id: "000014_000013",
    source_image_id: "14",
    true_class: "dog",
    predicted_class: "dog",
    probability: 0.8812,
  },
  {
    crop_id: "000020_000021",
    source_image_id: "20",
    true_class: "cat",
    predicted_class: "cat",
    probability: 0.7203,
  },
];

const ERRORS = [
  {
    crop_id: "000275_000320",
    source_image_id: "275",
    true_class: "dog",
    predicted_class: "cat",
    probability: 0.5771472454071045,
  },
  {
    crop_id: "000301_000302",
    source_image_id: "301",
    true_class: "cat",
    predicted_class: "dog",
    probability: 0.6123,
  },
];

const READY = {
  status: "ready",
  run_id: "7e7b4a4b35464cfebb6b41714a3ad931",
  release: "v0.1.1",
  manifest_id: "v0.1.1-53fc84fdaa07",
  manifest_sha256: "5".repeat(64),
  checkpoint_sha256: "f".repeat(64),
  selected_at: "2026-10-02T05:43:20.705345Z",
  grid_row: "r02",
  best_val_accuracy: 0.9618320610687023,
  best_val_macro_f1: 0.9618231625575566,
  best_val_loss: 0.12332657724618912,
  classes: ["cat", "dog"],
  accuracy: 0.9863013698630136,
  macro_f1: 0.9862081995087851,
  confusion_matrix: [
    [39, 0],
    [1, 33],
  ],
  per_class: {
    cat: { precision: 0.975, recall: 1.0, f1: 0.9873417721518987, support: 39 },
    dog: { precision: 1.0, recall: 0.9705882352941176, f1: 0.9850746268656716, support: 34 },
  },
  total: 73,
  baseline_majority_accuracy: 0.5342465753424658,
  most_confused_class: "dog",
  recall_per_class: { cat: 1.0, dog: 0.9705882352941176 },
  accuracy_hides_low_recall: false,
  successes: SUCCESSES,
  errors: ERRORS,
};

function mockEvaluation(payload: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    )
  );
}

/** Coincide con el `<p>` cuyo texto completo (incluido el `<strong>`) es `text`. */
function paragraph(text: string) {
  return (_content: string, element: Element | null) =>
    element?.tagName === "P" && element.textContent === text;
}

async function openViewer() {
  return within(await screen.findByRole("region", { name: VIEWER_NAME }));
}

function click(name: string) {
  fireEvent.click(screen.getByRole("button", { name }));
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("EvaluationPage: navegación de la galería (P3-19)", () => {
  it("abre en el primer acierto con su recorte, clases y probabilidad", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);

    const viewer = await openViewer();
    expect(viewer.getByText("Acierto")).toBeInTheDocument();
    expect(viewer.getByText("1 de 5")).toBeInTheDocument();
    expect(viewer.getByAltText("Recorte seleccionado 000008_000001")).toHaveAttribute(
      "src",
      "/ml-api/crops/000008_000001"
    );
    expect(viewer.getByText(paragraph("real cat"))).toBeInTheDocument();
    expect(viewer.getByText(paragraph("pred cat · 95.34%"))).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Anterior" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Siguiente" })).toBeEnabled();
  });

  it("Siguiente avanza al ejemplo siguiente y actualiza posición, recorte y datos", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);
    await openViewer();
    click("Siguiente");

    const viewer = await openViewer();
    expect(viewer.getByText("2 de 5")).toBeInTheDocument();
    expect(viewer.getByAltText("Recorte seleccionado 000014_000013")).toHaveAttribute(
      "src",
      "/ml-api/crops/000014_000013"
    );
    expect(viewer.getByText(paragraph("real dog"))).toBeInTheDocument();
    expect(viewer.getByText(paragraph("pred dog · 88.12%"))).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Anterior" })).toBeEnabled();
  });

  it("al pasar del último acierto al primer error cambia la etiqueta a Error", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);
    await openViewer();
    click("Siguiente");
    click("Siguiente");
    expect((await openViewer()).getByText("Acierto")).toBeInTheDocument();
    click("Siguiente");

    const viewer = await openViewer();
    expect(viewer.getByText("Error")).toBeInTheDocument();
    expect(viewer.getByText("4 de 5")).toBeInTheDocument();
    expect(viewer.getByAltText("Recorte seleccionado 000275_000320")).toHaveAttribute(
      "src",
      "/ml-api/crops/000275_000320"
    );
    expect(viewer.getByText(paragraph("real dog"))).toBeInTheDocument();
    expect(viewer.getByText(paragraph("pred cat · 57.71%"))).toBeInTheDocument();
  });

  it("Anterior regresa al ejemplo previo", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);
    await openViewer();
    click("Siguiente");
    click("Siguiente");
    click("Anterior");

    const viewer = await openViewer();
    expect(viewer.getByText("2 de 5")).toBeInTheDocument();
    expect(viewer.getByAltText("Recorte seleccionado 000014_000013")).toBeInTheDocument();
  });

  it("en el último ejemplo Siguiente queda deshabilitado y no da la vuelta", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);
    await openViewer();
    for (let step = 0; step < 4; step += 1) click("Siguiente");

    const viewer = await openViewer();
    expect(viewer.getByText("5 de 5")).toBeInTheDocument();
    expect(viewer.getByAltText("Recorte seleccionado 000301_000302")).toBeInTheDocument();
    expect(viewer.getByText(paragraph("real cat"))).toBeInTheDocument();
    expect(viewer.getByText(paragraph("pred dog · 61.23%"))).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Siguiente" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Anterior" })).toBeEnabled();
  });

  it("sin aciertos navega solo por los errores", async () => {
    mockEvaluation({ ...READY, successes: [] });

    render(<EvaluationPage />);

    const viewer = await openViewer();
    expect(viewer.getByText("Error")).toBeInTheDocument();
    expect(viewer.getByText("1 de 2")).toBeInTheDocument();
    expect(viewer.getByAltText("Recorte seleccionado 000275_000320")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Anterior" })).toBeDisabled();
  });

  it("sin aciertos ni errores no muestra el visor", async () => {
    mockEvaluation({ ...READY, successes: [], errors: [] });

    render(<EvaluationPage />);

    await screen.findByText("98.63%");
    expect(screen.queryByRole("region", { name: VIEWER_NAME })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Siguiente" })).not.toBeInTheDocument();
  });
});
