import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EvaluationPage } from "./Evaluation";

/**
 * P3-15: la página refleja el candado de P3-11 y, cuando la selección está
 * cerrada y existen los reportes de P3-13, muestra las cifras reales y la
 * galería de recortes. Se mockea `fetch` (nunca se toca ml-api real).
 */

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
  successes: [
    {
      crop_id: "000008_000001",
      source_image_id: "8",
      true_class: "cat",
      predicted_class: "cat",
      probability: 0.9534333348274231,
    },
  ],
  errors: [
    {
      crop_id: "000275_000320",
      source_image_id: "275",
      true_class: "dog",
      predicted_class: "cat",
      probability: 0.5771472454071045,
    },
  ],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("EvaluationPage (P3-15)", () => {
  it("muestra el candado de P3-11 cuando la selección no está cerrada", async () => {
    mockEvaluation({
      status: "selection_not_closed",
      ticket: "P3-11",
      message: "selección no cerrada (no existe reports/selection.json)",
    });

    render(<EvaluationPage />);

    expect(await screen.findByText("Selección no cerrada")).toBeInTheDocument();
    expect(
      screen.getByText("selección no cerrada (no existe reports/selection.json)")
    ).toBeInTheDocument();
  });

  it("pinta las cifras reales y la galería con el recorte servido por ml-api", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);

    expect(await screen.findByText("98.63%")).toBeInTheDocument();
    expect(screen.getByText("72 / 73 recortes")).toBeInTheDocument();

    const crop = await screen.findByAltText("Recorte 000275_000320");
    expect(crop).toHaveAttribute("src", "/ml-api/crops/000275_000320");

    expect(screen.getByRole("link", { name: "Descargar predictions.csv" })).toHaveAttribute(
      "href",
      "/reports/evaluation/predictions.csv"
    );

    await waitFor(() =>
      expect(fetch).toHaveBeenCalledWith("/ml-api/evaluation", expect.anything())
    );
  });
});

/**
 * Pruebas de caracterización (P3-19): fijan lo que la página ya hacía tras P3-15.
 * No son un ciclo TDD; la navegación de la galería se prueba aparte.
 */
function cardOf(label: string): HTMLElement {
  const card = screen.getByText(label).parentElement;
  if (!card) throw new Error(`No hay tarjeta para ${label}`);
  return card;
}

/** Coincide con el `<p>` cuyo texto completo (incluido el `<strong>`) es `text`. */
function paragraph(text: string) {
  return (_content: string, element: Element | null) =>
    element?.tagName === "P" && element.textContent === text;
}

function listItemOf(altText: string): HTMLElement {
  const item = screen.getByAltText(altText).closest("li");
  if (!item) throw new Error(`No hay elemento de lista para ${altText}`);
  return item;
}

describe("EvaluationPage: caracterización (P3-19)", () => {
  it("con la selección no cerrada no muestra ninguna cifra", async () => {
    mockEvaluation({
      status: "selection_not_closed",
      ticket: "P3-11",
      message: "selección no cerrada (no existe reports/selection.json)",
    });

    render(<EvaluationPage />);

    await screen.findByText("Selección no cerrada");
    expect(screen.queryByText("98.63%")).not.toBeInTheDocument();
    expect(screen.queryByText(/72 \/ 73/)).not.toBeInTheDocument();
    expect(screen.queryByText("F1 macro")).not.toBeInTheDocument();
    expect(screen.queryByText("53.42%")).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("con la evaluación pendiente tampoco muestra cifras", async () => {
    mockEvaluation({
      status: "pending",
      ticket: "P3-13",
      message: "la evaluación final todavía no corrió",
    });

    render(<EvaluationPage />);

    await screen.findByText(/la evaluación final todavía no corrió/);
    expect(screen.queryByText("98.63%")).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("muestra el candidato r02 y la versión de datos", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);

    expect(await screen.findByText("r02 · 7e7b4a4b")).toBeInTheDocument();
    expect(screen.getByText("v0.1.1")).toBeInTheDocument();
    expect(screen.getByText("v0.1.1-53fc84fdaa07")).toBeInTheDocument();
  });

  it("muestra accuracy con su conteo, F1 macro y baseline", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);

    await screen.findByText("98.63%");
    const accuracy = cardOf("Accuracy (test)");
    expect(within(accuracy).getByText("98.63%")).toBeInTheDocument();
    expect(within(accuracy).getByText("72 / 73 recortes")).toBeInTheDocument();
    expect(within(cardOf("F1 macro")).getByText("0.9862")).toBeInTheDocument();
    expect(within(cardOf("Baseline clase mayoritaria")).getByText("53.42%")).toBeInTheDocument();
  });

  it("calcula el conteo N / total con la accuracy sin redondear", async () => {
    // 0.84996 se muestra como 85.00%, pero 85.00% de 100000 sería 85000.
    mockEvaluation({ ...READY, accuracy: 0.84996, total: 100000 });

    render(<EvaluationPage />);

    expect(await screen.findByText("85.00%")).toBeInTheDocument();
    expect(screen.getByText("84996 / 100000 recortes")).toBeInTheDocument();
  });

  it("pinta la tabla de métricas por clase", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);

    expect(
      await screen.findByRole("row", { name: "cat 0.9750 1.0000 0.9873 39" })
    ).toBeInTheDocument();
    expect(screen.getByRole("row", { name: "dog 1.0000 0.9706 0.9851 34" })).toBeInTheDocument();
  });

  it("pinta la matriz de confusión con filas reales y columnas predichas", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);

    expect(await screen.findByRole("row", { name: "pred. cat pred. dog" })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: "real cat 39 0" })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: "real dog 1 33" })).toBeInTheDocument();
  });

  it("la galería muestra clase real, predicha y probabilidad de cada ejemplo", async () => {
    mockEvaluation(READY);

    render(<EvaluationPage />);

    await screen.findByText("98.63%");
    const error = within(listItemOf("Recorte 000275_000320"));
    expect(error.getByText(paragraph("real dog"))).toBeInTheDocument();
    expect(error.getByText(paragraph("pred cat · 57.71%"))).toBeInTheDocument();
    const success = within(listItemOf("Recorte 000008_000001"));
    expect(success.getByText(paragraph("real cat"))).toBeInTheDocument();
    expect(success.getByText(paragraph("pred cat · 95.34%"))).toBeInTheDocument();
  });

  it("avisa cuando no hay ejemplos de error en el test", async () => {
    mockEvaluation({ ...READY, errors: [] });

    render(<EvaluationPage />);

    expect(await screen.findByText("Sin ejemplos en el test.")).toBeInTheDocument();
    expect(screen.queryByAltText("Recorte 000275_000320")).not.toBeInTheDocument();
  });
});
