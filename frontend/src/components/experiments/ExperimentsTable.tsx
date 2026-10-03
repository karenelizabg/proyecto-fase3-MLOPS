import { useEffect, useMemo, useState } from "react";
import { experimentsValiditySchema, listRuns } from "@/model/api/experiments";
import {
  filterCampaign,
  filterValid,
  getMLflowLink,
  type MLflowRun,
  sortByMetric,
} from "../../lib/experimentsUtils";
import { ExperimentCurves } from "./ExperimentCurves";

type ValidityReport = { valid: string[] };

function validityButtonLabel(
  onlyValid: boolean,
  validity: ValidityReport | null,
  failed: boolean
): string {
  if (!onlyValid) return "Solo válidas (P3-11)";
  if (validity === null) {
    return failed ? "Solo válidas (P3-11): no disponible" : "Cargando corridas válidas…";
  }
  return "Mostrar todas";
}

function emptyMessage(
  onlyValid: boolean,
  validity: ValidityReport | null,
  failed: boolean
): string {
  if (onlyValid && validity === null) {
    return failed
      ? "No se pudo cargar la lista de corridas válidas."
      : "Cargando corridas válidas…";
  }
  return "Sin corridas para mostrar.";
}

/**
 * Tabla de la campaña (P3-12/P3-15): las 10 corridas r01–r12 con filtros
 * (campaña / válidas), orden por métrica, curvas por corrida y enlace al run
 * en MLflow. El `experiment_id` llega resuelto por nombre desde la página.
 */
export const ExperimentsTable = ({ experimentId }: Readonly<{ experimentId: string }>) => {
  const [runs, setRuns] = useState<MLflowRun[]>([]);
  const [validity, setValidity] = useState<ValidityReport | null>(null);
  const [validityFailed, setValidityFailed] = useState(false);
  const [onlyCampaign, setOnlyCampaign] = useState(true);
  const [onlyValid, setOnlyValid] = useState(true);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    listRuns(experimentId, controller.signal)
      .then((data) => setRuns(data))
      .catch(() => setRuns([]));
    return () => controller.abort();
  }, [experimentId]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/reports/experiments_validity.json", { signal: controller.signal })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error("sin reporte"))))
      .then((report) => {
        const parsed = experimentsValiditySchema.safeParse(report);
        if (parsed.success) setValidity(parsed.data);
        else setValidityFailed(true);
      })
      .catch(() => setValidityFailed(true));
    return () => controller.abort();
  }, []);

  const displayedRuns = useMemo(() => {
    let processed = runs;
    if (onlyCampaign) processed = filterCampaign(processed);
    if (onlyValid) {
      // Sin la lista de válidas no se muestran corridas: enseñar todas con la
      // etiqueta "solo válidas" sería engañoso.
      if (validity === null) return [];
      processed = filterValid(processed, new Set(validity.valid));
    }
    return sortByMetric(processed, "best_val_loss", sortOrder);
  }, [runs, onlyCampaign, onlyValid, validity, sortOrder]);

  return (
    <div className="p-4">
      <div className="mb-4 flex flex-wrap items-center gap-4">
        <button
          type="button"
          onClick={() => setOnlyCampaign((value) => !value)}
          className="rounded border px-4 py-2 text-sm"
        >
          {onlyCampaign ? "Mostrar todas" : "Solo campaña (r01–r12)"}
        </button>
        <button
          type="button"
          onClick={() => setOnlyValid((value) => !value)}
          className="rounded border px-4 py-2 text-sm"
          disabled={validity === null}
        >
          {validityButtonLabel(onlyValid, validity, validityFailed)}
        </button>
        <button
          type="button"
          onClick={() => setSortOrder((order) => (order === "asc" ? "desc" : "asc"))}
          className="rounded border px-4 py-2 text-sm"
        >
          best_val_loss: {sortOrder.toUpperCase()}
        </button>
        <span className="text-sm text-ink-muted">{displayedRuns.length} corridas</span>
      </div>

      <table className="min-w-full border-collapse border text-sm">
        <thead>
          <tr className="bg-gray-100">
            <th className="border p-2">Run ID</th>
            <th className="border p-2">grid_row</th>
            <th className="border p-2">best_val_accuracy</th>
            <th className="border p-2">best_val_loss</th>
            <th className="border p-2">Curvas</th>
          </tr>
        </thead>
        <tbody>
          {displayedRuns.map((run) => (
            <tr key={run.info.run_id} className="text-center">
              <td className="border p-2 font-mono text-xs">
                <a
                  href={getMLflowLink(experimentId, run.info.run_id)}
                  target="_blank"
                  rel="noreferrer"
                  className="font-bold text-blue-600 hover:underline"
                >
                  {run.info.run_id.slice(0, 8)}
                </a>
              </td>
              <td className="border p-2">{run.data.tags?.grid_row || "—"}</td>
              <td className="border p-2">
                {run.data.metrics?.best_val_accuracy?.toFixed(4) ?? "N/A"}
              </td>
              <td className="border p-2">{run.data.metrics?.best_val_loss?.toFixed(4) ?? "N/A"}</td>
              <td className="border p-2">
                <button
                  type="button"
                  onClick={() => setSelectedRunId(run.info.run_id)}
                  className="rounded border px-3 py-1 text-xs text-blue-600"
                >
                  Ver curvas
                </button>
              </td>
            </tr>
          ))}
          {displayedRuns.length === 0 && (
            <tr>
              <td className="border p-3 text-center text-ink-muted" colSpan={5}>
                {emptyMessage(onlyValid, validity, validityFailed)}
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {selectedRunId && (
        <div className="mt-8 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <ExperimentCurves runId={selectedRunId} metricKey="val_loss" />
          <ExperimentCurves runId={selectedRunId} metricKey="train_loss" />
        </div>
      )}
    </div>
  );
};
