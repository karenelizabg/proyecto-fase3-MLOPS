import { useEffect, useState } from "react";
import { ExperimentsTable } from "@/components/experiments/ExperimentsTable";
import { EXPERIMENT_NAME } from "@/lib/experimentsUtils";
import { listExperiments } from "../api/experiments";

/**
 * P3-12/P3-15: resuelve el experimento `clasificador-perro-gato` por nombre
 * (`GET /ml-api/experiments`) en vez de fijar el `experiment_id` a mano.
 */
export function ExperimentsPage() {
  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    listExperiments(controller.signal)
      .then((experiments) => {
        const match = experiments.find((experiment) => experiment.name === EXPERIMENT_NAME);
        if (match) setExperimentId(match.experiment_id);
        else setFailed(true);
      })
      .catch(() => setFailed(true));
    return () => controller.abort();
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Experimentos de MLflow en vivo</h1>
      {failed && (
        <p className="text-sm text-ink-muted">
          No se pudo cargar el experimento «{EXPERIMENT_NAME}»: ml-api no responde o el experimento
          no existe.
        </p>
      )}
      {experimentId && <ExperimentsTable experimentId={experimentId} />}
    </div>
  );
}
