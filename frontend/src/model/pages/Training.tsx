import { useEffect, useState } from "react";
import { PageHeader } from "@/pipeline/components/PageHeader";
import type { TrainingJob } from "../api/contracts";
import { getTrainingJobs } from "../api/training";

type LoadState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "success"; jobs: TrainingJob[] };

export function TrainingPage() {
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    let active = true;
    getTrainingJobs()
      .then((result) => {
        if (active) setState({ status: "success", jobs: result.jobs });
      })
      .catch(() => {
        if (active) setState({ status: "error" });
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <PageHeader
          title="Training"
          subtitle="Corridas de la campaña de experimentos. El bucle real de entrenamiento lo escribe P3-09."
        />

        {state.status === "loading" && <p className="text-sm text-ink-muted">Cargando…</p>}
        {state.status === "error" && (
          <p className="text-sm text-ink-muted">No se pudo cargar el estado de las corridas.</p>
        )}
        {state.status === "success" && state.jobs.length === 0 && (
          <p className="text-sm text-ink-muted">Sin corridas todavía.</p>
        )}
        {state.status === "success" && state.jobs.length > 0 && (
          <ul className="flex flex-col gap-2">
            {state.jobs.map((job) => (
              <li key={job.id} className="rounded-2xl border border-border bg-surface p-4 text-sm">
                <span className="font-medium text-ink">{job.id}</span>{" "}
                <span className="text-ink-muted">
                  — {job.status} ({Math.round(job.progress * 100)}%)
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </main>
  );
}
