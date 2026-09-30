import type { z } from "zod";

/**
 * Cliente de `ml-api` (P3-03) -- mismo comportamiento que `lib/api/client.ts`
 * (parseo con Zod, `ApiError` con el status real), pero apuntado a
 * `/ml-api`, un servicio Python aparte del backend Node (ver
 * docs/api-contratos.md). No se reusa `lib/api/client.ts` tal cual porque
 * ese módulo fija `/api` como base; duplicar esta lógica simple es más
 * seguro que tocar un archivo compartido por 4 módulos ya existentes para
 * un solo consumidor nuevo.
 */

const configuredBaseUrl = import.meta.env.VITE_ML_API_BASE_URL as string | undefined;

export const ML_API_BASE_URL: string =
  configuredBaseUrl !== undefined && configuredBaseUrl.trim() !== ""
    ? configuredBaseUrl
    : "/ml-api";

export class MlApiError extends Error {
  constructor(
    public readonly status: number,
    message: string
  ) {
    super(message);
    this.name = "MlApiError";
  }
}

function defaultMessageForStatus(status: number): string {
  if (status === 404) return "No se encontró el recurso solicitado.";
  if (status >= 500) return "Error del servidor. Intenta de nuevo.";
  return `Error inesperado (${status}).`;
}

export async function mlApiRequest<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${ML_API_BASE_URL}${path}`, init);
  } catch {
    throw new MlApiError(0, "No se pudo conectar con ml-api.");
  }

  if (!res.ok) {
    throw new MlApiError(res.status, defaultMessageForStatus(res.status));
  }

  const json: unknown = await res.json();
  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    throw new Error(`Respuesta inválida de ml-api en ${path}: ${parsed.error.message}`);
  }
  return parsed.data;
}
