/**
 * The single place the browser talks to the API.
 *
 * Calls are always same-origin (`/api/v1/...`) and proxied to FastAPI by the rewrite in
 * `next.config.ts`, so the httpOnly `hl_session` cookie stays first-party. Every error is
 * normalised into an `ApiError` carrying the backend's `{"error":{"code","message"}}`.
 */

const BASE = "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }

  /** A record that is missing, or hidden from this user: both answer 404 identically. */
  get isNotFound(): boolean {
    return this.status === 404;
  }

  get isUnauthenticated(): boolean {
    return this.status === 401;
  }

  /** The endpoint itself does not exist yet (a route a parallel branch still owns). */
  get isUnimplemented(): boolean {
    return this.status === 404 || this.status === 501;
  }
}

function isErrorEnvelope(body: unknown): body is { error: { code?: string; message?: string } } {
  return typeof body === "object" && body !== null && "error" in body;
}

async function readError(response: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `Request failed (${response.status}).`;
  try {
    const body: unknown = await response.json();
    if (isErrorEnvelope(body)) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
    }
  } catch {
    // A non-JSON body (a proxy error page, say) keeps the generic message above.
  }
  return new ApiError(response.status, code, message);
}

export interface QueryParams {
  [key: string]: string | number | boolean | undefined | null;
}

function withQuery(path: string, params?: QueryParams): string {
  if (!params) return `${BASE}${path}`;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  }
  const query = search.toString();
  return query ? `${BASE}${path}?${query}` : `${BASE}${path}`;
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      credentials: "include",
      cache: "no-store",
      ...init,
    });
  } catch {
    throw new ApiError(0, "network_error", "Could not reach the API. Is the backend running?");
  }
  if (!response.ok) throw await readError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function apiGet<T>(path: string, params?: QueryParams, signal?: AbortSignal): Promise<T> {
  return request<T>(withQuery(path, params), { method: "GET", signal });
}

export function apiSend<T>(
  method: "POST" | "PATCH",
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  return request<T>(withQuery(path), {
    method,
    signal,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}
