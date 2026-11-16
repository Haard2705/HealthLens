"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError, errorMessage } from "./api";

export interface Resource<T> {
  data: T | null;
  loading: boolean;
  /** A human-readable message, or null while things are fine. */
  error: string | null;
  /** The raw error, so a caller can branch on 404 vs 501 vs network. */
  cause: ApiError | null;
  reload: () => void;
}

/**
 * Loads one API resource, with a loading and an error state, and cancels in flight when the
 * inputs change or the component unmounts. `load` must be stable (wrap it in useCallback).
 */
export function useResource<T>(load: (signal: AbortSignal) => Promise<T>): Resource<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cause, setCause] = useState<ApiError | null>(null);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    let live = true;
    setLoading(true);
    setError(null);
    setCause(null);
    load(controller.signal)
      .then((result) => {
        if (!live) return;
        setData(result);
      })
      .catch((caught: unknown) => {
        if (!live || controller.signal.aborted) return;
        setData(null);
        setCause(caught instanceof ApiError ? caught : null);
        setError(errorMessage(caught));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
      controller.abort();
    };
  }, [load, nonce]);

  return { data, loading, error, cause, reload };
}
