/**
 * Light/dark theme, remembered per browser. The source of truth is the
 * `data-theme` attribute on <html>; this module reads and writes it and mirrors
 * the choice into localStorage so it survives a reload.
 *
 * Every storage and document access is wrapped: localStorage throws in a private
 * window or when site data is blocked, and we must never let that break the page.
 */

export type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "hl-theme";

export function readStoredTheme(): Theme {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    if (value === "light" || value === "dark") return value;
  } catch {
    // Storage unavailable (private mode, blocked): fall through to "system".
  }
  return "system";
}

/** Apply a theme now: set or clear the attribute, and remember the choice. */
export function applyTheme(next: Theme): void {
  try {
    const root = document.documentElement;
    if (next === "system") {
      delete root.dataset.theme;
    } else {
      root.dataset.theme = next;
    }
  } catch {
    // No document (shouldn't happen in the browser) — nothing to do.
  }
  try {
    if (next === "system") window.localStorage.removeItem(STORAGE_KEY);
    else window.localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // Storage unavailable: the attribute is still set for this session.
  }
}

/** The theme currently shown, resolving "system" to what the OS prefers. */
export function resolvedTheme(): "light" | "dark" {
  const stored = readStoredTheme();
  if (stored !== "system") return stored;
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}
