"use client";

import { useEffect, useState } from "react";
import { applyTheme, resolvedTheme } from "@/lib/theme";

/**
 * A single button that flips between light and dark. It reads the theme the
 * no-flash script in <head> already applied, so there is no mismatch on first paint.
 */
export function ThemeToggle() {
  const [mode, setMode] = useState<"light" | "dark">("light");

  // Resolve on mount only: the server has no window, so read after hydration.
  useEffect(() => {
    setMode(resolvedTheme());
  }, []);

  function toggle() {
    const next = mode === "dark" ? "light" : "dark";
    applyTheme(next);
    setMode(next);
  }

  const goingDark = mode === "light";
  return (
    <button
      type="button"
      className="btn small theme-toggle"
      onClick={toggle}
      aria-label={goingDark ? "Switch to dark theme" : "Switch to light theme"}
      title={goingDark ? "Dark theme" : "Light theme"}
    >
      {goingDark ? "☾" : "☀"}
    </button>
  );
}
