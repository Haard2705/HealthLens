import { Suspense } from "react";
import { LoginForm } from "./LoginForm";

export const metadata = { title: "Sign in · HealthLens" };

export default function LoginPage() {
  return (
    <Suspense fallback={<p className="muted small">Loading the sign-in page…</p>}>
      <LoginForm />
    </Suspense>
  );
}
