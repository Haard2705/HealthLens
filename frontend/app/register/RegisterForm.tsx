"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "@/components/SessionProvider";
import { errorMessage } from "@/lib/api";

const MIN_PASSWORD = 8;

export function RegisterForm() {
  const router = useRouter();
  const { signUp } = useSession();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (password.length < MIN_PASSWORD) {
      setError(`Use at least ${MIN_PASSWORD} characters for your password.`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await signUp(fullName.trim(), email, password);
      router.push("/welcome");
    } catch (caught: unknown) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="signcard stack">
      <div>
        <h1 className="pt">Create your account</h1>
        <p className="muted">
          Takes a minute. No medical details yet — you&apos;ll answer a few quick health questions
          after this.
        </p>
      </div>

      {error && (
        <div className="alert bad" role="alert">
          <strong>Could not create your account.</strong> {error}
        </div>
      )}

      <form className="card stack" onSubmit={submit}>
        <label className="f">
          Full name
          <input
            type="text"
            name="full_name"
            autoComplete="name"
            required
            maxLength={120}
            value={fullName}
            onChange={(event) => setFullName(event.target.value)}
            placeholder="Jordan Avery"
          />
        </label>
        <label className="f">
          Email
          <input
            type="email"
            name="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@email.com"
          />
        </label>
        <label className="f">
          Password
          <input
            type="password"
            name="password"
            autoComplete="new-password"
            required
            minLength={MIN_PASSWORD}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder={`At least ${MIN_PASSWORD} characters`}
          />
        </label>
        <button
          className="btn primary"
          type="submit"
          disabled={busy}
          style={{ alignSelf: "flex-start" }}
        >
          {busy ? "Creating…" : "Create account"}
        </button>
      </form>

      <p className="small muted">
        Already have an account? <Link href="/login">Sign in</Link>.
      </p>
    </div>
  );
}
