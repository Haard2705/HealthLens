"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/components/SessionProvider";
import { errorMessage } from "@/lib/api";
import type { Role } from "@/lib/types";

/**
 * The demo password is deliberately public: this is a prototype on synthetic data, and the
 * professor has to be able to sign in without being handed a secret. Override it (and the
 * backend's DEMO_PASSWORD) if you ever point this at anything real.
 */
const DEMO_PASSWORD = process.env.NEXT_PUBLIC_DEMO_PASSWORD ?? "demo-password-123";
const DEMO_DOMAIN = "demo.healthlens.test";

interface DemoAccount {
  role: Role;
  email: string;
  who: string;
  sees: string;
  blurb: string;
}

const DEMO_ACCOUNTS: DemoAccount[] = [
  {
    role: "patient",
    email: `fatima.hassan@${DEMO_DOMAIN}`,
    who: "Patient — Fatima Hassan",
    sees: "own record only",
    blurb: "Her own conditions, medications and trends, in plainer language.",
  },
  {
    role: "doctor",
    email: `dr.chen@${DEMO_DOMAIN}`,
    who: "Doctor — Dr. Wei Chen",
    sees: "assigned patients only",
    blurb: "Six of the eight demo patients, with history, medications and observation trends.",
  },
  {
    role: "admin",
    email: `admin@${DEMO_DOMAIN}`,
    who: "Admin — Demo Admin",
    sees: "every action logged",
    blurb: "Manages users and reads the audit log. Opening a record is allowed, and recorded.",
  },
];

const LANDING: Record<Role, string> = {
  patient: "/me",
  doctor: "/doctor",
  admin: "/admin",
};

export function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { signIn } = useSession();

  const preset = DEMO_ACCOUNTS.find((a) => a.role === params.get("as"));
  const [email, setEmail] = useState(preset?.email ?? "");
  const [password, setPassword] = useState(preset ? DEMO_PASSWORD : "");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(withEmail: string, withPassword: string, tag: string) {
    setBusy(tag);
    setError(null);
    try {
      const user = await signIn(withEmail, withPassword);
      router.push(LANDING[user.role]);
    } catch (caught: unknown) {
      setError(errorMessage(caught));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="signcard stack">
      <div>
        <h1 className="pt">Sign in</h1>
        <p className="muted">
          Pick a demo role to see what that person is allowed to see. The three accounts hold
          synthetic data only.
        </p>
      </div>

      {error && (
        <div className="alert bad" role="alert">
          <strong>Could not sign in.</strong> {error}
        </div>
      )}

      <div className="stack" style={{ gap: 10 }}>
        {DEMO_ACCOUNTS.map((account) => (
          <button
            className="role"
            type="button"
            key={account.role}
            disabled={busy !== null}
            onClick={() => submit(account.email, DEMO_PASSWORD, account.role)}
          >
            <span className="row between">
              <strong>{account.who}</strong>
              <span className="pill acc">{account.sees}</span>
            </span>
            <span className="small muted">{account.blurb}</span>
            <span className="small mono muted">
              {busy === account.role ? "Signing in…" : account.email}
            </span>
          </button>
        ))}
      </div>

      <div className="card flat small">
        <strong>Demo password:</strong> <span className="mono">{DEMO_PASSWORD}</span>. It is printed
        here on purpose — this is a prototype over invented patients, not a real clinical system.
      </div>

      <form
        className="card stack"
        onSubmit={(event) => {
          event.preventDefault();
          void submit(email, password, "form");
        }}
      >
        <h2 style={{ fontSize: 16 }}>Or sign in with an email and password</h2>
        <label className="f">
          Email
          <input
            type="email"
            name="email"
            autoComplete="username"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder={`name@${DEMO_DOMAIN}`}
          />
        </label>
        <label className="f">
          Password
          <input
            type="password"
            name="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        <button
          className="btn primary"
          type="submit"
          disabled={busy !== null}
          style={{ alignSelf: "flex-start" }}
        >
          {busy === "form" ? "Signing in…" : "Sign in"}
        </button>
      </form>

      <p className="small muted">
        Sign-in sets an httpOnly session cookie. Accounts are invite-only: nobody can register
        themselves, and repeated failures are rate-limited.
      </p>
    </div>
  );
}
