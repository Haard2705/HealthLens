"use client";

import Link from "next/link";

export function Loading({ what }: { what: string }) {
  return (
    <p className="muted small" role="status">
      Loading {what}…
    </p>
  );
}

interface ErrorNoteProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorNote({ message, onRetry }: ErrorNoteProps) {
  return (
    <div className="alert bad" role="alert">
      <strong>That did not load.</strong> {message}
      {onRetry && (
        <>
          {" "}
          <button className="btn link small" type="button" onClick={onRetry}>
            Try again
          </button>
        </>
      )}
    </div>
  );
}

/**
 * The neutral answer for a record you may not see. The API returns the same 404 whether the
 * record is missing or merely hidden, so this page never says "forbidden" either.
 */
export function RecordNotFound({ backHref, backLabel }: { backHref: string; backLabel: string }) {
  return (
    <div className="stack" style={{ maxWidth: 560 }}>
      <div className="card">
        <span className="pill bad">404</span>
        <h1 className="pt" style={{ marginTop: 10 }}>
          We could not find that record
        </h1>
        <p className="muted" style={{ marginTop: 6 }}>
          The server did not return it. The page does not say whether the record exists, so it
          cannot be used to look people up.
        </p>
        <p className="small" style={{ marginTop: 10 }}>
          The attempt was written to the audit log.
        </p>
        <Link className="btn primary" style={{ marginTop: 14 }} href={backHref}>
          {backLabel}
        </Link>
      </div>
    </div>
  );
}

export function SignInRequired({ what }: { what: string }) {
  return (
    <div className="stack" style={{ maxWidth: 560 }}>
      <div className="card stack">
        <h1 className="pt">Sign in to see {what}</h1>
        <p className="muted">
          This area is behind a sign-in, because it shows a patient record. The demo accounts are
          on the sign-in page.
        </p>
        <Link className="btn primary" href="/login" style={{ alignSelf: "flex-start" }}>
          Go to sign in
        </Link>
      </div>
    </div>
  );
}

export function WrongRole({ expected }: { expected: string }) {
  return (
    <div className="stack" style={{ maxWidth: 560 }}>
      <div className="card stack">
        <h1 className="pt">This area is for a {expected} account</h1>
        <p className="muted">
          Roles change what the API will return, so the page is empty for anyone else. Sign out and
          pick a different demo account to look around.
        </p>
        <Link className="btn" href="/login" style={{ alignSelf: "flex-start" }}>
          Switch account
        </Link>
      </div>
    </div>
  );
}

/** Shown where the risk endpoints (H22, built on another branch) are not available yet. */
export function RiskUnavailable({ detail }: { detail?: string }) {
  return (
    <div className="card stack" style={{ gap: 8 }}>
      <h3>Risk estimates are not available yet</h3>
      <p className="muted small">
        {detail ??
          "This build has no personal-risk endpoint, so nothing is estimated here. Population numbers for every condition are on the Explore page, and the models behind them are on Model results."}
      </p>
      <div className="row">
        <Link className="btn small" href="/explore">
          Open Explore
        </Link>
        <Link className="btn small" href="/models">
          Model results
        </Link>
      </div>
    </div>
  );
}
