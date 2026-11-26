"use client";

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useSession } from "@/components/SessionProvider";

export default function WelcomePage() {
  const router = useRouter();
  const { user, loading } = useSession();

  // Once the session has resolved, a signed-out visitor has nothing to see here.
  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [loading, user, router]);

  if (loading) return <p className="muted small">Loading…</p>;
  if (!user) return null;

  const firstName = user.display_name.split(" ")[0];

  return (
    <div className="signcard stack">
      <div>
        <h1 className="pt">Welcome, {firstName}.</h1>
        <p className="muted">Your account is ready.</p>
      </div>

      <div className="card stack">
        <h2 style={{ fontSize: 16, margin: 0 }}>What&apos;s next</h2>
        <p className="muted" style={{ margin: 0 }}>
          In a moment you&apos;ll answer a few quick health questions — age, weight, whether you
          smoke, that sort of thing. They take about a minute and let HealthLens show you your own
          risk picture in plain language. That questionnaire is coming shortly.
        </p>
        <p className="small muted" style={{ margin: 0 }}>
          Nothing you enter is a diagnosis, and you can explore population patterns any time without
          answering a thing.
        </p>
      </div>

      <div className="row" style={{ gap: 12 }}>
        <Link className="btn primary" href="/explore">
          Explore the data
        </Link>
        <Link className="btn" href="/">
          Go to home
        </Link>
      </div>
    </div>
  );
}
