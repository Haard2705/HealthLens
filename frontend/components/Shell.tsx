"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { BrandMark } from "./Brand";
import { useSession } from "./SessionProvider";
import { SAFETY_BANNER } from "@/lib/format";
import type { Role } from "@/lib/types";

interface NavItem {
  href: string;
  label: string;
}

const PUBLIC_NAV: NavItem[] = [
  { href: "/", label: "Home" },
  { href: "/explore", label: "Explore" },
  { href: "/models", label: "Model results" },
];

const NAV_BY_ROLE: Record<Role, NavItem[]> = {
  patient: [
    { href: "/me", label: "My health" },
    { href: "/explore", label: "Explore" },
    { href: "/", label: "Home" },
  ],
  doctor: [
    { href: "/doctor", label: "My patients" },
    { href: "/explore", label: "Explore" },
    { href: "/models", label: "Model results" },
    { href: "/", label: "Home" },
  ],
  admin: [
    { href: "/admin", label: "Users and audit" },
    { href: "/models", label: "Model results" },
    { href: "/explore", label: "Explore" },
    { href: "/", label: "Home" },
  ],
};

function isCurrent(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

export function Shell({ children }: { children: React.ReactNode }) {
  const { user, loading, signOut } = useSession();
  const pathname = usePathname();
  const router = useRouter();
  const nav = user ? NAV_BY_ROLE[user.role] : PUBLIC_NAV;

  async function handleSignOut() {
    await signOut();
    router.push("/");
  }

  return (
    <>
      <header className="top">
        <div className="wrap">
          <Link className="brand" href="/" aria-label="HealthLens home">
            <BrandMark />
            HealthLens
          </Link>
          <div className="who">
            {loading ? (
              <span className="small muted">Checking sign-in…</span>
            ) : user ? (
              <>
                <span className="pill acc">{user.role}</span>
                <span>{user.display_name}</span>
                <button className="btn small" type="button" onClick={handleSignOut}>
                  Sign out
                </button>
              </>
            ) : (
              <Link className="btn small" href="/login">
                Sign in
              </Link>
            )}
          </div>
        </div>
      </header>

      <nav className="nav" aria-label="Sections">
        <div className="wrap">
          {nav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isCurrent(pathname, item.href) ? "page" : undefined}
            >
              {item.label}
            </Link>
          ))}
        </div>
      </nav>

      <div className="banner">
        <div className="wrap">{SAFETY_BANNER}</div>
      </div>

      <main className="wrap">{children}</main>

      <footer className="wrap">
        <div className="row">
          <Link className="btn link small" href="/models">
            Model results and the release gate
          </Link>
          <Link className="btn link small" href="/explore">
            Explore the population
          </Link>
        </div>
        HealthLens is a student project: FastAPI + per-condition models with SHAP explanations,
        read through this Next.js dashboard. Survey numbers come from the CDC BRFSS annual file;
        every patient in the demo is invented.
      </footer>
    </>
  );
}
