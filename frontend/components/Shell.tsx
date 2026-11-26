"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { BrandMark } from "./Brand";
import { useSession } from "./SessionProvider";
import { ThemeToggle } from "./ThemeToggle";
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
  base: [
    { href: "/welcome", label: "My health" },
    { href: "/explore", label: "Explore" },
    { href: "/", label: "Home" },
  ],
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
  super_admin: [
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

/**
 * Routes that can put a risk number or a patient record in front of someone. The safety
 * notice belongs there and nowhere else: repeated on the home page and on aggregate
 * population charts it is just noise, and a disclaimer people have learned to skip is
 * worse than one they read.
 */
const RISK_ROUTES = ["/me", "/doctor", "/models"];

function needsSafetyNotice(pathname: string): boolean {
  return RISK_ROUTES.some((route) => isCurrent(pathname, route));
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
            <ThemeToggle />
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

      {needsSafetyNotice(pathname) ? (
        <div className="banner">
          <div className="wrap">{SAFETY_BANNER}</div>
        </div>
      ) : null}

      <main className="wrap">{children}</main>

      <footer className="wrap">
        Student project · survey numbers from CDC BRFSS 2025 · every patient in the demo is
        invented
      </footer>
    </>
  );
}
