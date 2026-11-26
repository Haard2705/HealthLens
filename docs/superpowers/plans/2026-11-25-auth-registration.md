# Auth & Registration (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a visitor register a self-service `base` account and sign in, with the Login and Register pages rebuilt in Direction C (warm/sage) and a working light/dark toggle shared by every page.

**Architecture:** The backend already has argon2 hashing, a signed `hl_session` cookie, a `users` table, a store that works in-memory or on Postgres (`UserStore` / `PostgresUserStore`, both expose `.add()` which persists), and an auth router with `/login` `/logout` `/me`. We add a `register` endpoint beside them, two new roles, and a seeded super-admin. The frontend gains the Direction C token palette + a theme toggle, a `/register` page, and a `/welcome` placeholder.

**Tech Stack:** FastAPI · pydantic · argon2 · PyJWT · Next.js 15 (App Router, webpack build) · TypeScript.

**Spec:** `docs/superpowers/specs/2026-11-25-auth-registration-design.md`

## Global Constraints

- Branch `feat/auth-registration` cut from `main`.
- Commits are authored by Haard only — **no `Co-Authored-By` line, no generated-by footer** (project memory rule).
- Minimum password length: **8** characters (verbatim from spec).
- Default theme: **light**; the toggle persists the choice in `localStorage`.
- New signups get role **base**; roles above base are never self-assignable.
- Frontend production build uses webpack (`next build`, no `--turbopack`) — already set; don't reintroduce the flag.
- Every file written through a script must be UTF-8 (a prior cp1252 write broke a Vercel build).

## Review Focus

- **Duplicate email differing only in case** (`Foo@x.com` vs `foo@x.com`) must still be rejected — register normalizes before the uniqueness check. _(Task 2)_
- **Password length boundary** — 7 chars rejected (422), 8 accepted. _(Task 2)_
- **A `base` user hitting `/doctor/*` or `/admin/*`** gets 403 (role not permitted), never another user's data. _(Task 1)_
- **`localStorage` throwing** (private window, blocked storage) must not break the theme toggle or the page render. _(Task 3)_
- **Register → login round-trip** with the same credentials succeeds (hash stored, cookie honored). _(Task 2)_

---

### Task 1: Two new roles + seeded super-admin

**Files:**
- Modify: `backend/app/users.py` (Role enum; `seed_demo_users`)
- Modify: `backend/db/schema.sql:60` (the `users.role` CHECK constraint)
- Test: `backend/tests/test_auth.py` (new cases), `backend/tests/test_db.py` (seed count)

**Interfaces:**
- Produces: `Role.BASE = "base"`, `Role.SUPER_ADMIN = "super_admin"`; `seed_demo_users(repo, password)` now also yields a `User("u-superadmin", "super@demo.healthlens.test", "Platform Owner", Role.SUPER_ADMIN, pw)`.

- [ ] **Step 1: Write failing tests**

```python
def test_role_enum_has_base_and_super_admin():
    assert Role("base") is Role.BASE
    assert Role("super_admin") is Role.SUPER_ADMIN

def test_seed_includes_one_super_admin(repo):  # repo = SyntheticRepository()
    users = seed_demo_users(repo, "demo-password-123").all()
    supers = [u for u in users if u.role is Role.SUPER_ADMIN]
    assert len(supers) == 1 and supers[0].email == "super@demo.healthlens.test"

def test_base_user_forbidden_from_doctor_list(client, base_user):
    # base_user logged in via cookie; /doctor/patients requires Role.DOCTOR
    resp = client.get("/api/v1/doctor/patients")
    assert resp.status_code == 403
```

- [ ] **Step 2: Run tests, verify they fail** — `pytest backend/tests/test_auth.py -k "role_enum or super_admin or base_user_forbidden" -v` → FAIL.

- [ ] **Step 3: Add `BASE = "base"` and `SUPER_ADMIN = "super_admin"` to `Role` in `backend/app/users.py`.** Append the super-admin `User` to the list in `seed_demo_users` (before the `return`).

- [ ] **Step 4: Widen the schema CHECK** in `backend/db/schema.sql` from `role IN ('admin', 'doctor', 'patient')` to `role IN ('base', 'patient', 'doctor', 'admin', 'super_admin')`.

- [ ] **Step 5: Run tests, verify pass.** Also run the full `pytest backend/tests -q` to confirm nothing else broke (existing `active_admin_count`, role checks).

- [ ] **Step 6: Commit** — `feat(auth): add base and super_admin roles, seed a super admin`

---

### Task 2: `POST /auth/register` endpoint

**Files:**
- Modify: `backend/app/schemas.py` (add `RegisterRequest`)
- Modify: `backend/app/routers/auth.py` (add `register`)
- Modify: `backend/app/errors.py` only if no reusable 409 helper exists (check first; `AppError(409, ...)` inline is fine)
- Test: `backend/tests/test_auth.py`

**Interfaces:**
- Consumes: `hash_password` (security), `is_valid_email`/`normalize_email` (users), `User`/`Role.BASE`, `state.users.add`, `issue_token`/`COOKIE_NAME`, `UserOut.of`, the `_email_ref` helper already in `auth.py`.
- Produces: `RegisterRequest(full_name: str[min 1,max 120], email: Email, password: str[min 8,max 256])`; endpoint returns `UserOut` and sets the `hl_session` cookie exactly as `login` does.

- [ ] **Step 1: Write failing tests**

```python
def test_register_creates_base_user_and_sets_cookie(client):
    r = client.post("/api/v1/auth/register",
        json={"full_name": "Sam Lee", "email": "sam@example.com", "password": "pw-at-least-8"})
    assert r.status_code == 201
    body = r.json()
    assert body["role"] == "base" and body["email"] == "sam@example.com"
    assert "hl_session" in r.cookies

def test_register_then_login_roundtrip(client):
    client.post("/api/v1/auth/register",
        json={"full_name": "Sam Lee", "email": "Sam@Example.com", "password": "pw-at-least-8"})
    r = client.post("/api/v1/auth/login",
        json={"email": "sam@example.com", "password": "pw-at-least-8"})
    assert r.status_code == 200 and r.json()["role"] == "base"

def test_register_duplicate_email_differing_case_409(client):
    payload = {"full_name": "A", "email": "dup@example.com", "password": "pw-at-least-8"}
    client.post("/api/v1/auth/register", json=payload)
    r = client.post("/api/v1/auth/register",
        json={**payload, "email": "DUP@example.com", "full_name": "B"})
    assert r.status_code == 409

def test_register_password_too_short_422(client):
    r = client.post("/api/v1/auth/register",
        json={"full_name": "A", "email": "x@example.com", "password": "short7!"})  # 7 chars
    assert r.status_code == 422
```

- [ ] **Step 2: Run tests, verify fail** — `pytest backend/tests/test_auth.py -k register -v` → FAIL (404/405: route missing).

- [ ] **Step 3: Add `RegisterRequest` to `schemas.py`** (a `_Strict` model; `full_name: str = Field(min_length=1, max_length=120)`, `email: Email = Field(min_length=3)`, `password: str = Field(min_length=8, max_length=256)`).

- [ ] **Step 4: Implement `register(body, request, response, access) -> UserOut` in `auth.py`.** `@router.post("/register", response_model=UserOut, status_code=201)`, `access` via `Depends(anonymous_access)`. Algorithm: normalize email; `if not is_valid_email(email): raise AppError(422,...)`; `if state.users.by_email(email): raise AppError(409, "email_taken", "That email is already registered.")`; build `User(id=f"u-{uuid4().hex[:12]}", email=email, display_name=body.full_name.strip(), role=Role.BASE, password_hash=hash_password(body.password))`; `state.users.add(user)`; set the cookie with the same `response.set_cookie(...)` block as `login`; `access.as_user(user).record("auth.register", audit.ALLOWED, "session", _email_ref(email))`; `return UserOut.of(user)`.

- [ ] **Step 5: Run tests, verify pass**, then `pytest backend/tests -q`.

- [ ] **Step 6: Commit** — `feat(auth): add self-service register endpoint (base role)`

---

### Task 3: Direction C palette + light/dark toggle (shared theme layer)

**Files:**
- Modify: `frontend/app/globals.css` (replace color tokens with Direction C; add `:root[data-theme="dark"]` block)
- Modify: `frontend/app/layout.tsx` (no-flash inline script in `<head>`)
- Modify: `frontend/components/Shell.tsx` (toggle control in header)
- Create: `frontend/components/ThemeToggle.tsx`
- Test: `frontend/` has no unit runner for CSS; verify via `npm run build` + a manual note. Add a tiny pure helper with a test instead (below).
- Create: `frontend/lib/theme.ts` + `frontend/lib/theme.test.ts` _(only if a test runner exists; otherwise fold the helper into ThemeToggle and rely on the build)_

**Interfaces:**
- Produces: `applyTheme(next: "light" | "dark" | "system"): void` and `readStoredTheme(): "light" | "dark" | "system"` in `lib/theme.ts`, both wrapping every `localStorage`/`document` access in try/catch. `<ThemeToggle />` renders a button that flips light↔dark and persists.

- [ ] **Step 1 (tokens): Replace the light `:root` color tokens** in `globals.css` with Direction C, keeping the same token **names** so every existing component recolors automatically:
  `--bg:#eef0ea; --surface:#fbfcf9; --sunk:#e4e7dd; --ink:#2c342a; --muted:#6d7a65; --line:#cdd4c3; --accent:#4f6f4a; --accent-ink:#fbfcf9; --accent-weak:#e2e8da;` (leave `--up/--down/--warn*/--ok*/--bad*` as they are unless they clash; adjust only if unreadable).

- [ ] **Step 2 (dark):** In the existing `@media (prefers-color-scheme: dark) :root:not([data-theme="light"])` block set the warm-dark values: `--bg:#1a1d17; --surface:#23271f; --sunk:#2b3026; --ink:#e8ece0; --muted:#9aa48d; --line:#363d2e; --accent:#8fb382; --accent-ink:#141a0f; --accent-weak:#2a3222;`. Then **add a sibling block** `:root[data-theme="dark"] { … same warm-dark values …; color-scheme: dark }` so the toggle forces dark even on a light OS.

- [ ] **Step 3: Write the theme helper + failing test** (`lib/theme.ts`, `lib/theme.test.ts` if a runner exists):

```ts
test("readStoredTheme falls back to system when storage throws", () => {
  // mock localStorage.getItem to throw
  expect(readStoredTheme()).toBe("system");
});
```
If `frontend` has no JS test runner configured, skip the test file and instead assert the try/catch by code review; note it in the commit.

- [ ] **Step 4: Implement `lib/theme.ts`** — `applyTheme` sets/removes `document.documentElement.dataset.theme` and writes `localStorage["hl-theme"]`; `readStoredTheme` reads it; both swallow exceptions and default to `"system"`.

- [ ] **Step 5: No-flash script** — in `layout.tsx`, add a `<head>` with a tiny inline `<script dangerouslySetInnerHTML>` that reads `localStorage["hl-theme"]` in a try/catch and sets `document.documentElement.dataset.theme` before paint (only when it is `"light"`/`"dark"`).

- [ ] **Step 6: Build `ThemeToggle.tsx`** (a `"use client"` button using `applyTheme`/`readStoredTheme`, showing a sun/moon label) and place it in the `Shell` header's `.who` cluster, left of the sign-in/out control.

- [ ] **Step 7: Verify** — `cd frontend && npm run build` → green; open `http://localhost:3000`, toggle persists across reload, no flash. (Run the stack with `docker compose up` or `npm run dev`.)

- [ ] **Step 8: Commit** — `feat(ui): Direction C palette and a light/dark toggle`

---

### Task 4: Register page + session `signUp`

**Files:**
- Modify: `frontend/lib/types.ts:3` (widen `Role`)
- Modify: `frontend/components/SessionProvider.tsx` (add `signUp`)
- Create: `frontend/app/register/page.tsx`, `frontend/app/register/RegisterForm.tsx`
- Modify: `frontend/components/Shell.tsx` (NAV_BY_ROLE gains `base`, `super_admin`)
- Test: covered by `npm run build` (typecheck) + manual flow; no component test runner assumed.

**Interfaces:**
- Consumes: `apiSend`, `useSession`.
- Produces: `Role = "base" | "patient" | "doctor" | "admin" | "super_admin"`; `session.signUp(full_name, email, password): Promise<User>` (POSTs `/auth/register`, sets user, returns it).

- [ ] **Step 1: Widen `Role`** in `lib/types.ts` to the five-member union. `npm run build` will now flag every non-exhaustive `Record<Role, …>` — that list is the rest of this task.

- [ ] **Step 2: Add `signUp`** to `SessionProvider` mirroring `signIn` but calling `POST /auth/register`.

- [ ] **Step 3: Fix the flagged `Record<Role, …>` maps** — at minimum `NAV_BY_ROLE` in `Shell.tsx` (give `base` a minimal nav: `Home`, `Explore`; give `super_admin` the admin nav for now) and `LANDING` in `LoginForm.tsx` (`base: "/welcome"`, `super_admin: "/"`).

- [ ] **Step 4: Build `/register`** — `RegisterForm` with full name, email, password fields (Direction C styling via the shared tokens), a submit that calls `signUp` then `router.push("/welcome")`, friendly inline errors (email taken → 409 message, short password → validation). `page.tsx` sets `metadata.title = "Create account · HealthLens"`.

- [ ] **Step 5: Verify** — `npm run build` green; register a new account in the running app, confirm it lands on `/welcome` logged in, and that the new row appears (`docker compose exec db psql -U healthlens -d healthlens -c "select email, role from users where role='base';"`).

- [ ] **Step 6: Commit** — `feat(ui): register page and signUp, base/super_admin in Role`

---

### Task 5: Login page in Direction C + `/welcome` placeholder

**Files:**
- Modify: `frontend/app/login/LoginForm.tsx` (restyle, add a "Create account" link; keep the demo accounts block)
- Create: `frontend/app/welcome/page.tsx`
- Test: `npm run build` + manual.

**Interfaces:**
- Consumes: `useSession`, the shared tokens, `session.user`.

- [ ] **Step 1: `/welcome` placeholder** — a client page that greets `session.user?.display_name`, says the health questionnaire is coming next (Phase 2), and links to Explore. Redirect to `/login` if there is no user once `loading` is false.

- [ ] **Step 2: Restyle `LoginForm`** to Direction C using the shared tokens (it already uses token classes; adjust structure/spacing to match the mockup, add a prominent "New here? Create an account" link to `/register`). Keep the existing demo-account quick-login block — it is how the professor signs in.

- [ ] **Step 3: Route a signed-in base user** — confirm `LANDING.base === "/welcome"` so login sends base users to `/welcome`.

- [ ] **Step 4: Verify** — `npm run build` green; sign in as a demo doctor (still works, lands `/doctor`), sign in as a freshly-registered base user (lands `/welcome`), toggle theme on both pages.

- [ ] **Step 5: Commit** — `feat(ui): login page in Direction C, welcome placeholder`

---

### Task 6: Open the PR

- [ ] **Step 1:** Push `feat/auth-registration`, open a PR to `main` titled "Phase 1 — accounts, registration, Direction C theme". Body: what shipped, the five Review-Focus checks, and that questionnaire/dashboards/role-grant UI are Phases 2–4. **No generated-by footer.**
- [ ] **Step 2:** Confirm CI is green and report the preview URL.

---

## Self-review notes

- **Spec coverage:** register (T2), login+logout unchanged + restyle (T5), base role & limited view (T1/T4 nav), Direction C + toggle (T3), base+super_admin roles & seeded super admin (T1), auto-login→welcome placeholder (T2/T4/T5), deferred email-verify (not built, noted). Covered.
- **Super-admin powers** (doing everything admin can) are deliberately **not** wired in Phase 1 — the role exists and is seeded; its abilities arrive with the grant-roles UI in Phase 4. Matches the spec's "role exists now, UI later."
- **Uniform 404 vs 403:** endpoint role-gating returns **403** (`requires(...)` → forbidden); the uniform **404** is specific to individual patient lookups. The spec's "same 404" phrasing is imprecise; a base user is correctly refused doctor/admin endpoints with 403 (identical to how a patient is refused today). Test asserts 403 (T1).
