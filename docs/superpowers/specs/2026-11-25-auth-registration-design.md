# Phase 1 — Accounts & Auth (design)

_Date: 2026-11-25 · Branch: to be cut from `main` as `feat/auth-registration`_

## Why

HealthLens today only has seeded demo accounts; nobody can make their own. The
makeover starts here because every later screen (questionnaire, dashboards, role
management) depends on a real account existing and on roles being trustworthy.
Self-registration (Option A, agreed) lets a normal visitor sign up, get a
limited **base** account, and later answer a short questionnaire to see their own
risk. Roles above base are never self-chosen — only granted by a super admin — so
registration can never be used to reach another person's records.

## Goals

- A visitor can **register** (full name, email, password) and is auto-logged-in.
- A visitor can **sign in** and **sign out**.
- New accounts get the **base** role: can see public pages and their own profile,
  nothing else. After registering they land on a "questionnaire coming next"
  placeholder (the real questionnaire is Phase 2).
- Login and Register are rebuilt in **Direction C** (warm & approachable), with a
  working **light/dark toggle** that every page will share.
- The role ladder gains **base** (bottom) and **super_admin** (top); existing
  demo accounts and behaviour are unchanged.

## Non-goals (later phases)

- The questionnaire and risk for base users (Phase 2).
- The widget dashboards for patient/doctor/admin (Phase 3).
- The super-admin screen that grants roles (Phase 4). The **role** exists now; the
  UI to use it comes later. One super-admin account is seeded so the chain is real.
- Email verification and password reset (needs a mail server; deferred, noted as a
  real-deployment gap).

## Anti-goals (would make this a failure)

- Registration that lets a user pick doctor/admin, or any path from base to a
  higher role without a super-admin granting it.
- Storing passwords as anything but an argon2 hash (the app already does this).
- A second theme that's an afterthought — dark must be legible, not just inverted.

## Approach

One sensible build path, because the stack already provides the hard parts:
argon2 hashing (`backend/app/security.py`), a signed `hl_session` httpOnly cookie,
the `users` table, and an auth router with `/login`, `/logout`, `/me`. We add a
`register` endpoint beside them and two new pages.

**Roles / schema.** Change the `users.role` CHECK constraint from
`('admin','doctor','patient')` to add `'base'` and `'super_admin'`. New column
default is not used — role is always set explicitly in code. Apply by editing
`backend/db/schema.sql` and re-running the seed (idempotent). **No migration tool
yet** — a local, wipeable dev database doesn't need Alembic; we add it in a later
phase when a schema change first has to preserve real data. (My call; say if you'd
rather add Alembic now.)

**Access control.** `base` sits below `patient` in the role ladder. The existing
uniform-404 rule is unchanged: a base user hitting `/doctor/*` or `/admin/*` gets
the same 404 as a missing resource. `super_admin` can do everything `admin` can,
plus (Phase 4) change roles; only `super_admin` may change roles, so a promoted
admin can't promote itself further.

**Register endpoint.** `POST /api/v1/auth/register` taking `{full_name, email,
password}`. Validates email format and uniqueness (the table already has UNIQUE),
enforces a minimum password length of **8** (my call), hashes with argon2, creates
a `base` user, writes an `auth.register` audit row, sets the session cookie
(auto-login), returns the same `UserOut` login returns. Duplicate email returns a
friendly 409, not a stack trace.

**Frontend.** Rebuild `/login` and add `/register` in Direction C. A shared theme
layer (CSS variables + a toggle in the header, choice remembered in
`localStorage`, default **light**). After register, redirect to a placeholder
`/welcome` that says the questionnaire is coming — real target in Phase 2.

**Seeded super admin.** The seed adds one `super_admin` account alongside the demo
users so the grant-roles chain is real from day one. Its credentials live in the
seed/demo config, not in chat.

## What's left to the builder

- Exact copy on the pages (tune later; not a blocker).
- Visual polish of the toggle control within Direction C.
- Test coverage: register happy path, duplicate email, short password, that a base
  user is refused doctor/admin routes with a 404, and that login/logout still pass.
