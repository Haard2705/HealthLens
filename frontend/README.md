# HealthLens frontend

The Next.js dashboard for HealthLens: public population charts, role-checked patient records,
and the model-results page that shows which conditions are allowed to produce a personal
estimate.

Next.js 15 (App Router) + TypeScript. No Tailwind, no UI library, no chart library: one
global stylesheet of CSS custom properties, bars drawn with `div`s, trends drawn with inline
SVG. `next`, `react` and `react-dom` are the only runtime dependencies.

## Prerequisites

- **Node 22** (`node --version`)
- The **HealthLens backend** running on port 8000 (see below)

## Run it

Two terminals, from the repository root.

**1. The API**

```bash
# Windows PowerShell
$env:DEMO_PASSWORD = "demo-password-123"
.venv\Scripts\python -m uvicorn backend.app.main:app --port 8000

# macOS / Linux
DEMO_PASSWORD=demo-password-123 .venv/bin/python -m uvicorn backend.app.main:app --port 8000
```

**2. This app**

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

Other scripts: `npm run build` (production build), `npm start` (serve the build),
`npm run lint`, `npx tsc --noEmit`.

## Demo logins

Sign-in is at `/login`, which has a one-click button per role. The password is printed on the
page on purpose: this is a prototype over invented patients.

| Role    | Email                                 | Sees                                            |
| ------- | ------------------------------------- | ----------------------------------------------- |
| Patient | `fatima.hassan@demo.healthlens.test`  | her own record only                             |
| Doctor  | `dr.chen@demo.healthlens.test`        | the six patients assigned to Dr. Chen           |
| Admin   | `admin@demo.healthlens.test`          | users, system status and the audit log          |

Password: `demo-password-123` (the backend's `DEMO_PASSWORD`; change both together).

**Worth demonstrating:** sign in as the doctor, then use the "Try P-0107" control at the
bottom of the patient list. P-0107 belongs to Dr. Okafor, and the API answers with exactly the
same `404 not found` it gives for a patient that does not exist — never `403 forbidden`, so the
response cannot be used to work out who is a patient. Sign back in as the admin and the denied
attempt is in the audit log.

## Environment variables

Copy `.env.example` to `.env.local` only if you need to change something; both values have
working defaults.

| Variable                    | Default                 | What it does                                                 |
| --------------------------- | ----------------------- | ------------------------------------------------------------ |
| `BACKEND_URL`               | `http://localhost:8000` | Where `/api/*` is proxied to (read at build/start time).      |
| `NEXT_PUBLIC_DEMO_PASSWORD` | `demo-password-123`     | The password shown and used by the demo buttons on `/login`. |

`.env*` files are gitignored; only `.env.example` is tracked.

## How it talks to the backend

The session is an **httpOnly cookie** (`hl_session`), so JavaScript cannot read it and the
browser has to send it automatically. For that to work without CORS credentials gymnastics,
every request is **same-origin**:

```
browser  ──fetch("/api/v1/...", { credentials: "include" })──▶  Next.js (:3000)
                                                                   │ rewrite in next.config.ts
                                                                   ▼
                                                               FastAPI (:8000)
```

`next.config.ts` rewrites `/api/:path*` to `${BACKEND_URL}/api/:path*`. Nothing in the app
ever uses an absolute API URL, so the same code works locally and behind one domain in
production.

All API access goes through `lib/api.ts`, which normalises the backend's
`{"error": {"code", "message"}}` envelope into an `ApiError` carrying the status. Screens use
`lib/useResource.ts` for a loading state, an error state with a retry, and request
cancellation.

### Endpoints used

| Screen                 | Endpoints                                                                   |
| ---------------------- | --------------------------------------------------------------------------- |
| Shell, sign-in         | `GET /auth/me`, `POST /auth/login`, `POST /auth/logout`                     |
| Home                   | `GET /conditions`                                                           |
| Explore                | `GET /conditions`, `GET /stats/prevalence?condition=&by=`                   |
| Model results          | `GET /conditions`, `GET /models?condition=`                                 |
| Doctor                 | `GET /doctor/patients?q=`, `GET /doctor/patients/{id}`, `…/timeline`        |
| My health (patient)    | `GET /me/record`, `GET /me/medications`, `GET /me/observations`             |
| Admin                  | `GET /admin/users`, `POST /admin/users/invite`, `PATCH /admin/users/{id}`, `GET /admin/audit`, `GET /admin/system` |
| Risk panel             | `GET /patients/{id}/risk-context?condition=` (optional — see below)         |

### Things that degrade instead of breaking

- **Risk estimates.** The per-patient estimate endpoint is being built on another branch. If it
  answers 404 or 501, the risk tab shows a short "risk estimates are not available yet" panel
  and points at Explore and Model results.
- **Untrained models.** `GET /models` answers `404 model_not_trained` until the pipeline has
  run. The Model results page then shows the gate rule and every condition, with
  "not trained in this build" in place of a score; it fills in by itself once
  `ml/models/<condition>/metrics.json` exists.
- **No cleaned survey file.** `GET /stats/prevalence` falls back to clearly-labelled
  illustrative numbers (`source: "sample"`), and Explore says so above the chart.
- **A record you may not see.** The 404 is rendered as a neutral "we could not find that
  record" page. The frontend never says "forbidden".
- **Doctor observation trends.** The doctor endpoints expose check-ups through the timeline
  summary, so `lib/format.ts` reads the three numbers back out of that sentence. Anything that
  does not parse is skipped rather than guessed.

## Layout

```
app/
  layout.tsx          fonts, global CSS, SessionProvider + Shell
  page.tsx            home (server component) + <AtAGlance/> island
  login/              demo role buttons and the email/password form
  explore/            condition + group-by selects, prevalence bars
  models/             model results table with the release gate
  doctor/             patient list, [id]/ record detail
  me/                 the patient's own record
  admin/              users, invite, system status, audit log
components/           Shell, RecordDetail, RiskContextPanel, BarList, TrendChart, States
lib/                  api.ts, useResource.ts, format.ts, types.ts
```
