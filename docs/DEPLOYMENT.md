# Deployment

Two services on two hosts, because they have different shapes.

| Service | Host | Why |
| --- | --- | --- |
| `frontend/` — Next.js | **Vercel** | Static assets plus short request handlers; exactly Vercel's model. |
| `backend/` — FastAPI + models | **Fly.io** | A long-running process that loads the trained pipelines at startup and keeps them in memory. Vercel's functions are per-request and capped well below that. |
| Postgres | **Supabase** (or Fly Postgres) | Managed, free tier, and the app already speaks to it through `DATABASE_URL`. |

The browser only ever talks to the Vercel domain. `frontend/next.config.ts` rewrites
`/api/*` to `BACKEND_URL`, so API calls are same-origin and the `hl_session` cookie stays
first-party. That is the whole reason the frontend proxies instead of the browser calling
Fly directly — see the cookie warning in H28 of the roadmap.

```
browser ──► Vercel (Next.js) ──rewrite /api/*──► Fly.io (FastAPI) ──► Supabase (Postgres)
                                                      │
                                                      └─► ml/models/*/explainer.joblib
                                                          (baked into the image)
```

## The one thing that is easy to get wrong

`ml/models/` is gitignored. A host that builds from the Git repo produces an image with
**no model artifacts**, and every prediction answers 503 while the rest of the app looks
fine. H27 requires artifacts to reach the backend baked into the image or downloaded at
start — never via git.

So the API image must be built where the artifacts exist: your machine, after
`python -m ml.src.train`. `fly deploy --local-only` does exactly that — it builds with
local Docker and pushes the finished image.

## Backend → Fly.io

```bash
fly launch --no-deploy --copy-config
```

```bash
fly secrets set JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" DEMO_PASSWORD="<pick one>"
```

```bash
fly deploy --local-only
```

`fly.toml` already sets `memory = "1gb"`. Do not drop it to the 256 MB default: the
container peaked at **541 MB** serving a diabetes prediction, because diabetes selects a
random forest whose explainer is 82 MB on disk and allocates further when
`shap.TreeExplainer` runs. Every other condition's explainer is under 2 MB.

Point it at Postgres when you want persistence (without it the API runs on the in-memory
demo store, which is fine for a demo but resets on restart):

```bash
fly secrets set DATABASE_URL="postgresql+psycopg://...supabase..."
```

The schema is not created automatically. Against a fresh database, run the seed once:

```bash
fly ssh console -C "python -m backend.scripts.seed_db"
```

## Frontend → Vercel

Import the repo at vercel.com/new, then:

- **Root Directory** → `frontend`  (without this the build fails; the repo root is not a Next app)
- **Environment variable** `BACKEND_URL` → `https://healthlens-api.fly.dev`
- **Environment variable** `NEXT_PUBLIC_DEMO_PASSWORD` → the same value you gave Fly

Framework, build and install commands come from `frontend/vercel.json`.

**Set `BACKEND_URL` before the first build, not after.** Next resolves the rewrite in
`next.config.ts` while `next build` runs and compiles the destination into
`.next/routes-manifest.json`. It is not read per request. If you add or change the
variable later you must **redeploy**, or the deployed app keeps proxying to the old
address and every `/api/*` call fails — with the UI itself looking perfectly healthy.
Changing it in the dashboard alone does nothing.

`next.config.ts` sets `output: "standalone"` for the Docker image; Vercel ignores it and
uses its own build pipeline, so the same config serves both.

## Verifying a deploy

```bash
curl https://healthlens-api.fly.dev/health
```

Then on the Vercel URL: sign in as `dr.chen@demo.healthlens.test`, open a patient, and
confirm a diabetes risk estimate renders with its contribution list. That exercises the
whole chain — proxy, cookie, RBAC, model load and SHAP explanation.

## Running the whole stack locally instead

No accounts needed:

```bash
docker compose up --build
```

Postgres, the API and the frontend, seeded, on <http://localhost:3000>.
