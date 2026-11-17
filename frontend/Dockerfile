# HealthLens frontend image, for the local full-stack run (`docker compose up`).
#
# Vercel does NOT use this file: it builds the Next app from frontend/ with its own
# pipeline. This exists so the whole stack can run on one machine with one command,
# and so the frontend is reproducible without a Vercel account.
#
# The build context is frontend/, NOT the repo root: the frontend needs nothing from
# outside this directory, and the root .dockerignore deliberately excludes frontend/
# to keep the API image small.
#     docker build -f frontend/Dockerfile -t healthlens-web ./frontend

# ---- deps: install against the lockfile only, so this layer caches ----------------
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- build: compile the standalone server ----------------------------------------
FROM node:22-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules node_modules
COPY . ./
# Next evaluates next.config.ts rewrites during `next build` and writes the resolved
# destination into .next/routes-manifest.json, so BACKEND_URL has to be present HERE, at
# build time. Setting it only in the container's environment has no effect: the compiled
# manifest still points at the build-time default and every /api/* call is refused.
# Vercel gets this right for free, because it exposes the project's env vars to the build.
ARG BACKEND_URL=http://127.0.0.1:8000
ENV BACKEND_URL=${BACKEND_URL}
RUN npm run build

# ---- run: just the standalone output ---------------------------------------------
FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
# next.config.ts sets output:"standalone", so .next/standalone holds the server plus
# only the dependencies it really imports. static/ and public/ are served beside it.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
RUN addgroup -g 10001 web && adduser -D -u 10001 -G web web && chown -R web:web /app
USER web
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "server.js"]
