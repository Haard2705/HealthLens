# HealthLens

[![CI](https://github.com/Haard2705/HealthLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Haard2705/HealthLens/actions/workflows/ci.yml)

The proposed HealthLens project is connected to health informatics because it involves the
processing, analysis and presentation of public health data to facilitate the understanding of
relationships between health risk factors and outcomes.

A diabetes-risk dashboard and model-comparison tool: survey microdata → Logistic
Regression vs. XGBoost → SHAP explanations → FastAPI + PostgreSQL → Next.js dashboard.

> **Not a diagnostic tool.** HealthLens is a course project that explores population-level
> risk factors in self-reported survey data. It doesn't give medical advice.

**Plan & assignment tracker:** [`healthlens-todo.md`](healthlens-todo.md)

## Repository layout

```
data/        raw/ (gitignored) · processed/ (gitignored) · codebooks/
ml/          src/ (pipeline) · tests/ · notebooks/ (exploration only) · models/ (gitignored)
backend/     FastAPI app (H16+)
frontend/    Next.js app (H21+)
docs/        adr/ · assignments/
```

## Quick start (ML pipeline)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r ml/requirements.txt -r requirements-dev.txt
pre-commit install
pytest
```

Download the Kaggle **Diabetes Health Indicators** dataset into `data/raw/`
(see `healthlens-todo.md` §5.1). It is gitignored and must never be committed.

## Local stack

```bash
cp .env.example .env
docker compose up db
```

Backend and frontend services are added in H16 / H21.

## Contributing

Branch `h##-slug` off `main` → PR using the template → 1 review + green `CI / ci-ok` → squash-merge.
Conventional commits (`feat(ml): …`, `fix(api): …`).
