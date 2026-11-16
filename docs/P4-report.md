# P4: Prototype and Progress Evaluation

**COMP 4310 — Web Health Informatics, Lakehead University**

**HealthLens: An Interactive Analytics and Risk Prediction Dashboard for Public Health Data**

Team HealthLens — Haard Pathak (hpathak1@lakeheadu.ca), Krish Patel (kpate168@lakeheadu.ca),
Rudra Ukani (rukani@lakeheadu.ca), Vishwarajsinh Chauhan (vchauha7@lakeheadu.ca)

**Repository:** https://github.com/Haard2705/HealthLens (public)
**Date:** 15 November 2026

---

## A1. Prototype Overview and Implemented Functionality

### What the prototype does

HealthLens is a working web application built on the 2025 CDC BRFSS survey. It has three parts,
all running end to end:

1. **A reproducible data and machine-learning pipeline.** One command cleans the raw 800 MB
   survey file into 356,158 respondents across 22 canonical columns; a second trains and scores
   a model for each of nine chronic conditions, produces SHAP explanations, and writes a model
   card per condition.
2. **An explainable prediction layer.** Each individual estimate is decomposed into the factors
   that raised or lowered it, written out in plain language, and placed beside the prevalence
   for people in the same age and BMI band. A release gate decides whether a condition is
   allowed to produce an individual estimate at all.
3. **A role-based clinical demo.** A FastAPI backend with three roles (admin, doctor, patient),
   access checked on the server for every request, and an append-only audit log, with a Next.js
   dashboard over it. Every patient record is synthetic.

The nine conditions are diabetes, hypertension, heart disease, stroke, asthma, COPD, arthritis,
chronic kidney disease and depression.

### Requirements implemented

| ID | Requirement (abbreviated from P3) | Status | Evidence |
|---|---|---|---|
| FR1 | One-command clean of BRFSS; codes to missing; weights preserved | **Complete** | `python -m ml.src.clean` → 356,158 rows; `ml/tests/test_clean.py` |
| FR2 | Weighted prevalence with filters (age, sex, income, risk factor) | **Complete** | `GET /stats/prevalence`; groupings: age band, BMI band, sex, income band, physical activity. Fig. 2 |
| FR3 | Three models per condition, scores side by side | **Partial** | Logistic regression and XGBoost compared on one split for all nine; random forest in progress. Fig. 3 |
| FR4 | SHAP explanations summing to the model output, as sentences | **Complete** | `ml/src/explain.py`, `explanations.py`; sum-to-margin test. Fig. 5, 6 |
| FR5 | Model card per condition, incl. subgroup results | **Complete** | `docs/model-cards/*.md`, nine cards with calibration figures |
| FR6 | Estimate from entered risk factors, or an explanation of why not | **Complete** | `POST /api/v1/predict`; blocked conditions return HTTP 409 with the gate's reason |
| FR7 | Admin manages accounts; doctor sees assigned only; patient sees own | **Complete** | `backend/tests/test_access_matrix.py`. Fig. 4 |
| NFR1 | Server-side access checks, hashed passwords, audit log | **Complete** | argon2 hashing; uniform 404; append-only audit; CI blocks data commits |
| NFR2 | No individual estimate unless ROC-AUC ≥ 0.70, Brier beats baseline, not sensitive | **Complete** | `ml/src/gate.py`, ADR-0016; enforced in the API, not only the interface |
| NFR3 | One command rebuilds everything; fixed seeds; coverage > 70% | **Complete** | 97% (ML) and 91% (backend) coverage |
| NFR4 | Usability and accessibility | **Partial** | Keyboard focus, colour-contrast tokens, no horizontal scroll at 375px; classmate usability test not yet run |

### How this supports the P2 contribution

P2 identified a gap between two kinds of existing work: public dashboards that describe data
without modelling it, and prediction studies that model it but are unreadable to non-specialists
and rarely admit where a model is weak. Three implemented features close that gap.

- **Population charts that are survey-weighted and filterable** give what a public dashboard
  gives (Fig. 2).
- **A model comparison in the open** (Fig. 3) lets a reader see that the simple model usually
  ties the complex one, rather than being asked to trust a black box.
- **A refusal mechanism.** Two of the nine conditions are not permitted to produce an individual
  estimate: asthma because the model is too weak (ROC-AUC 0.66), depression because it is a
  sensitive condition. The refusal is enforced in the API, so it holds even if the interface is
  bypassed. A tool that declines to answer is the part generic risk calculators do not have.

The bridge between the two halves is the **"patients like this"** panel (Fig. 6): an individual
estimate is shown next to the measured prevalence among survey respondents in the same age and
BMI band, so the number is read as population context rather than a personal verdict.

### Screenshots

| Figure | File | Shows |
|---|---|---|
| Fig. 1 | `docs/screenshots/01-home.jpg` | Landing page |
| Fig. 2 | `docs/screenshots/02-explore-prevalence.jpg` | Weighted diabetes prevalence by age band, real group sizes |
| Fig. 3 | `docs/screenshots/03-model-results.jpg` | Model results: 7 of 9 conditions pass the release gate |
| Fig. 4 | `docs/screenshots/04-doctor-patients.jpg` | Doctor sees only assigned patients |
| Fig. 5 | `docs/screenshots/05-access-denied-404.jpg` | Another doctor's patient returns a neutral "not found" |
| Fig. 6 | `docs/screenshots/06-risk-context-shap.jpg` | Risk estimate with SHAP contributions |
| Fig. 7 | `docs/screenshots/07-patients-like-this.jpg` | "Patients like this" and the plain-language summary |

---

## A2. Repository and Implementation Evidence

**Repository:** https://github.com/Haard2705/HealthLens (public)

```
ml/         src/   pipeline, models, SHAP explanations, release gate
            tests/ 338 tests
backend/    app/   FastAPI: auth, RBAC, audit log, statistics, predictions
            tests/ 224 tests
frontend/          Next.js + TypeScript dashboard
data/              raw/ and processed/ gitignored; codebooks kept
docs/       adr/   architecture decision records
            assignments/   implementation notes, one per task
            model-cards/   one per condition
scripts/           dataset download
```

### Running it

Prerequisites: Python 3.12+ and Node.js 20+. Full instructions are in the repository README.
**No database is required** — the backend uses an in-memory store seeded with synthetic patients
unless a connection string is supplied, so the prototype can be inspected with no setup.

```
# terminal 1
python -m venv .venv && .venv\Scripts\activate
pip install -r backend/requirements.txt -r ml/requirements.txt
python -m uvicorn backend.app.main:app --port 8000

# terminal 2
cd frontend && npm install && npm run dev      # http://localhost:3000
```

Demo accounts (throwaway credentials over synthetic data, documented so the prototype can be
inspected): `fatima.hassan@demo.healthlens.test` (patient), `dr.chen@demo.healthlens.test`
(doctor), `admin@demo.healthlens.test` (admin); password `demo-password-123`.

Model artifacts are not committed. To reproduce them:
`python scripts/download_brfss.py` → `python -m ml.src.clean` → `python -m ml.src.train` →
`python -m ml.src.gate`.

**No credentials, API keys or patient data are in the repository.** Survey data and model
binaries are gitignored and a CI job fails the build if either is committed.

### Evidence of development process

40 commits and 15 merged pull requests; every change reviewed through a pull request with a
required status check; branch protection on `main`; 4 architecture decision records; 16
implementation notes; 9 model cards. Continuous integration runs linting, formatting, both test
suites with a coverage gate, a secret scan over full history, and a job that blocks raw survey
data and model binaries from being committed.

---

## A3. Progress Evaluation

### Position against the P3 timeline

P3 scheduled 8–14 November for the pipeline, API skeleton and first chart, and 15–21 November
for endpoints, model cards and frontend screens. Actual position: the data and machine-learning
work is **ahead of plan** (pipeline, nine conditions, explanations, gate and model cards all
complete), the backend is **on plan** (merged, including the estimate endpoints scheduled for
this week), and the frontend is **behind plan but functional** — the risk identified in P3 as the
most likely to materialise, and it did.

### Automated testing

| Suite | Tests | Coverage |
|---|---|---|
| ML pipeline | 338 | 97% |
| Backend API | 224 | 91% |
| **Total** | **562** | gate: 70% |

The most important suite is a parametrised role × endpoint × expected-status matrix. It proves
that a doctor cannot read another doctor's patient, a patient cannot reach admin routes, a
deactivated user loses access immediately, and denied attempts are written to the audit log.
Tests require neither a database nor network access.

### Preliminary results

Cleaned dataset: **356,158 respondents**, 2025 CDC BRFSS. Weighted prevalence:

| Condition | Prevalence | Condition | Prevalence |
|---|---|---|---|
| Hypertension | 35.2% | Heart disease | 6.8% |
| Arthritis | 26.6% | COPD | 6.4% |
| Depression | 21.5% | Kidney disease | 4.2% |
| Asthma | 16.0% | Stroke | 3.7% |
| Diabetes | 12.5% | | |

Model performance on a held-out 20% test set (~70,000 respondents per condition), for the model
selected on validation data:

| Condition | Model | ROC-AUC | PR-AUC | Recall | Precision | Individual estimate |
|---|---|---|---|---|---|---|
| COPD | Logistic regression | 0.836 | 0.351 | 0.80 | 0.20 | Enabled (strong) |
| Heart disease | Logistic regression | 0.826 | 0.317 | 0.80 | 0.21 | Enabled (strong) |
| Diabetes | XGBoost | 0.803 | 0.384 | 0.81 | 0.29 | Enabled (strong) |
| Hypertension | Logistic regression | 0.790 | 0.705 | 0.80 | 0.61 | Enabled (moderate) |
| Kidney disease | Logistic regression | 0.788 | 0.170 | 0.79 | 0.11 | Enabled (moderate) |
| Arthritis | Logistic regression | 0.787 | 0.629 | 0.80 | 0.54 | Enabled (moderate) |
| Stroke | Logistic regression | 0.783 | 0.140 | 0.78 | 0.09 | Enabled (moderate) |
| Depression | XGBoost | 0.736 | 0.448 | 0.80 | 0.31 | **Blocked — sensitive** |
| Asthma | XGBoost | 0.661 | 0.278 | 0.81 | 0.20 | **Blocked — below 0.70** |

Recall and precision are reported at the recall-oriented operating point chosen on validation
data, which is the threshold a screening tool would use. Three observations worth recording:

- **Seven of nine conditions pass the release gate.** The two that do not are blocked for
  different reasons, and the application states which.
- **Logistic regression wins for six of nine conditions.** The team's rule — prefer the simpler
  model unless the complex one is clearly better on validation data — meant the interpretable
  model was selected more often than not. This is a result, not a shortcut.
- **Precision is low where the condition is rare.** For stroke (3.7% prevalence) a threshold set
  to catch 78% of cases flags roughly eleven people for every one who has it. The interface
  states this in words rather than presenting the percentage alone, which is the reason the
  explanation layer exists.

Verified example output, from the running system (patient P-0103, synthetic, heart disease):
estimated risk 24%, strong signal, logistic regression. Largest contributions: self-rated
general health *fair* (+0.81 log-odds), age 65–69 (+0.63), male (+0.37), daily smoker (+0.34).
Among survey respondents aged 65–69 with a BMI of 30 or more, 15.2% report heart disease
(n = 11,952).

---

## A4. Remaining Work, Challenges, and Updated Plan

### Remaining before the final project

1. **Random forest** as the third model (FR3), to complete the comparison promised in P3.
2. **Frontend completion**: remaining accessibility work and the usability test with five
   classmates (NFR4).
3. **Deployment** of both tiers, and promotion of PostgreSQL from optional to default.
4. **A fuller hyperparameter search**, and subgroup and calibration analysis written into the
   Tier 1 model cards.
5. **CCHS decision** — whether to add the Canadian survey as a second source.

### Challenges encountered

- **Class weighting had to be abandoned.** P2 planned to handle class imbalance with balanced
  class weights. In practice they inflated the predicted probabilities past the no-skill Brier
  score, making the percentages untrustworthy. We removed them and instead choose a
  recall-oriented decision threshold on validation data.
- **A skipped survey question is not a "no".** BRFSS uses per-question sentinel codes for
  "don't know" and "refused". Respondents with a missing outcome are dropped from that
  condition's split rather than counted as negative, which changes the usable sample per
  condition.
- **Some conditions are weakly predictable from lifestyle answers.** Asthma reaches only 0.661
  ROC-AUC. Rather than present a weak number, the gate blocks individual estimates and the
  interface explains why.
- **An interface-level integration defect.** The frontend was written against the estimate
  endpoint before it existed and assumed the explanation was a single string; the API returns
  one per reading level. The page crashed on the first real response. It was found by running
  the two tiers together rather than by either test suite, which is a reminder that
  component-level tests do not substitute for end-to-end checks.
- **The frontend started late**, exactly as P3 predicted. Mitigated by building one thin
  end-to-end slice first and by keeping the backend runnable with no database.

### Changes from P3, with justification

| Change | Reason |
|---|---|
| Charts drawn as inline SVG rather than with Recharts | Removes a dependency; the charts are bars and single-series lines. Runtime dependencies are only `next`, `react` and `react-dom` |
| Database optional, in-memory by default | A reviewer can run the prototype with no setup. PostgreSQL is enabled by setting one environment variable |
| Supabase project hosted in US East rather than Canada | Free-tier region selection. All patient data is synthetic; a real pilot would require Canadian hosting, as stated in P3 |
| Optional Ollama summaries not implemented | P3 identified these as the first thing to drop; priority went to the three-model comparison and the estimate endpoints |

### Updated plan

| Window | Work |
|---|---|
| 16–21 Nov | Random forest; remaining frontend screens; subgroup results in the Tier 1 model cards |
| 22–27 Nov | Deployment; accessibility fixes; usability test with five classmates; security review. **Feature freeze 27 November** |
| 28 Nov – 4 Dec | Final evaluation, report, demonstration rehearsal |

The principal remaining risk is frontend scope. If it slips, the admin console and the optional
AI summaries are dropped first; the graded core — three roles, prevalence charts, model
comparison, explained estimates and the audit log — is protected.

---

### Note on data and privacy

All survey data is public and aggregated. Every patient record in the application is synthetic
and generated by a seeded script; no real health information is used anywhere in the project.
HealthLens is a course project and is not a diagnostic tool.
