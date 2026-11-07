# H15 — Subgroup Performance, Model Card & Limitations

## What we built (plain English)

`python -m ml.src.model_card` writes an honest one-page **model card** for every trained
condition: what the model is for, what it must not be used for, the data, how well it works,
**where it works less well** (by sex, age group and income group), the release-gate result,
and its limitations in plain language. It also draws a calibration chart ("when the model
says 20%, do about 20% of those people really have it?") comparing logistic regression and
XGBoost, and saves the limitation sentences the app shows next to every estimate (H25).

Cards are **generated from the trained model files**, so they can't drift from the models.
This also completes H49's "auto-generated model card for each condition".

## Key decisions

- **Tier 1** (the three fully-done conditions) get subgroup results; **Tier 2** cards show
  overall metrics only, as the roadmap says (`--subgroups` adds them for everyone).
- **Subgroups are measured on the held-out test set**, for the chosen model, at its
  recall-oriented cut-off: recall, PR-AUC, and average estimate vs actual rate.
- **Readable groups**: age bands merged into 18–34 / 35–49 / 50–64 / 65+; income into
  under $35k / $35k–$100k / $100k+ / not stated.
- **Groups under 30 people aren't scored** ("too few people"): small groups give noisy numbers.
- **Weak spots become limitation sentences automatically**: if a group's recall is ≥ 10
  points below overall, or its average estimate is ≥ 5 points off its real rate. Example
  from a synthetic run: *"It misses more cases for people in the group "age group 18-34"
  (catches 28% vs 73% overall)."*
- **Six general limitations** always shown: not a diagnosis; self-reported data; a snapshot
  (association, not cause); US adults, not Canada or children; only the survey's factors;
  unweighted training.
- **Wording lives in the generator**, not in hand-edited cards, so retraining never loses
  edits. Change `ml/src/model_card.py` to change the text.
- Race/ethnicity subgroups aren't reported because race isn't a feature (ADR-0003); the
  team may want to add it as a *reporting-only* group later.

## How it works

| Output | Where | Used by |
|---|---|---|
| Model card | `docs/model-cards/<condition>.md` | report, graders, H48 |
| Calibration chart | `docs/figures/<condition>/calibration.png` | report, card |
| Limitations | `ml/models/<condition>/limitations.json` | API → UI (H25, H40) |

## How to run / verify it

```bash
python -m pytest ml/tests/test_model_card.py
python -m ml.src.train && python -m ml.src.model_card   # needs the processed BRFSS file
```

17 tests: readable groups and their order, subgroup metrics against a hand-worked example,
small groups skipped, calibration bins, chart written, general + automatic limitation
sentences, every card section present, Tier 1 vs Tier 2 content, `limitations.json`, CLI.

## Results

Synthetic run (8,000 rows where risk rises with age and BMI; not reported results): the
card correctly flagged that the model catches far fewer cases in younger groups (28% for
18–34 vs 94% for 65+). Expect a similar pattern on BRFSS, since younger respondents rarely
report chronic conditions; that belongs in the report's discussion.

Real cards: after the BRFSS run, commit `docs/model-cards/*.md` and the calibration PNGs.

## Gotchas / follow-ups

- The card's figure link is relative (`../figures/...`), so it renders on GitHub.
- Builds on H50 (PR #17) and below; merge in order.
