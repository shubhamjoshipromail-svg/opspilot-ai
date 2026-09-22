# OpsPilot

**Confidence-gated ticket routing.** A transformer classifier assigns support
tickets to operational queues, and a rules layer decides which predictions are
trustworthy enough to act on and which go to a human.

Built on 23,747 labeled tickets with fixed splits, leakage controls, per-class
error analysis, and a benchmark against three model families — including a
zero-shot decision model that lost.

---

## Results

Ticket routing from `subject + body`, fixed splits, `random_state=42`.

| Model | Taxonomy | Split | Accuracy | Macro-F1 | Weighted-F1 | Trained on |
|---|---|---|---:|---:|---:|---|
| TF-IDF + Logistic Regression | 10-class | test | 0.4409 | 0.4121 | 0.4430 | 16,622 |
| ModernBERT-base | 10-class | unrecorded | 0.5229 | 0.4725 | 0.5158 | 16,622 |
| **ModernBERT-base + `clean_v1`** | **7-class** | **validation** | **0.7313** | 0.5630 | **0.7127** | 16,622 |
| Jev (TypeSafe System One) | 7-class | validation | 0.5458 | 0.3289 | 0.5552 | **0 — zero-shot** |
| *Majority-class baseline* | *7-class* | *validation* | *0.5946* | *—* | *—* | *—* |

Re-running the published checkpoint locally on the same split reproduces
`0.7294` against the recorded `0.7313`, a 0.2pp difference attributable to
tokenizer and batching details. Both systems have been run row-by-row, so the
comparison is paired: ModernBERT is exclusively correct on 883 tickets, Jev on
229 (McNemar p = 5.5e-91). The gap is not sampling noise.

### Confidence gating

Coverage and accuracy on auto-routed tickets, ModernBERT, validation split:

| Threshold | Auto-routed | Accuracy on auto-routed |
|---:|---:|---:|
| 0.70 | 71.8% | 83.4% |
| 0.75 | 66.5% | 85.0% |
| **0.80** | **61.8%** | **86.5%** |
| 0.85 | 56.0% | 88.5% |
| 0.90 | 48.9% | 90.3% |

**Calibration:** ModernBERT's expected calibration error is 0.0841. The
zero-shot model measured 0.2896 on the same rows — 3.4x worse. This matters
more than the accuracy gap, because the routing layer gates on confidence: a
miscalibrated model does not just make mistakes, it makes them while claiming
to be certain.

The `clean_v1` figures are reproducible from the published training run:
[`trainer/checkpoint-3117/trainer_state.json`](https://huggingface.co/shubhamjoshipro/opspilot-routing-modernbert-base-clean-v1/raw/main/trainer/checkpoint-3117/trainer_state.json)
in the model repository records `eval_accuracy` 0.73133, `eval_macro_f1`
0.56304 and `eval_weighted_f1` 0.71272 at epoch 3, with `training_config.json`
confirming the run name, label map and seed.

**Read the taxonomy column before the accuracy column.** The jump from 0.5229
to 0.7313 is *not* a pure modeling gain — it combines a real improvement with a
relabeling that merged four overlapping queues into two. A 7-class problem is
easier than a 10-class one. The honest claim is that queue design mattered as
much as architecture, which is itself the more interesting finding.

Two caveats stated plainly:

- **The splits are not consistently recorded.** The `clean_v1` and Jev numbers
  are both validation, so that comparison is like-for-like. The TF-IDF figure
  is test. Which split produced the 10-class ModernBERT result was never
  written down, so treat the 0.5229 to 0.7313 delta as indicative rather than
  exact until it is re-measured.
- **Macro-F1 lags accuracy badly** in every run. At 0.5630 against 0.7313, the
  `clean_v1` model is carried by the majority queue and is weak on the small
  ones. Accuracy alone overstates how well this works.

## What the routing layer actually does

A classifier alone is not a product. Every prediction passes through
`routing.py`, which combines model confidence with deterministic risk rules:

| Condition | Route |
|---|---|
| Escalation risk >= 0.80 | `human_review` |
| Policy-sensitive language or category | `supervisor_review` |
| Category confidence < 0.65 | `human_review` |
| High-priority with sufficient confidence | `priority_queue` |
| Otherwise | `auto_triage_suggestion` |

This is the part that makes a 73%-accurate model usable. Escalation keywords
(`chargeback`, `legal`, `regulator`, `charged twice`) and policy-sensitive
categories override the model entirely — a ticket the classifier is confident
about still goes to a human if it mentions a lawsuit.

## Repository map

```text
backend/                      FastAPI service, ticket CRUD, prediction persistence
frontend/                     Operator-facing Streamlit console
verticals/ticket-intelligence/
  ml/ticket_intelligence/     Training, evaluation, routing, error analysis
    jev_benchmark/            Zero-shot decision-model benchmark (see below)
  docs/ML_RESULTS_REPORT.md   Full results, confusion matrices, error analysis
  ml/.../MODEL_CARD.md        Intended use, limitations, failure modes
docs/experiments/             Experiment write-ups
data/                         Dataset regeneration and policy documents
```

## Dataset

Regenerated from the public `Tobi-Bueck/customer-support-tickets` set. English
only, exact duplicates removed, normalized to a fixed schema.

| Split | Rows |
|---|---:|
| Train | 16,622 |
| Validation | 3,562 |
| Test | 3,563 |

Stratified by category, `random_state=42`, saved to disk so every model is
measured on identical rows.

**Leakage control:** the dataset carries a `reference_answer` field written
*after* a ticket is resolved. It is excluded from every model input. Using it
would inflate accuracy substantially and invalidate the results.

This is a public benchmark dataset, not private customer data.

## The Jev experiment

A controlled test of whether a zero-shot typed-decision model can replace a
fine-tuned encoder on this task.

**It cannot — not here.** Jev scored 0.5458 against the fine-tuned model's
0.7294 on the same validation rows, and did not clear the 0.5946
majority-class baseline. On a paired test the fine-tuned model is exclusively
correct on 883 tickets to Jev's 229 (McNemar p = 5.5e-91).

Calibration was the sharper failure: ECE 0.2896 against ModernBERT's 0.0841.
Where Jev reported 0.98 confidence, actual accuracy was 0.65.

The result is worth keeping precisely because it is negative, and because the
likely cause is informative: a fine-tuned model learns *this dataset's filing
conventions*, which no zero-shot model can infer from a prompt.

Full write-up, methodology and next steps: **[docs/experiments/jev_vs_modernbert.md](docs/experiments/jev_vs_modernbert.md)**

## Implementation status

| Capability | Status |
|---|---|
| FastAPI backend, ticket CRUD | Implemented |
| Hugging Face routing service | Implemented |
| Prediction persistence, model metadata | Implemented |
| Alembic migrations | Implemented |
| Dashboard APIs | Implemented |
| Streamlit operator console | Implemented |
| Confidence-gated routing rules | Implemented |
| Zero-shot benchmark harness | Implemented |
| Human review records and overrides | Planned |
| Policy retrieval / RAG | Planned |
| Response drafting | Planned |
| Authentication and permissions | Planned |

## Known limitations

Stated up front rather than buried:

- **Priority prediction is weak** (0.5240). Real priority depends on SLA,
  customer tier and business impact — none of which are in the ticket text.
  This is a data problem, not a tuning problem.
- **Queue labels genuinely overlap.** The dominant error is bidirectional
  confusion between `customer_general` and `technical_product_support`. This
  caps every model tested and is the strongest open lead.
- **Escalation risk is rule-derived, not observed.** The dataset has no real
  escalation outcomes, so risk labels are policy heuristics. They should not be
  presented as ground truth.
- **Auto-routing coverage is low at high precision.** At a 0.65 confidence
  threshold the v1 baseline auto-routes about 1% of tickets. Raising coverage
  without losing precision is the core product problem.
- English only.

## Roadmap

Ordered by expected value, not by ease.

1. **Test the label-noise hypothesis.** Hand-label 50 of the
   `customer_general` / `technical_product_support` disagreements blind. If the
   labels are ambiguous to a human, every reported ceiling is a dataset
   artifact and the 73% partly measures memorized convention. Costs an hour.
2. **Record the 10-class ModernBERT split.** It is the one number in the table
   whose provenance is unknown, and it anchors the taxonomy comparison.
3. **Re-measure the 10-class run** so the taxonomy delta rests on a recorded
   split rather than an assumed one.
4. **Add metadata features** (`ticket_type`, `tags`, channel) — the current
   models use text alone and priority especially needs more.
5. **Raise auto-route coverage** at fixed precision. This is the number that
   determines whether the system saves anyone time.

## Running it

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Dataset regeneration and training entry points are documented in
[verticals/ticket-intelligence/README.md](verticals/ticket-intelligence/README.md).

## Intended use

Decision support for human operators. Not autonomous action. The system does
not make final decisions on refunds, legal or compliance matters, account
enforcement, or anything customer-facing without review — see the
[model card](verticals/ticket-intelligence/ml/ticket_intelligence/MODEL_CARD.md).
